# HRAC（History-Residual Adaptive Clipping）方法详解与数据来源

## 一、方法概述

HRAC 是一种**拜占庭鲁棒的联邦聚合**方法：只根据每个客户端自己的**历史行为**做裁剪和加权，不依赖“谁是诚实/恶意”的标签，对 Non-IID 友好。

核心思路：
1. **全局范数帽**：用本轮所有客户端更新范数的**中位数**定一个上界，防止少数极大更新主导。
2. **每客户端残差裁剪**：相对该客户端自己的**历史均值 b** 算残差，用**自适应阈值 τ** 裁剪残差，再与 b 相加得到“安全更新”。
3. **基于 ν 的加权**：用**变化基线 ν**（刻画该客户端本轮相对历史的“变化幅度”）做惩罚，ν 越大权重越小，聚合时对异常客户端降权。

下面按**数据来源**逐项说明。

---

## 二、输入与全局量（每轮一开始就有）

### 1. `messages`（形状 N×D）

- **含义**：本轮 N 个客户端上传的**原始更新向量**（例如梯度或 momentum）。
- **来源**：主循环里各节点本地算出的更新，拼成 `(N, D)` 的 tensor 传入 `aggregation.run(messages)`。

### 2. `norms`、`median_norm`、`B`

- **norms**：‖messages[i]‖_2，每个客户端更新的 L2 范数。
- **median_norm**：norms 的**中位数**，用于定“正常尺度”。
- **B**：B = c_g·median_norm，**全局范数上界**（c_g 默认 3.0）。  
- **用途**：  
  - 后面所有“相对 B 的缩放”都用这个 B；  
  - 若启用 scale anomaly 检测，还会和上一轮的 `median_norm_prev` 比较。

### 3. `messages_clipped`（全局 cap）

- **计算**：对每个客户端 i，若 ‖messages[i]‖ > B，则缩放到范数 B：  
  scale_global[i] = min(1, B/norms[i])，messages_clipped = messages * scale_global。
- **含义**：**范数不超过 B** 的版本，用于后续**聚合路径**（算 b、残差、processed 等）。

### 4. `messages_eff`（仅用于统计路径，存在小范数时）

- **条件**：若存在 `norms[i] < 0.5 * median_norm`，则对这些客户端启用 “lift”。
- **计算**：对小范数客户端，把其向量**按比例放大**到范数等于 `median_norm`，得到 `messages_eff[i]`；其余客户端用 `messages_clipped[i]`。
- **用途**：只用于**统计路径**里计算 τ、μ、ν、b_norm、r_prev 等，**不参与最终聚合**；聚合仍用 `messages_clipped` 和后面得到的 `processed`。

---

## 三、每客户端的持久状态（跨轮保存）

这些在 `run()` 里会读/写，下一轮继续用。

| 变量 | 含义 | 更新方式 |
|------|------|----------|
| **b** | 该客户端在**聚合路径**上的历史均值（EMA） | b ← ρ_b·b + (1−ρ_b)·delta_safe_capped。delta_safe_capped = 当前轮 processed[i] 再按 B 做范数 cap 后的向量（见第四节） |
| **b_norm** | 仅用于**统计路径**的历史均值（小范数时用 scale 后的向量） | b_norm ← ρ_b·b_norm + (1−ρ_b)·delta_tilde_norm |
| **μ (mu)** | “范数基线”：残差/更新范数大小的 EMA | μ ← ρ_μ·μ + (1−ρ_μ)·‖r_bar_norm‖；小范数客户端时用 ‖delta_tilde_norm‖ |
| **ν (nu)** | “变化基线”：本轮相对上轮变化幅度的 EMA | ν ← ρ_ν·ν + (1−ρ_ν)·d_eff，其中 d_eff = ‖r_bar_norm − r_prev‖（可选 d_nu_cap 上界） |
| **r_prev** | 上一轮统计路径上的裁剪后残差（用于算 d） | 每轮末 r_prev ← r_bar_norm |
| **h** | 残差方向的 EMA（当前实现里用于扩展，核心权重主要看 ν） | h ← ρ_h·h + (1−ρ_h)·r_bar_norm |

- **ρ 系数**：`rho_b=0.98`, `rho_mu=0.95`, `rho_h=0.95`, `rho_nu=0.87`（main 里可改），都是**历史权重大、当前观测权重小**的 EMA。

---

## 四、每客户端本轮计算（按顺序）

### 1. 聚合路径（决定最终参与聚合的向量）

- **delta_t_i**：`messages_clipped[i]`（该客户端经全局 cap 后的更新）。
- **残差**：r = delta_t_i − b（相对历史均值 b 的偏差）。
- **τ (tau)**：裁剪阈值，τ = c·μ + ε（c=2.5），用当前 μ 衡量“正常残差大小”。
- **r_bar**：对 r 做**范数裁剪**：若 ‖r‖ > τ 则缩放到长度 τ，否则不变（_clip_by_norm(r, tau)）。
- **delta_tilde**：b + r_bar，再对整体做一次范数 cap：若 ‖delta_tilde‖ > B 则缩放到 B。
- **processed[i]**：就是上面这个 **delta_tilde**，即**该客户端参与加权聚合的向量**。

因此：
- **norm_pre**：‖messages[i]‖，原始上传范数。
- **norm_post**：‖messages_clipped[i]‖，全局 cap 后范数。
- **norm_final**：‖processed[i]‖，**最终参与聚合的向量**的范数。

（若存在“小范数 lift”，τ、μ、ν 会用统计路径的 `delta_tilde_norm`/`r_bar_norm` 等算，但 norm_pre/post/final 仍是聚合路径上的这三个范数。）

### 2. 统计路径（只影响 τ、μ、ν、b_norm、r_prev）

- 若无小范数：用 `r_bar`、`delta_tilde` 当作 `r_bar_norm`、`delta_tilde_norm`。
- 若有小范数：用 `messages_eff[i]` 相对 `b_norm` 的残差，裁剪后得到 `r_bar_norm`、`delta_tilde_norm`。
- **d**：d = ‖r_bar_norm − r_prev‖，表示“本轮的裁剪残差相对上一轮裁剪残差的变化量”。
- **d_eff**：若启用 d_nu_cap_factor，会对 d 做上界限制（如 d_eff = min(d, ν·factor)），否则 d_eff = d。
- **ν 更新**：ν_new = ρ_ν·ν + (1−ρ_ν)·d_eff。  
  即：**ν 是“残差变化量 d”的 EMA**，用来衡量该客户端**相对自己历史是否变化剧烈**（大则更像异常）。

---

## 五、权重与聚合

### 1. 是否使用 ν 惩罚

- 若 iteration_count ≤ nu_penalty_start_iter（默认 50）或本轮被判定为 **scale_abnormal**：**均匀权重** w_i = 1/N。
- 否则：按 ν 算权重（见下）。

### 2. ν 惩罚权重的计算

- **nu_values**：每个客户端当前轮的 ν（新客户端用 `median_norm` 初始化）。
- **nu_mean**：所有客户端 ν 的**平均值**。
- **nu_excess**：对每个 i，若 ν_i > 1 则 excess_i = (ν_i − nu_mean)^+，否则 0。
- **raw_weights**：w_i^raw = 1 / (1 + α·nu_excess_i)，nu_penalty_alpha=5。
- **weights_tensor**：对 raw_weights 做**归一化**（和为 1），再对单客户端权重做**上界 cap**（如 `nu_weight_max=0.30`），再重新归一化。

因此：**ν 越大、且越高于平均 ν，权重越小**；ν≤1 则不受罚。

### 3. 聚合结果 g_t

- g_t = Σ_i w_i · processed[i]。
- 可选：对 `g_t` 再做范数 cap（如不超过 `B * agg_norm_cap_mult` 或历史 `||g_t||` 的 k 倍）；若启用 aggregate momentum，还可对 g_t 做 EMA 再输出。

---

## 六、日志里每个数据对应关系（小结）

| 日志项 | 含义 | 数据来源 |
|--------|------|----------|
| **iter** | 当前轮数 | aggregation.iteration_count（每轮 run 开头 +1） |
| **acc** | 当前准确率 | 主循环在 display_interval 时评估得到，通过 set_accuracy() 写入 |
| **τ (tau)** | 残差裁剪阈值 | τ = c·μ + ε，μ 来自该客户端历史残差范数的 EMA |
| **μ (mu)** | 范数基线 | 统计路径上 ‖r_bar_norm‖（或小范数时 ‖delta_tilde_norm‖）的 EMA |
| **ν (nu)** | 变化基线 | d = ‖r_bar_norm − r_prev‖ 的 EMA（d_eff 的 EMA） |
| **norm_pre** | 原始更新范数 | ‖messages[i]‖ |
| **norm_post** | 全局 cap 后范数 | ‖messages_clipped[i]‖ |
| **norm_final** | 参与聚合的向量范数 | ‖processed[i]‖（残差裁剪 + 加回 b + B cap 后） |
| **weight** | 聚合权重 | 由 ν 和 nu_excess 得到，归一化并 cap 后 |
| **median_norm** | 本轮所有客户端更新范数的中位数 | torch.median(norms) |
| **B / cap** | 全局范数上界 | B = c_g·median_norm |
| **g_norm** | 聚合向量范数 | ‖g_t‖ |
| **w_max / w_min** | 最大/最小客户端权重 | weights_tensor.max() / .min() |

说明：表中“τ, μ, ν: real”表示这些是**实际参与计算的值**；当存在 norm<0.5*median 的客户端时，它们的 τ、μ、ν 是用**放大到 median 的统计路径**（messages_eff / b_norm / r_bar_norm）算出来的，而 norm_pre/post/final 仍是聚合路径上的原始、clip、processed 的范数。

---

## 七、整体数据流简图

```
messages (原始)
    → norms, median_norm, B
    → messages_clipped (全局 cap)
    → [可选] messages_eff (小范数 lift，仅统计用)

对每个客户端 i:
  聚合路径: delta_t_i = messages_clipped[i]
           r = delta_t_i - b[i]
           r_bar = clip(r, τ),  τ = c*μ + ε
           delta_tilde = b[i] + r_bar, 再 cap 到 B
           processed[i] = delta_tilde

  统计路径: (用 delta_t_i 或 messages_eff[i], b_norm[i])
           r_bar_norm, delta_tilde_norm
           d = ||r_bar_norm - r_prev[i]||
           ν_new = ρ_ν*ν + (1-ρ_ν)*d_eff
           μ_new = ρ_μ*μ + (1-ρ_μ)*||r_bar_norm||
           b[i], b_norm[i], r_prev[i] 更新

权重: 由 ν 得到 raw_weights → 归一化 → cap → 再归一化 → weights_tensor
聚合: g_t = Σ weight[i] * processed[i]
```

这样，**每个数据从哪里来、代表什么**都在上面一一对应；你要写论文或调参时，可以直接按“数据来源”这一列对到代码和公式。
