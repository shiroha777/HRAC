# HRAC (History-Residual Adaptive Clipping) 方法详解

## 一、方法概述

**HRAC (History-Residual Adaptive Clipping)** 是一种针对联邦学习中拜占庭攻击的鲁棒聚合方法。其核心设计理念是：

1. **只抑制当轮攻击影响，不做长期权重压制**：不标记“坏客户端”，只在本轮降低可疑更新的影响
2. **每个客户端只与自己的历史比较**：对 Non-IID 数据友好，不会误伤数据分布不同的诚实客户端
3. **自适应阈值**：基于每个客户端的历史更新模式动态调整裁剪阈值

---

## 二、核心数据结构（每个客户端维护的历史状态）

HRAC 为每个客户端 `cid` 维护以下历史状态：

| 变量 | 含义 | 更新方式 |
|------|------|----------|
| `b[cid]` | **长期偏置 (Bias)**：客户端更新方向的长期均值 | EMA: `b_new = ρ_b * b_old + (1-ρ_b) * delta_safe` |
| `h[cid]` | **残差方向 (Residual Direction)**：残差的 EMA 方向 | EMA: `h_new = ρ_h * h_old + (1-ρ_h) * r_bar` |
| `μ[cid]` | **残差范数基线 (Residual Norm Baseline)**：残差范数的 EMA | EMA: `μ_new = ρ_μ * μ_old + (1-ρ_μ) * ||r_bar||` |
| `ν[cid]` | **变化基线 (Change Baseline)**：相邻轮残差变化的 EMA | EMA: `ν_new = ρ_ν * ν_old + (1-ρ_ν) * d`，其中 `d = ||r_bar - r_prev||` |
| `r_prev[cid]` | **上一轮裁剪后的残差**：用于计算 `ν` | 每轮更新：`r_prev = r_bar` |

**EMA (Exponential Moving Average)** 参数：
- `ρ_b = 0.98`：偏置更新很慢（长期记忆）
- `ρ_μ = 0.95`：残差范数基线更新较慢
- `ρ_h = 0.95`：残差方向更新较慢
- `ρ_ν = 0.9`：变化基线更新较快（对变化更敏感）

---

## 三、主要方法详解

### 3.1 `__init__()` - 初始化方法

**功能**：初始化 HRAC 聚合器的所有超参数和历史状态字典。

**关键参数**：

```python
def __init__(self, honest_nodes, byzantine_nodes,
             rho_b=0.98, rho_mu=0.95, rho_h=0.95, rho_nu=0.9,
             c=2.5, c_g=3.0,
             enable_logging=True, log_interval=100, eps=1e-12, log_file=None,
             nu_penalty_start_iter=200, nu_penalty_alpha=5.0):
```

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `rho_b` | 0.98 | 偏置 `b` 的 EMA 衰减率（越大越慢） |
| `rho_mu` | 0.95 | 残差范数基线 `μ` 的 EMA 衰减率 |
| `rho_h` | 0.95 | 残差方向 `h` 的 EMA 衰减率 |
| `rho_nu` | 0.9 | 变化基线 `ν` 的 EMA 衰减率 |
| `c` | 2.5 | 客户端自适应裁剪倍数：`τ = c * μ` |
| `c_g` | 3.0 | 全局中位数范数上限倍数：`B = c_g * median(||Δ_i||)` |
| `nu_penalty_start_iter` | 200 | 从第几轮开始启用基于 `ν` 的权重惩罚 |
| `nu_penalty_alpha` | 5.0 | `ν` 惩罚强度：`weight = exp(-alpha * max(0, ν-1))` |
| `nu_weight_max` | 0.30 | 单客户端权重上限（防止权重过度集中） |
| `agg_norm_cap_mult` | 1.5 | 聚合更新范数上限倍数：`||g_t|| ≤ 1.5 * B` |

**初始化状态**：
- 所有历史字典（`b`, `h`, `μ`, `ν`, `r_prev`）初始为空字典，在首次遇到客户端时初始化

---

### 3.2 `_clip_by_norm(r, tau)` - 范数裁剪辅助方法

**功能**：将向量 `r` 裁剪到范数不超过 `tau`。

**实现**：
```python
n = ||r|| + eps
scale = min(1, tau / n)
return r * scale
```

**作用**：如果 `||r|| > tau`，则按比例缩小到 `tau`；否则保持不变。

---

### 3.3 `run(messages, client_ids)` - 核心聚合方法

这是 HRAC 的核心方法，每轮调用一次。完整流程如下：

#### **阶段 1：全局鲁棒裁剪（防御 MSA 攻击）**

```python
norms = ||messages||  # 每个客户端的更新范数
median_norm = median(norms)  # 中位数范数（鲁棒统计量）
B = c_g * median_norm  # 全局上限
messages_clipped = clip(messages, B)  # 所有更新裁剪到 B
```

**目的**：
- 防止攻击者通过发送超大范数更新来控制阈值
- 使用**中位数**而非均值，对异常值鲁棒

---

#### **阶段 2：逐客户端处理（残差去偏 + 自适应裁剪）**

对每个客户端 `i`：

**Step A: 计算残差（去除长期偏置）**
```python
r = delta_t_i - b[i]  # 当前更新减去历史偏置
```

**为什么去偏？**
- 攻击者可能通过多轮小更新逐步“拖拽”模型方向
- 去除偏置后，残差 `r` 反映的是**本轮相对于历史的变化**，更容易检测异常

**Step B: 自适应裁剪残差**
```python
τ = c * μ[i]  # 阈值 = 倍数 × 该客户端的历史残差范数基线
r_bar = clip(r, τ)  # 裁剪残差
```

**为什么自适应？**
- 每个客户端的数据分布不同（Non-IID），正常更新范数也不同
- 用客户端自己的历史 `μ[i]` 作为参考，避免误伤

**Step C: 加回偏置**
```python
delta_tilde = b[i] + r_bar  # 裁剪后的残差加回偏置
delta_tilde = clip(delta_tilde, B)  # 再次用全局 B 限制（防止偏置漂移）
```

**为什么加回偏置？**
- `delta_tilde` 是最终参与聚合的更新
- 保留偏置信息，但通过全局 `B` 防止偏置被攻击者拖拽

---

#### **阶段 3：基于 `ν` 的权重分配（防御 Label Flipping 等攻击）**

**`ν` 的含义**：
- `ν[i] = EMA(||r_bar - r_prev||)`：相邻轮残差变化的 EMA
- 如果客户端更新**变化剧烈**（如 Label Flipping），`ν` 会较大
- 如果更新**稳定**（如正常训练），`ν` 较小

**权重计算**（迭代 > `nu_penalty_start_iter` 后启用）：

```python
# 1. 计算原始权重（指数惩罚）
nu_excess = max(0, ν[i] - 1.0)  # 只惩罚超过 1 的部分
raw_weight[i] = exp(-alpha * nu_excess)
# 例如：ν=1.2, alpha=5 → weight ≈ 0.30
#      ν=0.8, alpha=5 → weight = 1.0（不惩罚）

# 2. 归一化
weights = raw_weights / sum(raw_weights)

# 3. 迭代式权重上限（防止权重过度集中）
while any(weights > nu_weight_max):
    weights = clip(weights, max=nu_weight_max)
    weights = weights / sum(weights)  # 重新归一化
```

**为什么需要权重上限？**
- 如果很多诚实客户端 `ν > 1` 被惩罚，少数客户端（可能含攻击者）会获得大部分权重
- 权重上限（0.30）确保每轮至少约 4 个客户端有非小权重，防止被 1-2 个坏客户端主导

**聚合**：
```python
g_t = Σ(weights[i] * delta_tilde[i])  # 加权平均
```

**聚合更新范数上限**：
```python
if ||g_t|| > 1.5 * B:
    g_t = g_t * (1.5 * B / ||g_t||)  # 裁剪到 1.5*B
```

**为什么限制 `||g_t||`？**
- 即使权重分配不理想，单轮更新也不会过大
- 防止一步更新把模型推到极差区域（导致 acc 骤降）

---

#### **阶段 4：更新历史状态**

对每个客户端（跳过新初始化的）：

**更新 `b`（偏置）**：
```python
delta_safe = b[i] + r_bar  # 用于更新偏置的“安全更新”
delta_safe = clip(delta_safe, B)  # 用全局 B 限制
b[i] = ρ_b * b[i] + (1-ρ_b) * delta_safe  # EMA 更新
```

**更新 `μ`（残差范数基线）**：
```python
μ[i] = ρ_μ * μ[i] + (1-ρ_μ) * ||r_bar||
μ[i] = max(μ[i], μ_min)  # 防止塌陷
```

**更新 `h`（残差方向）**：
```python
h[i] = ρ_h * h[i] + (1-ρ_h) * r_bar  # EMA 方向
```

**更新 `ν`（变化基线）**：
```python
d = ||r_bar - r_prev[i]||  # 相邻轮残差变化

# 限制 d 的范围（防止异常值污染 ν）
d_max_global = median(ν) * 100  # 全局参考
d_max_local = ν[i] * 5.0  # 本地参考（基于客户端自身历史）
d_max = min(d_max_global, d_max_local)  # 取更严格的限制
d_min = ν[i] * 0.01  # 防止突然塌陷
d_capped = clamp(d, min=d_min, max=d_max)

# EMA 更新
ν[i] = ρ_ν * ν[i] + (1-ρ_ν) * d_capped
ν[i] = max(ν[i], ν_min)
```

**为什么限制 `d`？**
- 防止 `r_bar` 突然变化（如梯度爆炸或攻击）导致 `ν` 突变
- 使用全局和本地双重参考，更鲁棒

**更新 `r_prev`**：
```python
r_prev[i] = r_bar  # 保存本轮残差，下轮用于计算 d
```

---

### 3.4 `_log_statistics()` - 日志记录方法

**功能**：记录每轮的统计信息，用于调试和分析。

**记录内容**：
- 全局裁剪：`median_norm`, `B`, 裁剪比例
- 各客户端：`τ`, `μ`, `ν`, 权重, 范数
- 聚合更新：`||g_t||`, `w_max`, 是否被裁剪
- 权重分布：均值、最小值、最大值

**输出位置**：
- 控制台（如果 `enable_logging=True`）
- 文件（如果 `log_file` 指定）

---

## 四、关键设计思想总结

### 4.1 为什么“只抑制当轮，不做长期压制”？

- **避免误伤**：在 Non-IID 下，诚实客户端的数据分布可能差异很大，长期压制会误伤
- **灵活性**：允许客户端在不同轮次有不同的表现，不因一次异常就永久降权

### 4.2 为什么“每个客户端只与自己的历史比较”？

- **Non-IID 友好**：不同客户端的数据分布不同，正常更新范数也不同
- **公平性**：不因数据分布差异而惩罚客户端

### 4.3 防御机制

| 攻击类型 | 防御机制 |
|---------|---------|
| **MSA (Momentum Stealing Attack)** | 全局中位数范数上限 `B`，防止攻击者控制阈值 |
| **Label Flipping** | 基于 `ν` 的权重惩罚：变化剧烈的客户端（`ν > 1`）权重降低 |
| **Scaling Attack** | 全局 `B` + 客户端自适应 `τ` 双重裁剪 |
| **Bias Drift Attack** | 全局 `B` 限制 `delta_tilde` 和 `b` 的更新，防止偏置被拖拽 |

### 4.4 防止突变（Anti-Mutation）机制

1. **权重上限**：迭代式 cap，确保 `w_max ≤ 0.30`
2. **聚合更新上限**：`||g_t|| ≤ 1.5 * B`
3. **`d` 的限制**：全局和本地双重参考，防止 `ν` 突变
4. **`μ` 和 `ν` 的最小值**：防止阈值塌陷

---

## 五、算法流程总结

```
输入: messages (N, D) - 所有客户端的原始更新

1. 全局裁剪: messages_clipped = clip(messages, B = c_g * median(||messages||))

2. 对每个客户端 i:
   a. 残差: r = messages_clipped[i] - b[i]
   b. 自适应裁剪: r_bar = clip(r, τ = c * μ[i])
   c. 加回偏置: delta_tilde[i] = clip(b[i] + r_bar, B)

3. 权重分配（如果 iter > nu_penalty_start_iter）:
   a. raw_weight[i] = exp(-alpha * max(0, ν[i] - 1))
   b. weights = normalize(raw_weights)
   c. weights = iterative_cap(weights, max=0.30)

4. 聚合: g_t = Σ(weights[i] * delta_tilde[i])
5. 限制: g_t = clip(g_t, 1.5 * B)

6. 更新历史: b[i], μ[i], h[i], ν[i], r_prev[i]

输出: g_t - 聚合后的更新
```

---

## 六、超参数调优建议

| 参数 | 调优方向 | 影响 |
|------|---------|------|
| `c` | 增大 → 更宽松的残差裁剪 | 可能放过攻击，但减少误伤 |
| `c_g` | 增大 → 更宽松的全局上限 | 对 MSA 防御变弱 |
| `nu_penalty_alpha` | 增大 → 更严厉的 `ν` 惩罚 | 更有效抑制 Label Flipping，但可能误伤 Non-IID |
| `nu_weight_max` | 减小 → 权重更分散 | 更安全，但可能降低对攻击的抑制 |
| `agg_norm_cap_mult` | 减小 → 更保守的聚合更新 | 更稳定，但可能收敛变慢 |

---

## 七、与现有方法的对比

| 方法 | 特点 | HRAC 的优势 |
|------|------|------------|
| **Mean** | 简单平均 | 对攻击无防御 |
| **Trimmed Mean** | 去除极端值 | 需要硬阈值，Non-IID 下易误伤 |
| **FABA** | 基于距离的过滤 | 需要全局比较，Non-IID 不友好 |
| **CC (Centered Clipping)** | 固定阈值裁剪 | 阈值不自适应，Non-IID 下效果差 |
| **HRAC** | 历史-残差自适应裁剪 | ✅ 自适应阈值<br>✅ Non-IID 友好<br>✅ 只抑制当轮，不做长期压制 |

---

## 八、实验建议

1. **IID vs Non-IID**：HRAC 在 Non-IID 下表现更好（因为与自身历史比较）
2. **不同攻击**：测试 MSA、Label Flipping、Scaling 等
3. **超参数敏感性**：测试 `c`, `nu_penalty_alpha` 等的影响
4. **收敛速度**：与 Mean 等基线方法对比收敛速度

---

## 九、常见问题

**Q: 为什么 `ν` 初始化为 `median_norm` 而不是 0？**  
A: 防止新客户端在第一轮就被过度惩罚。用全局中位数作为合理初始值。

**Q: 为什么需要全局 `B` 和客户端 `τ` 双重裁剪？**  
A: `B` 防御 MSA（防止攻击者控制阈值），`τ` 适应 Non-IID（每个客户端有自己的正常范围）。

**Q: 权重上限为什么用迭代式而不是一次 cap？**  
A: 一次 cap + 归一化后，最大权重可能仍超过上限。迭代式确保最终 `w_max ≤ 0.30`。

**Q: 为什么 `ν` 更新时要限制 `d`？**  
A: 防止 `r_bar` 突然变化（梯度爆炸或攻击）导致 `ν` 突变，进而导致权重分配突变。

---

## 十、代码结构总结

```
C_HRAC
├── __init__()              # 初始化超参数和历史字典
├── _clip_by_norm()         # 范数裁剪辅助方法
├── _cos()                  # 余弦相似度（未使用，保留接口）
├── run()                   # 核心聚合方法
│   ├── 全局裁剪
│   ├── 逐客户端处理（残差去偏 + 自适应裁剪）
│   ├── 权重分配（基于 ν）
│   ├── 聚合 + 范数限制
│   └── 更新历史状态
└── _log_statistics()       # 日志记录
```

---

**文档版本**: v1.0  
**最后更新**: 2025-01-27
