# HSM-Adaptive 方法详解

## 一、核心思想

**HSM-Adaptive (History-Soft Momentum Adaptive)** 是一个**无需先验信息**的鲁棒联邦学习聚合方法，通过**历史一致性**和**结构一致性**来识别并抑制恶意更新。

### 关键设计原则

1. **无先验信息 (No Oracle)**: 不使用 `honest_nodes` 或 `byzantine_nodes` 信息进行防御判断
2. **软权重 (Soft Weighting)**: 不硬过滤客户端，而是通过权重 soft-down-weight 可疑更新
3. **自适应 (Adaptive)**: 所有阈值和权重基于数据分布自动确定，无固定超参
4. **历史累积 (History Accumulation)**: 通过 EMA 累积客户端的历史行为，长期攻击者会逐渐暴露

---

## 二、算法流程（逐步解析）

### Step 0: 自适应模长裁剪 (Adaptive Norm Clipping)

```python
norms = ||Δ_i||  # 每个客户端的更新模长
median_norm = median(norms)
clip_scales = min(1, median_norm / norms)  # 只缩小，不放大
Δ'_i = Δ_i * clip_scales
```

**作用**: 
- 限制极端模长的更新（如 MSA 攻击常产生超大模长）
- 使用中位数而非固定阈值，适应不同的学习率/任务

---

### Step A: 参考方向选择 (Reference Direction)

```python
if last_agg is not None:
    ref_direction = last_agg  # 使用上一轮聚合结果
else:
    ref_direction = server_momentum  # 初始使用服务器动量
```

**设计要点**:
- **不混入 coordinate-wise median**：避免被本轮异常更新污染
- 参考方向 = 历史真实聚合方向，代表"正确的训练方向"

---

### Step B: 特征提取 (Feature Extraction)

#### 1. 全局对齐 (Global Alignment)

```python
a_i = cos(Δ'_i, ref_direction)  # 余弦相似度
a_centered_i = a_i - median(a_j)  # 相对中位数
```

**含义**:
- `a_i` 高 → 客户端更新与全局方向一致 → 更可信
- `a_centered_i` → 相对于其他客户端的对齐优势（Non-IID 保护）

#### 2. 尺度异常 (Scale Anomaly)

```python
r_prev_i = EMA(||Δ_i||)  # 历史模长的指数移动平均
scale_dev_i = |log(||Δ'_i|| / r_prev_i)|  # log 比例
```

**含义**:
- 检测客户端更新模长是否突然变化
- Label Flipping 常导致更新模长不稳定
- Log 比例对 Non-IID 的"固定大/固定小"不敏感

---

### Step C: 信任感知权重 (Trust-Aware Weighting, TAW)

```python
# 历史对齐累积
T_i = rho * T_{i,t-1} + (1-rho) * a_i

# 归一化到相对尺度
T_med = median(T_j)
T_mad = MAD(T_j)
T_z_i = (T_i - T_med) / T_mad
w_hist_i = sigmoid(T_z_i)  # 历史权重
```

**核心思想**:
- **长期对齐优势** → 高权重
- **长期偏离** → 低权重（Label Flipping 会逐渐被识别）
- 使用相对 z-score + sigmoid，避免硬阈值

---

### Step D: 符号草图一致性 (Sign-Sketch Consistency)

```python
# CountSketch: 将高维更新投影到低维（256维）
sketch_i = CountSketch(Δ'_i)  # [256]
sketch_sign_i = sign(sketch_i)  # {+1, -1}

# 共识符号
consensus = sign(mean(sketch_sign_j))  # 多数客户端符号

# 一致率
agree_rate_i = mean(sketch_sign_i == consensus)
w_sketch_i = clamp(2 * agree_rate_i - 1, 0, 1)  # 0.5→0, 1.0→1
```

**核心思想**:
- MSA/HisMSA 攻击会**打乱坐标位置**，导致 sketch 符号模式随机化
- 正常客户端 → 符号模式一致 → `agree_rate` 高
- 攻击者 → 符号模式随机 → `agree_rate` 低（接近 0.5）

**为什么有效**:
- 只看符号（±1），对缩放不敏感
- 低维投影（256维）降低计算和存储成本
- 固定 CountSketch（h, s 固定），减少随机性

---

### Step E: 权重组合 (Weight Combination)

```python
# 组合权重
weights_i = w_hist_i * w_sketch_i * scale_gate_i

# Scale gate: 轻量惩罚
scale_gate_i = 1 / (1 + scale_dev_i)  # z=0→1, z=3→0.25
```

**设计**:
- **乘法组合**：三个指标都高 → 高权重
- **Scale gate 是软惩罚**：不直接砍掉，只是降权重
- 所有权重加 `+1e-3` 保证非零（避免完全排除）

---

### Step F: 聚合与更新 (Aggregation & Update)

```python
# 加权平均聚合
aggregated = Σ(weights_i * Δ'_i) / Σ(weights_i)

# 更新历史状态
server_momentum = beta * server_momentum + (1-beta) * aggregated
last_agg = aggregated
client_history_c[i] = rho * client_history_c[i] + (1-rho) * a_i
client_history_r[i] = rho * client_history_r[i] + (1-rho) * ||Δ'_i||
```

---

## 三、对抗不同攻击的策略

### 1. Label Flipping (LF)

**攻击特点**: 
- 更新方向可能不明显异常（尤其早期）
- 更新模长可能不稳定

**防御机制**:
- **TAW (w_hist)**: 长期累积发现方向不一致 → 逐渐降权重
- **Scale Anomaly (scale_gate)**: 模长不稳定 → 惩罚

### 2. MSA (Model Shuffling Attack)

**攻击特点**:
- 打乱参数坐标位置
- 模长和方向可能正常

**防御机制**:
- **Sign-Sketch (w_sketch)**: 坐标打乱 → sketch 符号随机 → `agree_rate` 低 → 权重低
- 这是**唯一能有效对抗 MSA** 的机制

### 3. HisMSA (Historical MSA)

**攻击特点**:
- 缓慢漂移（慢于单轮检测）
- 长期累积导致模型偏移

**防御机制**:
- **TAW (w_hist)**: 长期历史发现方向逐渐偏离 → 累积惩罚
- **Sign-Sketch**: 每轮的坐标打乱都会被检测

---

## 四、关键设计选择

### 1. 为什么用 `last_agg` 而不是 `coordinate-wise median`?

```
❌ coordinate-wise median: 高维参数空间下不等于几何中值，可能被异常坐标污染
✅ last_agg: 代表真实的聚合方向，更可靠
```

### 2. 为什么用 Log-Ratio 而不是 MAD Z-Score?

```
❌ MAD Z-Score: 对 Non-IID 的"固定大/固定小"可能误判
✅ Log-Ratio: 检测"变化"，而非"绝对值差异"
```

### 3. 为什么用 CountSketch 而不是全维度?

```
❌ 全维度: 计算 O(N*D)，内存 O(N*D)
✅ CountSketch: 计算 O(N*D)，内存 O(N*256)，符号模式足够识别结构异常
```

### 4. 为什么用乘法组合权重?

```
w = w_hist * w_sketch * scale_gate

含义: 三个指标都高 → 可信；任一指标低 → 降权重
```

### 5. 为什么保留 `+1e-3` 最小权重?

```
保证软权重: 即使可疑也不完全排除（避免误伤 Non-IID 客户端）
```

---

## 五、与 Recall 的关系

**Recall 计算**（用于评估效果）:

```python
# 预测: 权重最低的 k 个客户端
predicted_attackers = topk(weights, k=num_attackers, largest=False)

# Recall = 正确识别的攻击者 / 总攻击者
recall = |predicted_attackers ∩ true_attackers| / |true_attackers|
```

**要点**:
- Recall 基于**实际权重**，直接反映防御机制的效果
- 权重低 → 被 down-weight → 被识别为攻击者（predicted）
- **权重 = 防御机制的实际输出**，所以 recall 真实反映方法有效性

---

## 六、参数设置（最小化超参）

```python
rho = 0.9          # 历史 EMA 衰减（标准）
beta = 0.9         # 服务器动量衰减（标准）
sketch_dim = 256   # CountSketch 维度（固定，无需调）
tau_quantile = 0.3 # 仅用于日志参考，不影响聚合
```

**无硬编码超参**:
- `w_hist`, `w_sketch`, `scale_gate` 都是自适应计算的
- 阈值基于数据分布的 quantile/MAD，无需手动设定

---

## 七、计算复杂度

- **内存**: O(N*D + N*256) ≈ O(N*D)（主要来自消息存储）
- **计算**: 
  - CountSketch: O(N*D)（每个元素一次 scatter_add）
  - 其他: O(N*D)（对齐、权重计算等）
  - 总计: O(N*D)，与 Mean 相同

---

## 八、总结

HSM-Adaptive 通过**三个互补的机制**实现鲁棒聚合：

1. **TAW (Trust-Aware Weighting)**: 长期方向一致性 → 对抗 Label Flipping
2. **Sign-Sketch**: 结构一致性 → 对抗 MSA/HisMSA
3. **Scale Gate**: 尺度稳定性 → 辅助检测不稳定更新

**核心优势**:
- ✅ 无需先验信息（no oracle）
- ✅ 软权重，Non-IID 友好
- ✅ 自适应阈值，超参少
- ✅ 计算高效（O(N*D)）
- ✅ 对 MSA 特别有效（Sign-Sketch 独有）
