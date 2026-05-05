# HSM-Adaptive 权重误判分析

## 三个主要权重组件

### 1. **w_hist** (历史信任权重 - TAW)
- **计算位置**: 第960行
- **公式**: `w_hist = sigmoid(T_z) + 1e-3`
- **依赖**: 
  - `T_z = (T_values - T_med) / T_mad`
  - `T_values` = EMA历史对齐分数（基于`align_centered`）
  - **关键参数**: `rho=0.9` (历史衰减系数)

**误判场景**:
- ❌ **攻击者伪装对齐**: 如果攻击者长期伪装成"好客户端"（`align_centered`接近中位数），`T_values`会逐渐累积为正，导致`w_hist`偏高
- ❌ **历史污染**: `rho=0.9`意味着历史权重很高，一旦攻击者早期"养肥"了历史信任，后期即使行为异常也难以快速降低权重
- ❌ **门控因子失效**: `gate_factor = clamp(prev_w_self, min=0.3)`，如果`w_self`本身被攻击者欺骗，门控也无法阻止历史污染

**诊断方法**:
```python
# 查看日志中的 T_values 分布
# 如果攻击者的 T_values 接近或高于 honest 客户端 → w_hist 误判
```

---

### 2. **w_sketch** (符号-草图一致性权重)
- **计算位置**: 第1007行
- **公式**: `w_sketch = clamp(2.0 * agree_rate - 1.0, 0.0, 1.0) + 1e-3`
- **依赖**: 
  - `agree_rate` = 客户端签名与共识签名的一致率
  - CountSketch投影的符号模式

**误判场景**:
- ❌ **MSA攻击绕过**: 如果攻击者知道CountSketch映射（白盒/灰盒），可以构造更新使得`sketch_sign`与共识对齐，`agree_rate`会很高
- ❌ **HisMSA慢漂移**: 攻击者缓慢改变更新模式，`agree_rate`可能不会立即下降
- ❌ **IID场景失效**: 在IID数据下，所有客户端更新方向相似，攻击者的`agree_rate`可能也很高

**诊断方法**:
```python
# 查看日志中的 agree_rate
# 如果攻击者的 agree_rate > 0.5 → w_sketch 误判
# 如果攻击者的 agree_rate 接近 honest → 说明攻击者成功伪装了结构
```

---

### 3. **w_self** (自一致性权重)
- **计算位置**: 第1050行
- **公式**: `w_self = sigmoid(self_align_z) + 1e-3`
- **依赖**:
  - `self_align_z = (self_alignments_sketch - self_align_med) / self_align_mad`
  - `self_alignments_sketch` = 当前轮sketch签名与历史EMA的余弦相似度

**误判场景**:
- ❌ **早期轮次**: 新客户端或sketch映射更新后，`self_alignments_sketch = 1.0`（默认值），所有客户端`w_self`都接近1.0，失去区分力
- ❌ **稳定攻击**: 如果攻击者保持一致的攻击模式（如固定方向的label flipping），`self_alignments_sketch`可能仍然很高
- ❌ **历史EMA污染**: 如果攻击者早期成功伪装，历史EMA被污染，后续即使行为异常，与历史EMA的相似度仍可能较高

**诊断方法**:
```python
# 查看日志中的 self_alignments (Self-Consistency Scores)
# 如果攻击者的 self_alignments 接近 1.0 → w_self 误判
# 如果攻击者的 self_alignments 接近 honest → 说明攻击模式稳定/伪装成功
```

---

## 权重组合流程

```python
# 第1057行: 基础权重
weights = w_hist * w_sketch * w_self

# 第1061行: 尺度门控
scale_gate = 1.0 / (1.0 + scale_z_scores)
weights = weights * scale_gate + 1e-3

# 第1106行: 信任门控
weights = weights * torch.exp(eta * trust_z)

# 第1111行: Neff保护
weights = torch.pow(weights, p) + 1e-3
```

## 误判诊断流程

### 步骤1: 检查基础权重
```python
# 如果攻击者权重高，检查哪个组件贡献最大
if attacker_weight > honest_weight:
    if attacker_w_hist > honest_w_hist:
        print("❌ w_hist 误判: 历史信任被污染")
    if attacker_w_sketch > honest_w_sketch:
        print("❌ w_sketch 误判: 结构一致性被欺骗")
    if attacker_w_self > honest_w_self:
        print("❌ w_self 误判: 自一致性被伪装")
```

### 步骤2: 检查门控效果
```python
# 检查 scale_gate 和 trust_gate 是否生效
if attacker_scale_gate > 0.5:
    print("⚠️ scale_gate 未有效降低攻击者权重")
if attacker_trust_z > -1.0:
    print("⚠️ trust_gate 未有效惩罚攻击者")
```

## 常见误判模式

### 模式1: **历史污染型误判** (w_hist 问题)
- **症状**: 攻击者早期伪装，后期即使行为异常，权重仍然较高
- **原因**: `rho=0.9`太高，历史权重衰减太慢
- **解决**: 降低`rho`到`0.8`或`0.85`，或增加`gate_factor`的惩罚

### 模式2: **结构伪装型误判** (w_sketch 问题)
- **症状**: 攻击者的`agree_rate`接近或高于honest客户端
- **原因**: 攻击者成功构造了与共识对齐的更新
- **解决**: 增加sketch映射的随机更新频率（当前2%可能不够），或增加`sketch_dim`

### 模式3: **稳定攻击型误判** (w_self 问题)
- **症状**: 攻击者保持一致的攻击模式，`self_alignments`很高
- **原因**: 攻击模式稳定，与历史EMA相似度高
- **解决**: 降低`rho_sketch`（当前等于`rho=0.9`），或增加对异常`self_align`的惩罚

## 参数调整建议

### 如果 w_hist 误判:
```python
# 降低历史权重
rho = 0.85  # 从0.9降到0.85
# 或增加门控惩罚
gate_factor = clamp(prev_w_self, min=0.2)  # 从0.3降到0.2
```

### 如果 w_sketch 误判:
```python
# 增加sketch映射更新频率
should_update_sketch = torch.rand(()) < 0.05  # 从2%提高到5%
# 或增加sketch维度
sketch_dim = 512  # 从256提高到512
```

### 如果 w_self 误判:
```python
# 降低sketch历史权重
rho_sketch = 0.8  # 从0.9降到0.8
# 或增加异常惩罚
self_align_z = clamp(self_align_z, -5.0, 5.0)  # 扩大惩罚范围
```

## 日志检查清单

在日志中检查以下指标：

1. **w_hist 检查**:
   - `T_values` (历史信任分数): 攻击者是否接近honest？
   - `w_hist`: 攻击者是否 > 0.5？

2. **w_sketch 检查**:
   - `agree_rate`: 攻击者是否 > 0.5？
   - `w_sketch`: 攻击者是否接近honest？

3. **w_self 检查**:
   - `self_alignments`: 攻击者是否接近1.0？
   - `w_self`: 攻击者是否 > 0.5？

4. **最终权重**:
   - `weights`: 攻击者是否在最低k个权重中？
   - `detection_recall`: 是否 < 100%？
