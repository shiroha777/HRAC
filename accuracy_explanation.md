# Accuracy计算说明

## 当前代码中的Accuracy是什么？

### 计算位置
- **文件**: `ByrdLab/centraliedAlgorithm.py` 第1426-1429行
- **函数**: `one_node_loss_accuracy_dist()`

### 关键信息

```python
test_loss, test_accuracy = one_node_loss_accuracy_dist(
    server_model,           # ← 全局模型（聚合后的模型）
    self.get_test_iter,     # ← 测试集迭代器
    self.loss_fn, 
    self.test_fn,
    weight_decay=0, 
    node_list=self.honest_nodes
)
```

### 详细分析

1. **模型**: `server_model` - **全局模型**（聚合后的模型）
2. **数据集**: `get_test_iter()` - **测试集**（不是训练集）
3. **模式**: `model.eval()` - **评估模式**（不是训练模式）
4. **时机**: 每 `display_interval` 次迭代（每100次迭代）计算一次

### 结论

**这个accuracy是：**
- ✅ **全局模型在测试集上的准确率**
- ✅ **不是训练时的准确率**
- ✅ **不是客户端本地模型的准确率**
- ✅ **这是HisMSA攻击要破坏的目标**

### 为什么这很重要？

HisMSA攻击的目标就是：
- 降低**全局模型**的**测试准确率**
- 所以看这个accuracy是正确的指标

### 当前结果分析

- **HisMSA**: accuracy ≈ 0.88
- **基线（label_flipping）**: accuracy ≈ 0.63

如果基线是0.63，HisMSA是0.88，说明：
- **HisMSA攻击效果比label_flipping还弱**（准确率更高）
- 这可能是因为：
  1. Step2过度压缩（gamma太小）
  2. Step1在CNN上效果不够强
  3. 需要更激进的参数

### 建议

关闭Step2后，如果accuracy明显下降（比如降到0.7以下），说明：
- Step1是有效的
- 问题在Step2的边界估计

如果关闭Step2后accuracy仍然很高（>0.85），说明：
- Step1可能太弱
- 需要更激进的参数或不同的攻击策略

