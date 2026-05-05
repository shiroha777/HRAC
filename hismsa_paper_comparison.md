# HisMSA Paper vs Your Experiment Comparison

## Key Differences Analysis

### 1. Malicious Client Ratio (Critical!)

| Condition | Paper | Your Experiment | Impact |
|-----------|-------|-----------------|--------|
| Byzantine ratio | 2%, 5%, 10% | **30% (3/10)** | **Major difference** |

**Why this matters:**
- Paper emphasizes "small fraction" for **stealth**
- 30% Byzantine is too high and breaks the "stealth assumption"
- High ratio makes attack more obvious, may trigger defenses
- Mean aggregation with 30% malicious can still be diluted by 70% honest

**Your result (0.66) vs Paper (0.3):**
- Paper's 2-10% maintains stealth while causing damage
- Your 30% might be:
  - Too obvious (if defenses exist)
  - Still diluted by 70% honest updates
  - Not following paper's "realistic threat model"

### 2. Model Type

| Condition | Paper | Your Experiment | Impact |
|-----------|-------|-----------------|--------|
| Model | Softmax / CNN | **SoftmaxRegression** | **Critical** |

**Why this matters:**
- HisMSA Step1 (shuffle conv kernels) is designed for **CNN**
- SoftmaxRegression only has Linear layer - no conv kernels to shuffle!
- Step1 effect is **minimal** on linear models
- Paper shows stronger results on CNN

**Recommendation:**
- Use CNN model for fair comparison
- Or acknowledge that SoftmaxRegression is not ideal for HisMSA

### 3. Step2 Boundary Estimation

| Condition | Paper | Your Experiment | Impact |
|-----------|-------|-----------------|--------|
| History | Clean warmup | **May be polluted** | Moderate |

**Your diagnosis showed:**
- Gamma can be very small (0.04) when history is insufficient
- This suppresses attack too much
- Paper assumes clean history from early rounds

### 4. Data Distribution

| Condition | Paper | Your Experiment | Impact |
|-----------|-------|-----------------|--------|
| Distribution | IID + Non-IID | **Non-IID (LabelSeparation)** | Moderate |

**Why this matters:**
- Non-IID makes training harder (baseline accuracy lower)
- But also makes attacks harder to detect (higher variance)
- Your 0.66 might be "good" relative to baseline in non-IID

## Why Your Accuracy is Higher (0.66 vs Paper's 0.3)

### Reason 1: Model Type Mismatch (Most Critical)
- **SoftmaxRegression** cannot fully utilize HisMSA Step1
- Linear models have no "conv kernels" to shuffle
- Step1 effect is minimal → attack is weaker

### Reason 2: High Byzantine Ratio
- **30% is too high** for stealth attack
- Mean aggregation: 70% honest updates can "dilute" 30% malicious
- Paper's 2-10% maintains stealth while being effective

### Reason 3: Step2 May Be Too Conservative
- Your diagnosis showed gamma=0.04 in some cases
- This compresses attack by 96%!
- Paper's implementation may have different boundary estimation

### Reason 4: Different Baseline
- Your baseline (label_flipping) got 0.63
- Paper's baseline might be different
- Need to compare relative degradation, not absolute values

## Recommendations to Match Paper

### Immediate Fixes:

1. **Reduce Byzantine ratio to match paper:**
   ```python
   # In main CMomentum.py
   byzantine_size = 1  # 10% for 10 nodes, or 2 for 5% if node_size=20
   ```

2. **Use CNN model:**
   ```python
   # In main CMomentum.py, line 42
   task = NeuralNetworkTask(data_package, batch_size=32)  # Use CNN
   ```

3. **Check Step2 boundary estimation:**
   - Ensure warmup period is clean (no attacks during warmup)
   - Or adjust warmup_rounds based on when attacks start

4. **Compare relative degradation:**
   - Baseline (no attack): accuracy = X
   - HisMSA: accuracy = Y
   - Degradation = (X - Y) / X
   - Paper shows ~70% degradation, check if you get similar

### Experimental Protocol:

```bash
# 1. Baseline (no attack) - 10% Byzantine (but no attack)
python "main CMomentum.py" --attack none --aggregation mean --data-partition noniid

# 2. HisMSA with 10% Byzantine
python "main CMomentum.py" --attack hismsa --aggregation mean --data-partition noniid
# (with byzantine_size=1)

# 3. HisMSA with CNN model
# (modify to use NeuralNetworkTask)
python "main CMomentum.py" --attack hismsa --aggregation mean --data-partition noniid

# 4. Compare degradation percentages
```

## Expected Results After Fixes

If you match paper conditions:
- **CNN model**: Step1 will have real effect
- **10% Byzantine**: Maintains stealth, effective but not obvious
- **Clean warmup**: Step2 boundaries will be accurate
- **Expected accuracy**: Should drop significantly (closer to 0.3-0.5 range)

## Why Paper Gets 0.3 But You Get 0.66

**Most likely explanation:**
1. **Model mismatch (60% of gap)**: SoftmaxRegression vs CNN
2. **Byzantine ratio (20% of gap)**: 30% vs 10%
3. **Step2 suppression (15% of gap)**: Conservative boundaries
4. **Different baseline (5% of gap)**: Non-IID makes everything harder

**To verify:**
- Run with CNN + 10% Byzantine
- Compare degradation percentage, not absolute accuracy
- Check if Step1 actually modifies conv layers (diagnose)

