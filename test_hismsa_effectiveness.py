"""
Diagnostic script: Verify if HisMSA attack is actually effective
"""
import torch
import copy
from ByrdLab.attack import HisMSA
from ByrdLab.library.dataset import mnist
from ByrdLab.tasks.softmaxRegression import softmaxRegressionTask
from ByrdLab.library.RandomNumberGenerator import RngPackage

print("=" * 60)
print("HisMSA Attack Effectiveness Diagnosis")
print("=" * 60)

# Create task and model
data_package = mnist()
task = softmaxRegressionTask(data_package, batch_size=32)
model = task.model

# Create attack instance
attack = HisMSA(
    shuffle_prob=1.0,
    scaling_factor_range=(0.8, 1.2),
    strict_equivalence=False,  # Allow inconsistency for stronger attack
    warmup_rounds=5
)

# Create test input (ensure data type matches)
from ByrdLab import FEATURE_TYPE
test_input = torch.randn(10, 784, dtype=FEATURE_TYPE)  # 10 samples

print("\n1. Check model structure...")
print(f"   Model type: {type(model)}")
print(f"   Parameter count: {sum(p.numel() for p in model.parameters())}")
for name, param in model.named_parameters():
    print(f"   {name}: {param.shape}")

print("\n2. Check if HisMSA can find attackable layers...")
layers = attack._get_ordered_layers(model)
print(f"   Found {len(layers)} layers:")
for name, param, layer_type, module in layers:
    if param is not None:
        print(f"   - {name}: {layer_type}, shape={param.shape}")
    else:
        print(f"   - {name}: {layer_type} (BN)")

if len(layers) == 0:
    print("   [WARNING] No attackable layers found!")
    print("   Softmax Regression only has Linear layers, HisMSA may have weak effect")

print("\n3. Test Step 1 functional equivalence...")
model0 = copy.deepcopy(model)
model1 = copy.deepcopy(model)

# Get original output
model0.eval()
with torch.no_grad():
    output0 = model0(test_input)

# Apply Step 1
attack.apply_step1_to_model(model1, RngPackage())

# Get attacked output
model1.eval()
with torch.no_grad():
    output1 = model1(test_input)

# Compute differences
diff = torch.abs(output1 - output0)
max_diff = diff.max().item()
mean_diff = diff.mean().item()

print(f"   Max difference: {max_diff:.6e}")
print(f"   Mean difference: {mean_diff:.6e}")

if max_diff < 1e-5:
    print("   [WARNING] Difference is very small, attack may not be effective or too weak")
else:
    print("   [OK] Attack indeed changed model output")

print("\n4. Check if parameters were modified...")
param_changed = False
for (name0, p0), (name1, p1) in zip(model0.named_parameters(), model1.named_parameters()):
    if not torch.allclose(p0, p1, atol=1e-6):
        param_changed = True
        diff_norm = torch.norm(p1 - p0).item()
        print(f"   [OK] {name0} modified (diff norm: {diff_norm:.6e})")

if not param_changed:
    print("   [WARNING] No parameters were modified! Attack may not be effective")

print("\n5. Test Step 2 (gamma clipping)...")
# Create a "malicious" update
local_model = copy.deepcopy(model)
global_model = copy.deepcopy(model)

# Make some modifications to local_model (simulate post-training update)
with torch.no_grad():
    for param in local_model.parameters():
        param.data += torch.randn_like(param) * 0.1

# Compute update norm
update_norm_before = 0.0
for (name1, param1), (name2, param2) in zip(
    local_model.named_parameters(),
    global_model.named_parameters()
):
    if name1 == name2:
        update_norm_before += torch.norm(param1.data - param2.data, p=2).item() ** 2
update_norm_before = (update_norm_before ** 0.5)

print(f"   Update norm (before clipping): {update_norm_before:.6e}")

# Apply gamma clipping
attack.apply_clipping_to_update(local_model, global_model)

# Compute clipped update norm
update_norm_after = 0.0
for (name1, param1), (name2, param2) in zip(
    local_model.named_parameters(),
    global_model.named_parameters()
):
    if name1 == name2:
        update_norm_after += torch.norm(param1.data - param2.data, p=2).item() ** 2
update_norm_after = (update_norm_after ** 0.5)

print(f"   Update norm (after clipping): {update_norm_after:.6e}")

if abs(update_norm_after - update_norm_before) < 1e-6:
    print("   [WARNING] Update norm did not change, gamma clipping may not be effective")
else:
    print(f"   [OK] Gamma clipping effective, norm change: {abs(update_norm_after - update_norm_before):.6e}")

print("\n6. Suggestions...")
print("   - If attack effect is not obvious, it may be because:")
print("     1. Softmax Regression is a linear model, HisMSA has weak effect on linear models")
print("     2. Suggest using neural network (NeuralNetworkTask) for testing")
print("     3. Or use more aggressive parameters: scaling_factor_range=(0.5, 1.5)")
print("   - Comparison experiment: run baseline without attack, check accuracy difference")
print("   - Baseline command: python \"main CMomentum.py\" --attack none --aggregation mean --data-partition noniid")

print("\n" + "=" * 60)
