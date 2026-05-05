"""
Debug HisMSA on CNN model - check if Step1 is actually applied to conv layers
"""
import torch
import copy
from ByrdLab.attack import HisMSA
from ByrdLab.library.dataset import mnist
from ByrdLab.tasks.neuralNetwork import NeuralNetworkTask
from ByrdLab.library.RandomNumberGenerator import RngPackage
from ByrdLab import DEVICE, FEATURE_TYPE

print("=" * 70)
print("Debug HisMSA on CNN Model")
print("=" * 70)

# Create CNN task
data_package = mnist()
task = NeuralNetworkTask(data_package, batch_size=32)
model = task.model.to(DEVICE)

print("\n1. Check CNN model structure...")
print(f"   Model type: {type(model)}")
for name, module in model.named_modules():
    if hasattr(module, 'weight') and module.weight is not None:
        print(f"   {name}: {type(module).__name__}, shape={module.weight.shape}")

# Create attack
attack = HisMSA(
    shuffle_prob=1.0,
    scaling_factor_range=(0.8, 1.2),
    strict_equivalence=False,
    warmup_rounds=5
)

print("\n2. Check if HisMSA can find attackable layers...")
layers = attack._get_ordered_layers(model)
print(f"   Found {len(layers)} layers:")
conv_count = 0
linear_count = 0
for name, param, layer_type, module in layers:
    if param is not None:
        print(f"   - {name}: {layer_type}, shape={param.shape}")
        if layer_type == 'conv':
            conv_count += 1
        elif layer_type == 'linear':
            linear_count += 1
    else:
        print(f"   - {name}: {layer_type} (BN)")

print(f"\n   Summary: {conv_count} conv layers, {linear_count} linear layers")

if conv_count == 0:
    print("   [ERROR] No conv layers found! HisMSA Step1 cannot work!")
else:
    print(f"   [OK] Found {conv_count} conv layers - Step1 can be applied")

print("\n3. Test Step1 on CNN model...")
model0 = copy.deepcopy(model)
model1 = copy.deepcopy(model)

# Get original output
model0.eval()
test_input = torch.randn(10, 1, 28, 28, dtype=FEATURE_TYPE).to(DEVICE)
with torch.no_grad():
    output0 = model0(test_input)

# Apply Step1
rng_pack = RngPackage()
attack.apply_step1_to_model(model1, rng_pack)

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
    print("   [ERROR] Step1 difference is very small - attack may not be working!")
elif max_diff < 0.1:
    print("   [WARNING] Step1 difference is small - attack may be weak")
else:
    print("   [OK] Step1 has significant difference")

# Check which parameters were modified
print("\n4. Check which parameters were modified by Step1...")
param_modified = False
for (name0, p0), (name1, p1) in zip(model0.named_parameters(), model1.named_parameters()):
    if not torch.allclose(p0, p1, atol=1e-6):
        param_modified = True
        diff_norm = torch.norm(p1 - p0).item()
        param_type = 'conv' if 'conv' in name0 else 'linear' if 'linear' in name0 or 'fc' in name0 else 'other'
        print(f"   [OK] {name0} ({param_type}) modified, diff_norm={diff_norm:.6e}")

if not param_modified:
    print("   [ERROR] No parameters were modified!")

print("\n5. Test Step2 gamma clipping...")
# Simulate a malicious update
local_model = copy.deepcopy(model)
global_model = copy.deepcopy(model)

# Make local model different (simulate training)
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

print(f"   Update norm (before Step2): {update_norm_before:.6e}")

# Simulate some history for boundary estimation
for i in range(6):
    fake_model = copy.deepcopy(global_model)
    with torch.no_grad():
        for param in fake_model.parameters():
            param.data += torch.randn_like(param) * 0.01
    
    if i == 0:
        attack.update_model_history(fake_model, None, round_num=i)
    else:
        prev_model = copy.deepcopy(global_model)
        with torch.no_grad():
            for param in prev_model.parameters():
                param.data += torch.randn_like(param) * 0.01 * (i-1)
        attack.update_model_history(fake_model, prev_model, round_num=i)

# Get gamma
gamma = attack.get_clipping_factor(update_norm_before)
print(f"   Gamma factor: {gamma:.6e}")

R_min, R_max = attack._compute_clipping_bounds()
print(f"   R_min: {R_min:.6e}, R_max: {R_max:.6e}")

if gamma < 0.1:
    print("   [WARNING] Gamma is very small - Step2 may suppress attack too much!")
    print("   Suggestion: Try disabling Step2 or adjust boundaries")
elif gamma > 0.9:
    print("   [OK] Gamma close to 1 - Step2 clipping is not severe")
else:
    print(f"   [INFO] Gamma={gamma:.3f} - Step2 has moderate clipping")

print("\n" + "=" * 70)
print("Diagnosis:")
print("=" * 70)
print("If conv layers found but attack still weak:")
print("  1. Step2 may be too conservative - try disabling it")
print("  2. Scaling range may be too small - try (0.5, 2.0)")
print("  3. Malicious ratio may need adjustment")
print("=" * 70)


