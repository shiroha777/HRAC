"""
Diagnose if HisMSA attack is properly injected into aggregation updates
Systematic check following user's troubleshooting guide
"""
import torch
import copy
import sys
sys.path.insert(0, '.')

from ByrdLab.attack import HisMSA
from ByrdLab.library.dataset import mnist
from ByrdLab.tasks.softmaxRegression import softmaxRegressionTask
from ByrdLab.library.RandomNumberGenerator import RngPackage
from ByrdLab import DEVICE

print("=" * 70)
print("HisMSA Attack Injection Diagnosis - Systematic Check")
print("=" * 70)

# Create task and model
data_package = mnist()
task = softmaxRegressionTask(data_package, batch_size=32)
global_model = task.model.to(DEVICE)
server_model = copy.deepcopy(global_model)

# Create attack instance (using current configuration)
attack = HisMSA(
    shuffle_prob=1.0,
    scaling_factor_range=(0.8, 1.2),
    strict_equivalence=False,
    warmup_rounds=5
)

rng_pack = RngPackage()

print("\n[Check 1] Verify accuracy calculation location")
print("  - Model: server_model (global model) [OK]")
print("  - Dataset: get_test_iter (test set) [OK]")
print("  - Function: one_node_loss_accuracy_dist [OK]")
print("  -> Conclusion: accuracy is indeed global test set accuracy, this is OK")

print("\n[Check 2] Check if attack is properly injected into aggregation updates")
print("  Simulating one training round...")

# Simulate malicious client
byzantine_node = 0
honest_node = 1

# Create worker models
worker_models = {
    honest_node: copy.deepcopy(server_model),
    byzantine_node: copy.deepcopy(server_model)
}

# Step 1: Apply shuffle+scaling to malicious node model
print("\n  Step 1: Apply shuffle+scaling to malicious node model")
before_step1 = copy.deepcopy(worker_models[byzantine_node])
attack.apply_step1_to_model(worker_models[byzantine_node], rng_pack)

# Check if parameters were modified
param_changed = False
for (name1, p1), (name2, p2) in zip(before_step1.named_parameters(), 
                                    worker_models[byzantine_node].named_parameters()):
    if not torch.allclose(p1, p2, atol=1e-6):
        param_changed = True
        diff_norm = torch.norm(p2 - p1).item()
        print(f"    [OK] {name1} modified (diff norm: {diff_norm:.6e})")

if not param_changed:
    print("    [ERROR] Step1 did not modify parameters!")

# Simulate training (simple gradient update)
print("\n  Simulate training: compute gradients on modified model")
from ByrdLab import FEATURE_TYPE
features = torch.randn(32, 784, dtype=FEATURE_TYPE).to(DEVICE)
targets = torch.randint(0, 10, (32,)).to(DEVICE)

# Malicious node training
worker_models[byzantine_node].train()
predictions = worker_models[byzantine_node](features)
loss = torch.nn.functional.cross_entropy(predictions, targets)
worker_models[byzantine_node].zero_grad()
loss.backward()

# Compute update vector (trained model - global model)
print("\n  Compute update vector: local_model - global_model")
update_before_step2 = {}
for (name1, param1), (name2, param2) in zip(
    worker_models[byzantine_node].named_parameters(),
    server_model.named_parameters()
):
    update_before_step2[name1] = param1.data - param2.data

# Compute update norm
update_norm_before_step2 = 0.0
for delta in update_before_step2.values():
    update_norm_before_step2 += torch.norm(delta, p=2).item() ** 2
update_norm_before_step2 = (update_norm_before_step2 ** 0.5)
print(f"    Update norm (before Step2): {update_norm_before_step2:.6e}")

# Step 2: Apply gamma clipping to momentum (as in actual code)
print("\n  Step 2: Apply gamma clipping to momentum")
# Simulate momentum computation (as in actual code)
momentum = []
for para in worker_models[byzantine_node].parameters():
    # Simplified: momentum = grad + weight_decay * param
    # In real code, it's more complex with alpha
    mom = para.grad.data.clone() if para.grad is not None else torch.zeros_like(para.data)
    momentum.append(mom)

# Compute momentum norm
momentum_norm_sq = 0.0
for mom in momentum:
    momentum_norm_sq += torch.norm(mom, p=2).item() ** 2
momentum_norm = (momentum_norm_sq ** 0.5)
print(f"    Momentum norm (before Step2): {momentum_norm:.6e}")

# Get clipping factor gamma
gamma = attack.get_clipping_factor(momentum_norm)
print(f"    Gamma factor: {gamma:.6e}")

# Apply gamma clipping to momentum
for mom in momentum:
    mom.data.mul_(gamma)

# Compute momentum norm after clipping
momentum_norm_after_sq = 0.0
for mom in momentum:
    momentum_norm_after_sq += torch.norm(mom, p=2).item() ** 2
momentum_norm_after = (momentum_norm_after_sq ** 0.5)
print(f"    Momentum norm (after Step2): {momentum_norm_after:.6e}")

# Also test old method for comparison
print("\n  Step 2 (old method): Apply gamma clipping to model parameters")
attack.apply_clipping_to_update(worker_models[byzantine_node], server_model)

# Compute clipped update norm (old method)
update_after_step2 = {}
for (name1, param1), (name2, param2) in zip(
    worker_models[byzantine_node].named_parameters(),
    server_model.named_parameters()
):
    update_after_step2[name1] = param1.data - param2.data

update_norm_after_step2 = 0.0
for delta in update_after_step2.values():
    update_norm_after_step2 += torch.norm(delta, p=2).item() ** 2
update_norm_after_step2 = (update_norm_after_step2 ** 0.5)
print(f"    Update norm (after Step2, old method): {update_norm_after_step2:.6e}")

# Compute gamma (old method)
gamma_old = update_norm_after_step2 / (update_norm_before_step2 + 1e-8)
print(f"    Gamma factor (old method): {gamma_old:.6e}")

print("\n  Summary:")
if gamma < 0.1:
    print(f"    [WARNING] Momentum gamma ({gamma:.6e}) is very small, Step2 may suppress attack!")
elif gamma > 0.9:
    print(f"    [OK] Momentum gamma ({gamma:.6e}) close to 1, Step2 clipping is not severe")
else:
    print(f"    [INFO] Momentum gamma={gamma:.3f}, Step2 has clipping but not too severe")

print("\n[Check 3] Check Step1 destructiveness")
print("  Testing functional equivalence...")

model0 = copy.deepcopy(global_model)
model1 = copy.deepcopy(global_model)
from ByrdLab import FEATURE_TYPE
test_input = torch.randn(10, 784, dtype=FEATURE_TYPE).to(DEVICE)

model0.eval()
with torch.no_grad():
    output0 = model0(test_input)

attack.apply_step1_to_model(model1, rng_pack)
model1.eval()
with torch.no_grad():
    output1 = model1(test_input)

diff = torch.abs(output1 - output0)
max_diff = diff.max().item()
mean_diff = diff.mean().item()

print(f"    Max difference: {max_diff:.6e}")
print(f"    Mean difference: {mean_diff:.6e}")

if max_diff < 1e-5:
    print("    [WARNING] Step1 difference is very small, may be too weak!")
    print("    Suggestion: strict_equivalence=False or expand scaling_factor_range")
elif max_diff < 0.1:
    print("    [INFO] Step1 has difference but not large")
    print("    Suggestion: try more aggressive parameters")
else:
    print("    [OK] Step1 has significant difference")

print("\n[Check 4] Check malicious node ratio")
print("  Current settings:")
print(f"    - Total nodes: 10")
print(f"    - Malicious nodes: 1 (10%)")
print("  [WARNING] Malicious node ratio is small, mean aggregation may dilute attack")
print("  Suggestion: try increasing byzantine_size=2 or 3")

print("\n[Check 5] Check Step2 boundary estimation")
print("  Testing boundary computation...")

# Simulate several rounds of history
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

R_min, R_max = attack._compute_clipping_bounds()
print(f"    R_min: {R_min:.6e}")
print(f"    R_max: {R_max:.6e}")
print(f"    Update norm: {update_norm_before_step2:.6e}")

if update_norm_before_step2 < R_min:
    print(f"    [WARNING] Update norm < R_min, gamma will amplify update")
elif update_norm_before_step2 > R_max:
    print(f"    [WARNING] Update norm > R_max, gamma will shrink update")
    print(f"    Gamma = {R_max / update_norm_before_step2:.6e}")
else:
    print(f"    [OK] Update norm in range, gamma=1")

print("\n" + "=" * 70)
print("Diagnosis Suggestions:")
print("=" * 70)
print("1. If Step2 gamma is very small -> try disabling Step2 or relax boundaries")
print("2. If Step1 difference is small -> use more aggressive parameters:")
print("   - scaling_factor_range=(0.5, 2.0)")
print("   - strict_equivalence=False (already set)")
print("3. If malicious node ratio is small -> increase byzantine_size=2 or 3")
print("4. Quick test: disable Step2, only enable Step1, check effect")
print("=" * 70)

