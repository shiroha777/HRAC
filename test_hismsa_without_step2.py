"""
Test HisMSA with Step2 disabled to see if Step1 alone is effective
This helps diagnose if Step2 is suppressing the attack too much
"""
import sys
import os
sys.path.insert(0, '.')

# Modify the attack to disable Step2
from ByrdLab.attack import HisMSA

class HisMSA_NoStep2(HisMSA):
    """HisMSA with Step2 disabled for testing"""
    def get_clipping_factor(self, grad_update_norm):
        """Override to always return 1.0 (no clipping)"""
        return 1.0

# Monkey patch to use modified attack
import ByrdLab.attack as attack_module
original_hismsa = attack_module.HisMSA

def create_hismsa_no_step2(*args, **kwargs):
    return HisMSA_NoStep2(*args, **kwargs)

print("=" * 70)
print("Testing HisMSA with Step2 DISABLED (gamma=1.0)")
print("=" * 70)
print("\nThis test will show if Step1 alone is effective.")
print("If accuracy drops significantly, Step2 was suppressing the attack.")
print("If accuracy is still high, Step1 may be too weak or other issues exist.")
print("\nTo run:")
print('  python "main CMomentum.py" --attack hismsa --aggregation mean --data-partition noniid')
print("\nBut first, modify main CMomentum.py to use HisMSA_NoStep2")
print("=" * 70)


