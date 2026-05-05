"""
HSM-Adaptive 中 trust 计算相关代码片段（第 1040-1180 行）
重点：a/m/c 的计算和更新
"""

# ============================================
# [STEP A] Reference Direction (第 1041-1046 行)
# ============================================
if self.last_agg is not None:
    ref_direction = self.last_agg
else:
    ref_direction = self.server_momentum

# ============================================
# [STEP B] Feature Extraction (第 1048-1071 行)
# ============================================

# 1. Global alignment (a_i) - 计算了
ref_norm = torch.norm(ref_direction)
msg_norms_clipped = torch.norm(messages_clipped, dim=1) + self.eps
if ref_norm < 1e-6:
    alignments = torch.zeros(num_nodes, device=device, dtype=dtype)
else:
    alignments = torch.sum(messages_clipped * ref_direction, dim=1) / (
        msg_norms_clipped * (ref_norm + self.eps)
    )
# alignments = a_i (第 1056 行)

align_centered = alignments - torch.median(alignments)  # 第 1071 行

# ============================================
# [STEP C] TAW (第 1073-1088 行)
# ============================================
# T_i = rho * T_{i,t-1} + (1-rho) * align_centered[i]
# w_hist = sigmoid((T_i - median(T)) / MAD(T))

# ============================================
# [STEP D] Sign-Sketch (第 1095-1126 行)
# ============================================
# w_sketch 计算

# ============================================
# [STEP E] 权重组合 (第 1128-1133 行)
# ============================================
weights = w_hist * w_sketch * scale_gate + 1e-3

# ============================================
# Trust 分数计算 (第 1135-1138 行)
# ============================================
w_global = torch.tanh(torch.std(alignments) * 5.0)
trust_scores_tensor = w_global * align_centered - 0.5 * scale_z_scores

# ============================================
# Logging 兼容字段 (第 1140-1148 行)
# ============================================
alignments_tensor = alignments           # ✅ a 有值
alignments_centered = align_centered     # ✅ a_centered 有值
self_alignments = None                   # ❌ c 设为 None (第 1147 行) - 导致日志中 c=0
# momentum_align 未计算，直接传 None      # ❌ m 没有计算 - 导致日志中 m=0

# ============================================
# 调用日志 (第 1169-1176 行)
# ============================================
self._log_statistics(
    alignments_tensor, alignments_centered, scale_deviations_tensor, trust_scores_tensor,
    weights_tensor, norms_clipped, aggregated_update, client_ids, new_clients, tau_t,
    task_consistency_tensor, alignments_hat, align_std_value, None, current_alpha_log,
    self_alignments,        # ❌ None (第 1173 行)
    None,                   # ❌ momentum_align=None (第 1174 行)
    accuracy=self.current_accuracy,
)

# ============================================
# 问题总结
# ============================================
"""
问题：
1. self_alignments = None (第 1147 行) - 没有计算 self-consistency (c)
2. momentum_align 未计算，直接传 None (第 1174 行) - 没有计算 momentum alignment (m)

导致日志中：
- c (Self Align) = 0.0
- m (Mom Align) = 0.0

解决方案：
需要计算：
1. m_i = cos(Δ'_i, server_momentum) - momentum alignment
2. c_i = cos(Δ'_i, client_history_update_i) - self-consistency

然后接入 trust score：
trust = w_a * align_centered + w_m * m + w_c * c - w_scale * scale_z_scores
"""
