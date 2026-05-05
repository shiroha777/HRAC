# 每次 acc drop 发生时代码的完整流程说明

## 1. 为什么 log 里还会出现 “acc drop”？

**这是预期行为。** “acc drop” 是 **触发 bypass 的检测结果**：当「本轮更新后准确率相对上一 display 下降超过阈值」时打印这条消息并启动回退与重做。所以训练中**仍会出现** acc drop 的 log，表示“检测到掉 acc、已触发 bypass”；该轮最终记录与报告的 acc 会用 bypass 替代后的结果（见第 3 节）。

- 配置：`bypass_on_acc_drop=True`，`bypass_acc_drop_tol=0.1`（在 `main CMomentum.py` 里对 HRAC 设为 0.1）
- 含义：当 `accuracy_before_this_round - test_accuracy_after > 0.1` 时判定为 acc drop，触发 bypass

---

## 2. 每次 acc drop 发生时代码具体做了什么（主流程）

以 **HRAC + label_flipping + CMomentum_under_DPA** 为例（即 `hrac-log-label_flipping.txt` 的典型配置），单次 acc drop 的完整流程如下。

### 2.1 主循环到某个 display 步（例如 iter=12500）

1. **记录本段起点准确率**  
   - `iteration % display_interval == 0` 时，`accuracy_before_this_round = acc_path[-1]`（上一 display 的准确率，例如 12400 的 acc）。

2. **各节点算 momentum，聚合一次**  
   - 主循环照常：各 node 前向+反向，更新 `worker_momentum`，然后 `worker_grad_flat = flatten_list(worker_momentum)`。
   - 调用 **一次** `aggrGrad_flat = self.aggregation.run(worker_grad_flat)`，用 HRAC 得到聚合梯度。

3. **用聚合结果更新服务器模型**  
   - `server_model -= lr * aggrGrad`，并保存 `previous_server_model` 用于 bypass。

4. **检测 acc drop（check_and_apply_bypass_after_update）**  
   - 在 `ByrdLab/aggregation.py` 的 `check_and_apply_bypass_after_update` 中：
     - 用当前 `server_model` 在测试集上算 `test_accuracy_after`；
     - 若 `accuracy_before_this_round - test_accuracy_after > bypass_acc_drop_tol`：
       - 设置 `_bypass_rewind_to = iteration - display_interval`（如 12400），`_bypass_redo_until = iteration`（12500）；
       - 设置 `_bypass_mode_active = True`，并打印 **acc drop** 那条 log（before/after/tol, rewind to …）。

### 2.2 主循环紧接着做 rewind + redo（centraliedAlgorithm）

5. **取 rewind 信息并恢复 checkpoint**  
   - `rewind_to, redo_until = self.aggregation.get_bypass_rewind_info()` → 得到 (12400, 12500)；
   - 若存在 `restore_to_display_checkpoint`，则调用  
     `self.aggregation.restore_to_display_checkpoint(server_model, rewind_to=rewind_to)`：
     - 把 **服务器模型** 恢复到 `rewind_to`（12400）时的参数；
     - 把 HRAC 内部状态（ν, μ, b, r_prev, **_agg_momentum** 等）从当时保存的 snapshot 恢复。

6. **Bypass 重做 100 步（与主循环相同：调用 run() 100 次，仅内部逻辑不同）**  
   - 打印：`[Bypass] redoing 12400->12500 with 100 run() steps (same as main loop, bypass mode)`；
   - 对 `redo_iter = 12401..12500` 循环 100 次，每一步：
     - 取一 batch，各节点算 momentum，得到 `worker_grad_flat_redo`；
     - 调用 **`aggregation.run(worker_grad_flat_redo)`**（与主流程相同的调用方式）；
     - **run() 内部**若 `_bypass_mode_active=True`：只做 **mean(所有 client)+momentum**，不跑完整 HRAC（无 ν 加权、无 τ,μ,b 更新），并每步 `_restore_state_from_snapshot(skip_agg_momentum=True)` 冻结 ν,μ,b,r_prev；
     - 用返回的聚合向量更新服务器：`server_model -= lr_redo * aggrGrad_redo`。
   - 退出 bypass 后：`_bypass_mode_active=False`，后续 run() 恢复完整 HRAC，ν,μ,b 等解冻、照常更新。

7. **Bypass 结束后**  
   - 再次调用 `check_and_apply_bypass_after_update(redo_until, ...)`，用当前 `server_model` 的 acc 与 `_acc_before_bypass` 比较，若在 tol 内则退出 bypass（`_bypass_mode_active = False`），后续主循环恢复正常 HRAC 更新。

8. **Display checkpoint 收尾**  
   - `iteration % display_interval == 0` 时调用 `finalize_display_checkpoint(iteration)`，更新“上一 display”的 checkpoint，供下次可能的 rewind 使用。

---

## 3. Bypass 后：该轮 acc 与 log 均为替代结果

- **保存**：bypass 重做结束后，`check_and_apply_bypass_after_update(redo_until, ...)` 会用 **bypass 替代后模型** 的测试 acc/loss 覆盖 `acc_path[-1]`、`loss_path[-1]`，因此记录里该轮就是 bypass 结果。
- **终端**：主循环在 bypass 重做后会再打一行 `[CMomentum] redo_until/... loss=... accuracy=... [bypass-replaced]`，用 bypass 后的 loss/acc，与保存一致，避免“先打了一行再 bypass”造成的混乱。
- **HRAC iter**：`iteration_count` 在每次 `run()` 开头 +1；bypass 用 run() 100 次，自然递增到 `redo_until`，与主循环对齐。

---

## 4. 流程小结（每次 acc drop）

| 步骤 | 位置 | 行为 |
|------|------|------|
| 1 | 主循环 | 本 display 步开始前记录 `accuracy_before_this_round` |
| 2 | 主循环 | 正常算 momentum → 调用 **一次** `aggregation.run()` → 更新 server 模型 |
| 3 | aggregation | `check_and_apply_bypass_after_update` 测当前 acc，若下降超过 tol 则打印 **acc drop**，设置 rewind_to/redo_until、_bypass_mode_active |
| 4 | 主循环 | `get_bypass_rewind_info()` 取 (rewind_to, redo_until) |
| 5 | 主循环 | `restore_to_display_checkpoint(server_model)`：模型和 HRAC 状态回到上一 display |
| 6 | 主循环 | **100 次** `run()`，与主循环相同调用；run() 内 bypass 时只做 mean+momentum、冻结 ν,μ,b |
| 8 | 主循环 | bypass 结束后再调一次 `check_and_apply_bypass_after_update`，满足条件则解冻、退出 bypass |

因此：**“acc drop” 仍会出现是设计如此**，表示「在这一步检测到掉 acc 并触发了 bypass」；当前实现下，**redo 阶段应只打一行 `[Bypass] redoing ... (no HRAC run)`**，不再每步打 HRAC Statistics。

---

## 5. Bypass 时为何不会刷屏

Bypass 时 `run()` 在开头短路：只做 mean+momentum + 冻结，**不进入**后面的 HRAC 聚合与 Statistics 打印，因此不会出现大量 `[HRAC] Iteration ... Statistics`。仅每 20 个 bypass_round 打一行短日志 `[Bypass] iter=... bypass_round=... (mean+momentum, ν/μ/b frozen)`。

**注意**：若用 **CMomentum_under_HisMSA**（HisMSA/MSA 攻击），该类 **没有** rewind/redo 块，只调用 `check_and_apply_bypass_after_update`，会打印 acc drop 但**不会**做 restore 和 100 步 redo。

---

## 6. 相关代码位置速查

- **acc drop 检测与设置 rewind**：`ByrdLab/aggregation.py` → `check_and_apply_bypass_after_update`（约 1531–1599 行）
- **rewind 信息**：`get_bypass_rewind_info`（约 1392 行）
- **恢复 checkpoint**：`restore_to_display_checkpoint`（约 1347 行）
- **Bypass 内单步更新（不调 run）**：`get_bypass_mean_momentum_step`（约 1400 行）
- **主循环 rewind + redo**：`ByrdLab/centraliedAlgorithm.py` → `CMomentum_under_DPA.run()`（约 397–457 行）；另一处类似逻辑在 855–913 行（如 CSGD_under_DPA 等）

---

## 7. 为何依旧出现 acc drop、第一次 drop 后性能变差

### 7.1 现象（来自 hrac-log-label_flipping.txt）

- **3300**：before=0.3233 → after=0.1089 → 进入 bypass，回退 3200，redo 100 步 → **exit acc=0.3361**（恢复良好）。
- **3400**：before=0.3361 → after=0.1218 → 再次 drop，回退 3300，redo 100 步后 acc 仍不在 tol 内，**未退出**；主循环 3401→3500、3501→3600 继续以 bypass（mean+momentum）跑，共 **300 步** → **exit acc=0.2619**（比 pre-bypass 0.3361 差约 7.4pp）。
- 之后 3700 等仍有 “acc drop but below tol”，整体曲线自第一次大 drop 后处于较低水平。

### 7.2 原因归纳

1. **Bypass 只替换「当前这一段」**  
   触发 bypass 时用 mean+momentum 重跑 100 步，替代「导致 drop 的那 100 步」。**重跑结束后下一段 100 步又是正常 HRAC**。若攻击/数据使下一段再次产生坏的 g_t（ν/权重异常），就会再次 drop。因此 bypass **不能防止下一次 drop**，只能撤销当前这一次坏步。

2. **退出条件允许「明显更差」**  
   退出条件为 `abs(acc - acc_before_bypass) <= tol`（tol=0.1）。只要 acc ∈ [acc_before−0.1, acc_before+0.1] 就退出。例如 pre-bypass=0.3361 时 acc=0.2619 就退出（0.0742 ≤ 0.1），**退出时模型已比 pre-bypass 差约 7.4pp**，后续从更差起点继续 HRAC，易再次 drop，形成恶性循环。

3. **Bypass 实际可能跑超过 100 轮**  
   退出只在**每个 display_interval 边界**检查。redo 块只跑 100 步；若这 100 步后 acc 仍不在 tol 内，不会退出，**主循环 3401、3402… 仍以 _bypass_mode_active=True 继续**，每步 run() 仍是 mean+momentum，直到下一个 display（3500、3600…）某次检查时 acc 进入 tol。因此会出现 300 步 bypass（100 redo + 200 主循环），且这 300 步用的数据流与 3301→3400 不同，模型容易漂到更差区域（如 0.26），加剧「第一次 drop 后性能变差」。

### 7.3 修改建议与已实现

- **已实现：强制「只 redo 100 步然后退出」**  
  redo 块执行完 100 次 run() 后，主循环在 redo 块末尾**无论 acc 是否在 tol 内都强制退出 bypass**（`_bypass_mode_active = False`），避免主循环 3401、3402… 继续以 bypass 跑 200、300 步导致漂移。每次触发 drop 只做「恰好 100 步 mean+momentum」然后恢复 HRAC。日志会打 `[Bypass] iter=... force exit after 100 redo steps`。
- **可选**：收紧退出条件（例如仅当 `acc >= acc_before_bypass - small_eps` 才允许退出）可减少「带着明显更差的模型退出」，但若 100 步后仍恢复不到 pre-bypass，可能长期卡在 bypass；当前采用「100 步后强制退出」更稳妥。

---

## 8. Bypass 后 3300 的显示：真实 ν/d 与日志均为上一 display（3200）

- **需求**：3300 发生 acc drop 时退回 3200 的**完整**状态（模型 + ν, μ, b, r_prev, momentum）；3300 的**输出**（日志与 display）应为：acc = bypass 替代结果，ν、d、τ、μ = **3200 轮**的值（冻结），而不是坏步 3300 的 ν/acc。
- **实现**：
  1. **Checkpoint 保存「当轮结束后」**：`finalize_display_checkpoint(iteration, server_model)` 在每段 display 步**结束后**调用，保存当前 `server_model` 与当前 aggregator 状态（`get_current_snapshot(iteration)`，含 ν, μ, b, d_eff_list），使 rewind 真正回到 3200 的 ν/d。
  2. **触发 bypass 时移除坏块**：在 `check_and_apply_bypass_after_update` 里一旦设置 `_bypass_rewind_to`，即调用 `_remove_last_display_block_from_buffer(iteration)`，从 buffer 中删掉本 iter 已写入的 `[HRAC ν detail]` 与 `[HRAC] Iteration N Statistics` 整块。
  3. **Bypass 结束后追加正确块**：在退出 bypass 时调用 `append_bypass_replacement_display_block(redo_until, bypass_acc, bypass_loss)`，用 `_bypass_snapshot`（即 rewind_to 的 3200 状态）的 ν、d、τ、μ 与 bypass 的 acc，写入一条 `[HRAC ν detail]` 和一条 `[HRAC] Iteration N Statistics`（标 `[bypass-replaced]`），保证 3300 的日志与真实状态一致：ν/d 为 3200，acc 为 bypass 结果。
