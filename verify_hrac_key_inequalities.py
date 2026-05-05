import argparse
import math
import random
import sys
from dataclasses import dataclass
from typing import Dict, List

import torch


sys.path.insert(0, r"g:\FYP\LPA")

from ByrdLab.aggregation import C_HRAC  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class RoundDebug:
    B: float
    messages: torch.Tensor
    messages_clipped: torch.Tensor
    processed: torch.Tensor
    weights: torch.Tensor
    honest_idx: List[int]
    byz_idx: List[int]
    residual_checks: List[Dict[str, float]]


class HRACDebug(C_HRAC):
    def step_with_debug(self, messages: torch.Tensor, client_ids=None) -> RoundDebug:
        N, D = messages.shape
        device, dtype = messages.device, messages.dtype

        if client_ids is None:
            client_ids = list(range(N))
        else:
            client_ids = [int(x) for x in client_ids]

        self.iteration_count += 1

        norms = torch.norm(messages, dim=1)
        median_norm = torch.median(norms)
        B = (self.c_g * median_norm).clamp_min(self.eps)
        scale_global = torch.minimum(torch.ones_like(norms), B / (norms + self.eps))
        messages_clipped = messages * scale_global.unsqueeze(1)

        norm_small = norms < (0.5 * median_norm)
        has_small_norm = norm_small.any().item()
        scale_lift = median_norm / (norms + self.eps)
        messages_eff = torch.where(
            norm_small.unsqueeze(1),
            messages * scale_lift.unsqueeze(1),
            messages_clipped,
        )

        if self.one is None or self.one.device != device or self.one.dtype != dtype:
            self.one = torch.tensor(1.0, device=device, dtype=dtype)

        processed = []
        r_bar_list = []
        r_bar_norm_list = []
        delta_tilde_norm_list = []
        new_cids = set()
        residual_checks: List[Dict[str, float]] = []

        for i, cid in enumerate(client_ids):
            delta_t_i = messages_clipped[i]
            if cid not in self.b:
                new_cids.add(cid)
                init_norm = median_norm.clamp_min(torch.tensor(1.0, device=device, dtype=dtype))
                self.b[cid] = delta_t_i.detach().clone()
                self.b_norm[cid] = (messages_eff[i] if has_small_norm else delta_t_i).detach().clone()
                self.mu[cid] = init_norm.detach()
                self.nu[cid] = init_norm.detach().clone()
                self.r_prev[cid] = torch.zeros(D, device=device, dtype=dtype)
                processed.append(delta_t_i)
                r_bar_list.append(torch.zeros(D, device=device, dtype=dtype))
                r_bar_norm_list.append(torch.zeros(D, device=device, dtype=dtype))
                delta_tilde_norm_list.append((messages_eff[i] if has_small_norm else delta_t_i).detach().clone())
                residual_checks.append(
                    {
                        "is_new": 1.0,
                        "tau": float(self.c * self.mu[cid] + self.eps),
                        "r_norm": 0.0,
                        "clip_gap": 0.0,
                        "clip_rhs": 0.0,
                        "pre_post_norm": float(torch.norm(delta_t_i).item()),
                    }
                )
                continue

            tau = self.c * self.mu[cid] + self.eps
            r = delta_t_i - self.b[cid]
            r_bar = self._clip_by_norm(r, tau) if self.enable_per_client_residual_clip else r
            delta_tilde_pre_cap = self.b[cid] + r_bar
            delta_tilde = delta_tilde_pre_cap
            if self.enable_post_residual_b_cap:
                delta_tilde = delta_tilde * torch.minimum(
                    self.one, B / (torch.norm(delta_tilde) + self.eps)
                )

            processed.append(delta_tilde)
            r_bar_list.append(r_bar)

            clip_gap = torch.norm(r_bar - r).item()
            clip_rhs = max(torch.norm(r).item() - float(tau.item()), 0.0)
            pre_post_norm = torch.norm(delta_tilde_pre_cap).item()

            residual_checks.append(
                {
                    "is_new": 0.0,
                    "tau": float(tau.item()),
                    "r_norm": float(torch.norm(r).item()),
                    "clip_gap": float(clip_gap),
                    "clip_rhs": float(clip_rhs),
                    "pre_post_norm": float(pre_post_norm),
                }
            )

            if has_small_norm:
                delta_eff_i = messages_eff[i]
                r_norm = delta_eff_i - self.b_norm[cid]
                r_bar_norm = self._clip_by_norm(r_norm, tau) if self.enable_per_client_residual_clip else r_norm
                delta_tilde_norm = self.b_norm[cid] + r_bar_norm
                if self.enable_post_residual_b_cap:
                    delta_tilde_norm = delta_tilde_norm * torch.minimum(
                        self.one, B / (torch.norm(delta_tilde_norm) + self.eps)
                    )
                r_bar_norm_list.append(r_bar_norm)
                delta_tilde_norm_list.append(delta_tilde_norm)
            else:
                r_bar_norm_list.append(r_bar)
                delta_tilde_norm_list.append(delta_tilde)

        processed_tensor = torch.stack(processed, dim=0)

        use_nu_penalty_this_round = self.iteration_count > self.nu_penalty_start_iter
        if use_nu_penalty_this_round:
            nu_values = torch.empty(N, device=device, dtype=dtype)
            for i, cid in enumerate(client_ids):
                nu_values[i] = self.nu[cid].item() if cid in self.nu else median_norm.item()
            nu_mean = nu_values.mean()
            nu_excess = torch.where(
                nu_values > 0.9,
                torch.clamp(nu_values - nu_mean, min=0.0),
                torch.zeros_like(nu_values),
            )
            raw_weights = 1.0 / (1.0 + self.nu_penalty_alpha * nu_excess)
            weights_tensor = raw_weights / (raw_weights.sum() + self.eps)
            for _ in range(10):
                if (weights_tensor > self.nu_weight_max).any().item():
                    weights_tensor = torch.clamp(weights_tensor, max=self.nu_weight_max)
                    weights_tensor = weights_tensor / (weights_tensor.sum() + self.eps)
                else:
                    break
        else:
            weights_tensor = torch.ones(N, device=device, dtype=dtype) / N

        mu_min_t = torch.tensor(self.mu_min, device=device, dtype=dtype)
        nu_min_t = torch.tensor(self.nu_min, device=device, dtype=dtype)

        for i, cid in enumerate(client_ids):
            if cid in new_cids:
                continue
            delta_processed = processed[i]
            r_bar = r_bar_list[i]
            r_bar_norm = r_bar_norm_list[i]
            delta_tilde_norm = delta_tilde_norm_list[i]

            ds_norm = torch.norm(delta_processed) + self.eps
            delta_safe_capped = delta_processed * torch.minimum(self.one, B / ds_norm)
            self.b[cid] = (self.rho_b * self.b[cid] + (1.0 - self.rho_b) * delta_safe_capped).detach()
            self.b_norm[cid] = (self.rho_b * self.b_norm[cid] + (1.0 - self.rho_b) * delta_tilde_norm).detach()

            r_bar_norm_scalar = torch.norm(r_bar_norm) + self.eps
            if has_small_norm and norm_small[i].item():
                r_bar_norm_scalar = torch.norm(delta_tilde_norm) + self.eps
            self.mu[cid] = (self.rho_mu * self.mu[cid] + (1.0 - self.rho_mu) * r_bar_norm_scalar).detach()
            self.mu[cid] = torch.maximum(self.mu[cid], mu_min_t)

            d = torch.norm(r_bar_norm - self.r_prev[cid])
            d_eff = d.clamp(min=self.eps)
            nu_new = (self.rho_nu * self.nu[cid] + (1.0 - self.rho_nu) * d_eff).detach()
            nu_new = torch.maximum(nu_new, nu_min_t)
            self.nu[cid] = nu_new
            self.r_prev[cid] = r_bar_norm.detach().clone()

        self.median_norm_prev = median_norm.item()

        honest_idx = [i for i, cid in enumerate(client_ids) if cid in self.honest_nodes]
        byz_idx = [i for i, cid in enumerate(client_ids) if cid in self.byzantine_nodes]

        return RoundDebug(
            B=float(B.item()),
            messages=messages.detach().clone(),
            messages_clipped=messages_clipped.detach().clone(),
            processed=processed_tensor.detach().clone(),
            weights=weights_tensor.detach().clone(),
            honest_idx=honest_idx,
            byz_idx=byz_idx,
            residual_checks=residual_checks,
        )


def make_messages(n_total: int, n_honest: int, dim: int, g_h: float, byz_scale: float) -> torch.Tensor:
    honest = torch.randn(n_honest, dim, dtype=torch.float64)
    honest = honest / (torch.norm(honest, dim=1, keepdim=True) + 1e-12)
    honest = honest * torch.rand(n_honest, 1, dtype=torch.float64) * g_h

    n_byz = n_total - n_honest
    byz = torch.randn(n_byz, dim, dtype=torch.float64)
    byz = byz / (torch.norm(byz, dim=1, keepdim=True) + 1e-12)
    byz = byz * (g_h * byz_scale) * (0.5 + torch.rand(n_byz, 1, dtype=torch.float64))
    return torch.cat([byz, honest], dim=0)


def check_close(name: str, value: float, rhs: float, tol: float, failures: List[str]) -> None:
    if abs(value - rhs) > tol:
        failures.append(f"{name}: |{value:.6e} - {rhs:.6e}| > {tol:.1e}")


def check_leq(name: str, value: float, rhs: float, tol: float, failures: List[str]) -> None:
    if value > rhs + tol:
        failures.append(f"{name}: {value:.6e} > {rhs:.6e} (+ tol {tol:.1e})")


def run_verification(args) -> int:
    set_seed(args.seed)

    n_total = args.n_total
    n_honest = args.n_honest
    n_byz = n_total - n_honest
    client_ids = list(range(n_total))

    failures: List[str] = []

    hrac_no_post = HRACDebug(
        honest_nodes=set(range(n_byz, n_total)),
        byzantine_nodes=set(range(n_byz)),
        rho_b=0.98,
        rho_mu=0.95,
        rho_nu=0.95,
        c=2.5,
        c_g=3.0,
        enable_logging=False,
        nu_penalty_start_iter=1,
        enable_post_residual_b_cap=False,
    )

    hrac_with_post = HRACDebug(
        honest_nodes=set(range(n_byz, n_total)),
        byzantine_nodes=set(range(n_byz)),
        rho_b=0.98,
        rho_mu=0.95,
        rho_nu=0.95,
        c=2.5,
        c_g=3.0,
        enable_logging=False,
        nu_penalty_start_iter=1,
        enable_post_residual_b_cap=True,
    )

    # Warm-up round for state initialization.
    init_messages = make_messages(n_total, n_honest, args.dim, args.g_h, args.byz_scale)
    hrac_no_post.step_with_debug(init_messages, client_ids=client_ids)
    hrac_with_post.step_with_debug(init_messages, client_ids=client_ids)

    max_mu_seen = 0.0
    max_b_norm_seen = 0.0

    for round_idx in range(args.rounds):
        messages = make_messages(n_total, n_honest, args.dim, args.g_h, args.byz_scale)
        dbg_no_post = hrac_no_post.step_with_debug(messages, client_ids=client_ids)
        dbg_post = hrac_with_post.step_with_debug(messages, client_ids=client_ids)

        B = dbg_no_post.B
        B_max = hrac_no_post.c_g * args.g_h

        # 1. Global clipping bound from implementation.
        for i in range(n_total):
            check_leq(
                f"round{round_idx}.global_clip[{i}]",
                float(torch.norm(dbg_no_post.messages_clipped[i]).item()),
                B,
                args.tol,
                failures,
            )

        # 2. Residual clipping identity in the no-post-cap variant.
        for i, info in enumerate(dbg_no_post.residual_checks):
            if info["is_new"] > 0.5:
                continue
            check_close(
                f"round{round_idx}.clip_identity[{i}]",
                info["clip_gap"],
                info["clip_rhs"],
                args.tol,
                failures,
            )

        # 3. Post-cap variant satisfies stronger norm bound ||processed_i|| <= B.
        for i in range(n_total):
            check_leq(
                f"round{round_idx}.post_cap[{i}]",
                float(torch.norm(dbg_post.processed[i]).item()),
                dbg_post.B,
                args.tol,
                failures,
            )

        # 4. Weight simplex and max-cap.
        w = dbg_post.weights
        check_close(
            f"round{round_idx}.weight_sum",
            float(w.sum().item()),
            1.0,
            args.tol,
            failures,
        )
        check_leq(
            f"round{round_idx}.weight_nonneg_min",
            float((-w.min()).clamp(min=0).item()),
            0.0,
            args.tol,
            failures,
        )
        check_leq(
            f"round{round_idx}.weight_max",
            float(w.max().item()),
            hrac_with_post.nu_weight_max,
            args.tol,
            failures,
        )

        # 5. Jensen bound for aggregate.
        g = (w[:, None] * dbg_post.processed).sum(dim=0)
        lhs = float(torch.norm(g).pow(2).item())
        rhs = float((w * torch.norm(dbg_post.processed, dim=1).pow(2)).sum().item())
        check_leq(f"round{round_idx}.jensen_g", lhs, rhs, args.tol, failures)

        # 6. Median-cap protection under honest majority for this synthetic setup.
        honest_norm_max = float(torch.norm(messages[dbg_no_post.honest_idx], dim=1).max().item())
        check_leq(
            f"round{round_idx}.median_cap_honest",
            B,
            hrac_no_post.c_g * honest_norm_max,
            args.tol,
            failures,
        )
        check_leq(
            f"round{round_idx}.Bmax_from_honest",
            B,
            B_max,
            args.tol,
            failures,
        )

        # 7. Empirical boundedness of b and mu under bounded honest majority setup.
        for cid in client_ids:
            max_mu_seen = max(max_mu_seen, float(hrac_no_post.mu[cid].item()))
            max_b_norm_seen = max(max_b_norm_seen, float(torch.norm(hrac_no_post.b[cid]).item()))
            check_leq(
                f"round{round_idx}.b_bound[c{cid}]",
                float(torch.norm(hrac_no_post.b[cid]).item()),
                B_max,
                1e-6,
                failures,
            )
            # With this implementation, small-norm lifting can make mu track ||delta_tilde_norm||,
            # so the safe empirical upper bound is max(2 B_max, B_max).
            check_leq(
                f"round{round_idx}.mu_bound[c{cid}]",
                float(hrac_no_post.mu[cid].item()),
                2.0 * B_max + 1e-6,
                1e-6,
                failures,
            )

    print("=== HRAC Key Inequality Verification ===")
    print(f"rounds={args.rounds}, n_total={n_total}, n_honest={n_honest}, dim={args.dim}")
    print(f"seed={args.seed}, G_H={args.g_h}, byz_scale={args.byz_scale}")
    print(f"max_mu_seen={max_mu_seen:.6f}, max_b_norm_seen={max_b_norm_seen:.6f}")

    if failures:
        print(f"FAILED: {len(failures)} checks")
        for item in failures[:50]:
            print(" -", item)
        if len(failures) > 50:
            print(f" - ... and {len(failures) - 50} more")
        return 1

    print("PASSED: all checked implementation-level inequalities held.")
    print("Checked items:")
    print(" - global clip bound ||messages_clipped_i|| <= B")
    print(" - residual clipping identity ||r_bar-r|| = (||r||-tau)_+ (post-cap disabled variant)")
    print(" - post-cap bound ||processed_i|| <= B (default variant)")
    print(" - weight simplex, nonnegativity, and max-weight cap")
    print(" - Jensen bound ||g||^2 <= sum_i w_i ||processed_i||^2")
    print(" - honest-majority median-cap bound B <= c_g * max_honest ||Delta_i||")
    print(" - empirical boundedness of b_i and mu_i under bounded-honest synthetic rounds")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify key HRAC implementation inequalities.")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--n-total", type=int, default=10)
    parser.add_argument("--n-honest", type=int, default=7)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--g-h", type=float, default=1.0)
    parser.add_argument("--byz-scale", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--tol", type=float, default=1e-8)
    args = parser.parse_args()
    return run_verification(args)


if __name__ == "__main__":
    raise SystemExit(main())
