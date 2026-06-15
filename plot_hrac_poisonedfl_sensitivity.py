import csv
import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
RECORD_DIR = PROJECT_ROOT / "record" / "NeuralNetwork_cifar10" / "Centralized_n=10_b=3" / "LabelSeperation"
OUT_DIR = PROJECT_ROOT / "ByrdLab" / "paper" / "imgs" / "hrac_sensitivity_poisonedfl_b3"

ATTACKERS = {7, 8, 9}
HONEST = set(range(7))


RUNS = [
    {
        "key": "default",
        "label": "default",
        "family": "default",
        "value": None,
        "record": "CMomentum_poisonedfl_HRAC_twoStageLR",
        "log": "hrac-log-poisonedfl.txt",
    },
    {
        "key": "rho_b_0.95",
        "label": r"$\rho_b=0.95$",
        "family": r"$\rho_b$",
        "value": 0.95,
        "record": "CMomentum_poisonedfl_HRAC_rhob0p95_twoStageLR",
        "log": "hrac-rhob0p95-log-poisonedfl.txt",
    },
    {
        "key": "rho_b_0.995",
        "label": r"$\rho_b=0.995$",
        "family": r"$\rho_b$",
        "value": 0.995,
        "record": "CMomentum_poisonedfl_HRAC_rhob0p995_twoStageLR",
        "log": "hrac-rhob0p995-log-poisonedfl.txt",
    },
    {
        "key": "rho_mu_0.90",
        "label": r"$\rho_\mu=0.90$",
        "family": r"$\rho_\mu$",
        "value": 0.90,
        "record": "CMomentum_poisonedfl_HRAC_rhomu0p9_twoStageLR",
        "log": "hrac-rhomu0p9-log-poisonedfl.txt",
    },
    {
        "key": "rho_mu_0.98",
        "label": r"$\rho_\mu=0.98$",
        "family": r"$\rho_\mu$",
        "value": 0.98,
        "record": "CMomentum_poisonedfl_HRAC_rhomu0p98_twoStageLR",
        "log": "hrac-rhomu0p98-log-poisonedfl.txt",
    },
    {
        "key": "rho_g_0.90",
        "label": r"$\rho_g=0.90$",
        "family": r"$\rho_g$",
        "value": 0.90,
        "record": "CMomentum_poisonedfl_HRAC_rhog0p9_twoStageLR",
        "log": "hrac-rhog0p9-log-poisonedfl.txt",
    },
    {
        "key": "rho_g_0.98",
        "label": r"$\rho_g=0.98$",
        "family": r"$\rho_g$",
        "value": 0.98,
        "record": "CMomentum_poisonedfl_HRAC_rhog0p98_twoStageLR",
        "log": "hrac-rhog0p98-log-poisonedfl.txt",
    },
]

DEFAULT_PARAM_LABELS = {
    r"$\rho_b$": r"default $\rho_b=0.98$",
    r"$\rho_\mu$": r"default $\rho_\mu=0.95$",
    r"$\rho_g$": r"default $\rho_g=0.95$",
}


def to_float(value):
    if hasattr(value, "detach"):
        return float(value.detach().cpu().item())
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def load_record(run):
    path = RECORD_DIR / run["record"]
    with path.open("rb") as f:
        record = pickle.load(f)
    acc = np.array([to_float(x) for x in record["acc_path"]], dtype=float)
    total_iterations = int(record.get("total_iterations", (len(acc) - 1) * 100))
    iterations = np.linspace(0, total_iterations, len(acc))
    return record, iterations, acc


SUMMARY_RE = re.compile(r"iter=(\d+)\s+acc=([0-9.]+).*?\|\s+c0:")
CLIENT_RE = re.compile(
    r"c(?P<id>\d+):\s+tau=(?P<tau>[-+0-9.eE]+)\s+"
    r"mu=(?P<mu>[-+0-9.eE]+)\s+nu=(?P<nu>[-+0-9.eE]+)\s+w=(?P<w>[-+0-9.eE]+)"
)
RCLIP_TOTAL_RE = re.compile(r"rclip_total=(\d+)")
RCLIP_H_RE = re.compile(r"rclip_h=(\d+)")
RCLIP_B_RE = re.compile(r"rclip_b=(\d+)")


def parse_log(run):
    path = RECORD_DIR / run["log"]
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "iter=" not in line or "| c0:" not in line:
            continue
        m = SUMMARY_RE.search(line)
        if not m:
            continue
        clients = {}
        for cm in CLIENT_RE.finditer(line):
            cid = int(cm.group("id"))
            clients[cid] = {
                "tau": float(cm.group("tau")),
                "mu": float(cm.group("mu")),
                "nu": float(cm.group("nu")),
                "w": float(cm.group("w")),
            }
        if len(clients) != 10:
            continue
        weights = {cid: clients[cid]["w"] for cid in clients}
        bottom3 = sorted(weights, key=weights.get)[:3]
        rclip_total = RCLIP_TOTAL_RE.search(line)
        rclip_h = RCLIP_H_RE.search(line)
        rclip_b = RCLIP_B_RE.search(line)
        rows.append(
            {
                "iteration": int(m.group(1)),
                "acc": float(m.group(2)),
                "honest_nu": float(np.mean([clients[i]["nu"] for i in HONEST])),
                "attacker_nu": float(np.mean([clients[i]["nu"] for i in ATTACKERS])),
                "honest_w": float(np.mean([clients[i]["w"] for i in HONEST])),
                "attacker_w": float(np.mean([clients[i]["w"] for i in ATTACKERS])),
                "bottom3_attackers": sum(1 for i in bottom3 if i in ATTACKERS),
                "rclip_total": int(rclip_total.group(1)) if rclip_total else None,
                "rclip_h": int(rclip_h.group(1)) if rclip_h else None,
                "rclip_b": int(rclip_b.group(1)) if rclip_b else None,
            }
        )
    return rows


def collect():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for run in RUNS:
        record, iterations, acc = load_record(run)
        diag = parse_log(run)
        final_diag = diag[-1] if diag else {}
        first_bottom3 = next((r["iteration"] for r in diag if r["bottom3_attackers"] == 3), None)
        late_rows = [r for r in diag if r["iteration"] >= 1000]
        mean_late_bottom3 = (
            float(np.mean([r["bottom3_attackers"] for r in late_rows])) if late_rows else np.nan
        )
        result = {
            **run,
            "record_obj": record,
            "iterations": iterations,
            "acc": acc,
            "diag": diag,
            "final_acc": float(acc[-1]),
            "best_acc": float(np.max(acc)),
            "first_bottom3": first_bottom3,
            "late_bottom3_mean": mean_late_bottom3,
            "final_honest_nu": final_diag.get("honest_nu", np.nan),
            "final_attacker_nu": final_diag.get("attacker_nu", np.nan),
            "final_nu_gap": final_diag.get("attacker_nu", np.nan) - final_diag.get("honest_nu", np.nan),
            "final_honest_w": final_diag.get("honest_w", np.nan),
            "final_attacker_w": final_diag.get("attacker_w", np.nan),
            "final_rclip_total": final_diag.get("rclip_total", np.nan),
            "final_rclip_round_h": final_diag.get("rclip_h", np.nan),
            "final_rclip_round_b": final_diag.get("rclip_b", np.nan),
        }
        results.append(result)
    return results


def plot_accuracy(results):
    default = next(r for r in results if r["key"] == "default")
    families = [r"$\rho_b$", r"$\rho_\mu$", r"$\rho_g$"]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.2), sharey=True)
    for ax, family in zip(axes, families):
        ax.plot(
            default["iterations"],
            default["acc"],
            color="0.35",
            lw=2.0,
            label=DEFAULT_PARAM_LABELS[family],
        )
        for r in [x for x in results if x["family"] == family]:
            ax.plot(r["iterations"], r["acc"], lw=1.8, label=r["label"])
        ax.annotate(
            f"default final={default['final_acc']:.4f}",
            xy=(default["iterations"][-1], default["acc"][-1]),
            xytext=(-112, -20),
            textcoords="offset points",
            fontsize=8,
            arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "0.35"},
            color="0.25",
        )
        ax.set_title(f"{family} sensitivity")
        ax.set_xlabel("Communication iteration")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
    axes[0].set_ylabel("Test accuracy")
    fig.suptitle("HRAC parameter sensitivity under PoisonedFL, CIFAR-10 non-IID, b=3")
    fig.tight_layout()
    out = OUT_DIR / "hrac_poisonedfl_b3_sensitivity_accuracy.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def plot_diagnostics(results):
    default = next(r for r in results if r["key"] == "default")
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 8.2), sharex=True)
    axes = axes.ravel()
    for r in results:
        diag = r["diag"]
        xs = [d["iteration"] for d in diag]
        axes[0].plot(xs, [d["attacker_w"] for d in diag], lw=1.6, label=r["label"])
        axes[1].plot(xs, [d["attacker_nu"] for d in diag], lw=1.6, label=r["label"])
        axes[2].plot(xs, [d["honest_nu"] for d in diag], lw=1.6, label=r["label"])
        axes[3].plot(xs, [d["bottom3_attackers"] for d in diag], lw=1.6, label=r["label"])
    axes[0].set_title("Mean attacker weight")
    axes[1].set_title(r"Mean attacker $\nu$")
    axes[2].set_title(r"Mean benign $\nu$")
    axes[3].set_title("Attackers in bottom-3 weights")
    for ax in axes:
        ax.set_xlabel("Communication iteration")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Mean weight")
    axes[1].set_ylabel(r"Mean $\nu$")
    axes[2].set_ylabel(r"Mean $\nu$")
    axes[3].set_ylabel("Count")
    axes[3].set_ylim(-0.1, 3.1)
    axes[3].set_yticks([0, 1, 2, 3])

    default_diag = default["diag"][-1]
    annotations = [
        (0, default_diag["attacker_w"], f"default={default_diag['attacker_w']:.4f}", (-96, 14)),
        (1, default_diag["attacker_nu"], f"default={default_diag['attacker_nu']:.2f}", (-92, 12)),
        (2, default_diag["honest_nu"], f"default={default_diag['honest_nu']:.2f}", (-92, 12)),
        (3, default_diag["bottom3_attackers"], f"default={default_diag['bottom3_attackers']}", (-84, -28)),
    ]
    for ax_idx, y_value, text, offset in annotations:
        ax = axes[ax_idx]
        ax.annotate(
            text,
            xy=(default_diag["iteration"], y_value),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
            arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "0.35"},
            color="0.25",
        )

    axes[3].legend(fontsize=8, ncol=2, loc="lower right")
    fig.suptitle("HRAC diagnostic sensitivity under PoisonedFL, CIFAR-10 non-IID, b=3")
    fig.tight_layout()
    out = OUT_DIR / "hrac_poisonedfl_b3_sensitivity_diagnostics.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def write_summary(results):
    out = OUT_DIR / "hrac_poisonedfl_b3_sensitivity_summary.csv"
    fields = [
        "key",
        "label",
        "family",
        "value",
        "final_acc",
        "best_acc",
        "first_bottom3",
        "late_bottom3_mean",
        "final_attacker_w",
        "final_honest_w",
        "final_nu_gap",
        "final_attacker_nu",
        "final_honest_nu",
        "final_rclip_total",
        "final_rclip_round_h",
        "final_rclip_round_b",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k) for k in fields})
    return out


def main():
    results = collect()
    paths = [
        plot_accuracy(results),
        plot_diagnostics(results),
        write_summary(results),
    ]

    print("Generated:")
    for p in paths:
        print(f"  {p}")
    print()
    print("Summary:")
    for r in results:
        print(
            f"{r['label']:>16s} | final={r['final_acc']:.4f} "
            f"best={r['best_acc']:.4f} attacker_w={r['final_attacker_w']:.4f} "
            f"nu_gap={r['final_nu_gap']:.4f} first_bottom3={r['first_bottom3']} "
            f"rclip_total={r['final_rclip_total']}"
        )


if __name__ == "__main__":
    main()
