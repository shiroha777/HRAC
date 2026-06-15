import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECORD_DIR = (
    PROJECT_ROOT
    / "record"
    / "NeuralNetwork_cifar10"
    / "Centralized_n=10_b=3"
    / "iidPartition"
)
OUT_DIR = PROJECT_ROOT / "ByrdLab" / "paper" / "imgs"

RHO_LOGS = [
    ("0.70", RECORD_DIR / "0.70hrac-log-label_flipping.txt"),
    ("0.80", RECORD_DIR / "0.80hrac-log-label_flipping.txt"),
    ("0.87", RECORD_DIR / "0.87hrac-log-label_flipping.txt"),
    ("0.90", RECORD_DIR / "0.90hrac-log-label_flipping.txt"),
    ("0.95", RECORD_DIR / "0.95hrac-log-label_flipping.txt"),
]

ITER_RE = re.compile(r"\[HRAC\] Iteration (\d+) Statistics:")
ACC_RE = re.compile(r"\]\s+Accuracy:\s+([0-9.eE+-]+)")
CLIENT_RE = re.compile(
    r"\]\s+(\d+)\s+\|\s+"
    r"([0-9.eE+-]+)\s+\|\s+"
    r"([0-9.eE+-]+)\s+\|\s+"
    r"([0-9.eE+-]+)\s+\|\s+"
    r"([0-9.eE+-]+)\s+\|\s+"
    r"([0-9.eE+-]+)\s+\|\s+"
    r"([0-9.eE+-]+)\s+\|\s+"
    r"([0-9.eE+-]+)\s+\|\s+"
    r"(YES|NO)"
)


def parse_log(path):
    accuracy = {}
    clients_by_iter = defaultdict(list)
    current_iter = None

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            iter_match = ITER_RE.search(line)
            if iter_match:
                current_iter = int(iter_match.group(1))
                continue

            if current_iter is None:
                continue

            acc_match = ACC_RE.search(line)
            if acc_match:
                accuracy[current_iter] = float(acc_match.group(1))
                continue

            client_match = CLIENT_RE.search(line)
            if not client_match:
                continue

            clients_by_iter[current_iter].append(
                {
                    "client_id": int(client_match.group(1)),
                    "nu": float(client_match.group(4)),
                    "weight": float(client_match.group(8)),
                    "is_attacker": client_match.group(9) == "YES",
                }
            )

    return accuracy, clients_by_iter


def summarise_clients(clients_by_iter, bottom_k=3):
    rows = []
    for iteration, clients in sorted(clients_by_iter.items()):
        benign = [row for row in clients if not row["is_attacker"]]
        attackers = [row for row in clients if row["is_attacker"]]
        if not benign or not attackers:
            continue

        bottom = sorted(clients, key=lambda row: row["weight"])[:bottom_k]
        rows.append(
            {
                "iteration": iteration,
                "benign_nu": sum(row["nu"] for row in benign) / len(benign),
                "attacker_nu": sum(row["nu"] for row in attackers) / len(attackers),
                "benign_weight": sum(row["weight"] for row in benign) / len(benign),
                "attacker_weight": sum(row["weight"] for row in attackers) / len(attackers),
                "attacker_bottom_count": sum(1 for row in bottom if row["is_attacker"]),
            }
        )
    return rows


def first_sustained_weight_gap(summary, min_gap=0.002, window=3):
    for idx, row in enumerate(summary):
        block = summary[idx : idx + window]
        if len(block) < window:
            continue
        if all(item["attacker_weight"] < item["benign_weight"] - min_gap for item in block):
            return row["iteration"], row["attacker_weight"]
    return None


def save_figure(fig, stem):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=220, bbox_inches="tight")
    print(f"WROTE {OUT_DIR / f'{stem}.pdf'}")


def plot_accuracy(all_runs):
    fig, ax = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    colours = {
        "0.70": "#9467bd",
        "0.80": "#1f77b4",
        "0.87": "#2ca02c",
        "0.90": "#ff7f0e",
        "0.95": "#d62728",
    }

    for rho, accuracy, _summary in all_runs:
        iterations = sorted(accuracy)
        values = [accuracy[it] for it in iterations]
        ax.plot(iterations, values, label=rf"$\rho_\nu={rho}$", linewidth=1.9, color=colours[rho])

    ax.set_title(r"Label flipping | IID | $b=3$")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 0.75)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True)
    save_figure(fig, "CMomentum_cifar10_b3_label_flipping_iid_rhonu_accuracy")
    plt.close(fig)


def plot_diagnostics(all_runs):
    fig_height = 2.75 * len(all_runs) + 0.4
    fig, axes = plt.subplots(len(all_runs), 3, figsize=(12.2, fig_height), sharex=True)

    for row_idx, (rho, _accuracy, summary) in enumerate(all_runs):
        iterations = [row["iteration"] for row in summary]
        benign_nu = [row["benign_nu"] for row in summary]
        attacker_nu = [row["attacker_nu"] for row in summary]
        benign_weight = [row["benign_weight"] for row in summary]
        attacker_weight = [row["attacker_weight"] for row in summary]
        bottom_counts = [row["attacker_bottom_count"] for row in summary]

        axes[row_idx, 0].plot(iterations, benign_nu, label="benign mean", color="#1f77b4", linewidth=1.6)
        axes[row_idx, 0].plot(iterations, attacker_nu, label="attacker mean", color="#c23b50", linewidth=1.6)
        axes[row_idx, 0].axhline(0.9, color="0.35", linestyle="--", linewidth=1.0, alpha=0.75)
        if row_idx == 0:
            axes[row_idx, 0].text(
                iterations[-1],
                0.9,
                r"$\nu_i=0.9$",
                color="0.25",
                fontsize=8,
                ha="right",
                va="bottom",
            )
        axes[row_idx, 0].set_ylabel(rf"$\rho_\nu={rho}$")

        axes[row_idx, 1].plot(iterations, benign_weight, label="benign mean", color="#1f77b4", linewidth=1.6)
        axes[row_idx, 1].plot(iterations, attacker_weight, label="attacker mean", color="#c23b50", linewidth=1.6)
        separation = first_sustained_weight_gap(summary)
        if separation is not None:
            sep_iter, sep_weight = separation
            axes[row_idx, 1].scatter(
                [sep_iter],
                [sep_weight],
                marker="o",
                s=28,
                color="#c23b50",
                edgecolor="white",
                linewidth=0.6,
                zorder=5,
            )
            axes[row_idx, 1].axvline(sep_iter, color="#c23b50", linestyle=":", linewidth=1.0, alpha=0.65)
            axes[row_idx, 1].text(
                sep_iter,
                sep_weight - 0.006,
                f"{sep_iter / 1000:.1f}k",
                color="#7a1f2f",
                fontsize=8,
                ha="center",
                va="top",
            )

        axes[row_idx, 2].plot(iterations, bottom_counts, color="#5c5c5c", linewidth=1.5)
        axes[row_idx, 2].set_ylim(-0.1, 3.1)

        for axis in axes[row_idx]:
            axis.grid(True, alpha=0.25)
            axis.axvline(50, color="0.5", linestyle=":", linewidth=1)

    axes[0, 0].set_title(r"Residual-change statistic $\nu_i$")
    axes[0, 1].set_title("Normalised aggregation weight")
    axes[0, 2].set_title("Attackers in bottom-3 weights")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.005))

    for axis in axes[-1]:
        axis.set_xlabel("Iterations")

    fig.tight_layout(rect=[0.0, 0.045, 1.0, 0.955])
    save_figure(fig, "HRAC_diagnostics_label_flipping_iid_rhonu_sensitivity")
    plt.close(fig)


def main():
    all_runs = []
    for rho, path in RHO_LOGS:
        if not path.exists():
            raise FileNotFoundError(path)
        accuracy, clients_by_iter = parse_log(path)
        summary = summarise_clients(clients_by_iter, bottom_k=3)
        all_runs.append((rho, accuracy, summary))
        final_iter = max(accuracy)
        print(f"rho={rho}: final_iter={final_iter}, final_acc={accuracy[final_iter]:.4f}")

    plot_accuracy(all_runs)
    plot_diagnostics(all_runs)


if __name__ == "__main__":
    main()
