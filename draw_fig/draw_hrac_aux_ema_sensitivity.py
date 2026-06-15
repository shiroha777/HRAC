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

RUNS = [
    ("Default EMA", RECORD_DIR / "0.87hrac-log-label_flipping.txt"),
    ("History/scale EMA = 0.90", RECORD_DIR / "0.90EMAhrac-log-label_flipping.txt"),
    ("History/scale EMA = 0.95", RECORD_DIR / "0.95EMAhrac-log-label_flipping.txt"),
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


def save_figure(fig, stem):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUT_DIR / f"{stem}.pdf"
    png_path = OUT_DIR / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    print(f"WROTE {pdf_path}")
    print(f"WROTE {png_path}")


def plot_accuracy(all_runs):
    fig, ax = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    colours = ["#2ca02c", "#1f77b4", "#ff7f0e"]

    for (label, accuracy, _summary), colour in zip(all_runs, colours):
        iterations = sorted(accuracy)
        values = [accuracy[it] for it in iterations]
        ax.plot(iterations, values, label=label, linewidth=2.0, color=colour)

    ax.set_title(r"Auxiliary EMA robustness | label flipping | IID | $b=3$")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 0.75)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True)
    save_figure(fig, "CMomentum_cifar10_b3_label_flipping_iid_auxema_accuracy")
    plt.close(fig)


def plot_diagnostics(all_runs):
    fig_height = 2.65 * len(all_runs) + 0.4
    fig, axes = plt.subplots(len(all_runs), 3, figsize=(12.0, fig_height), sharex=True)

    for row_idx, (label, _accuracy, summary) in enumerate(all_runs):
        iterations = [row["iteration"] for row in summary]
        benign_nu = [row["benign_nu"] for row in summary]
        attacker_nu = [row["attacker_nu"] for row in summary]
        benign_weight = [row["benign_weight"] for row in summary]
        attacker_weight = [row["attacker_weight"] for row in summary]
        bottom_counts = [row["attacker_bottom_count"] for row in summary]

        axes[row_idx, 0].plot(iterations, benign_nu, color="#1f77b4", linewidth=1.6, label="benign mean")
        axes[row_idx, 0].plot(iterations, attacker_nu, color="#c23b50", linewidth=1.6, label="attacker mean")
        axes[row_idx, 0].axhline(0.9, color="0.35", linestyle="--", linewidth=1.0, alpha=0.75)
        axes[row_idx, 0].set_ylabel(label)

        axes[row_idx, 1].plot(iterations, benign_weight, color="#1f77b4", linewidth=1.6)
        axes[row_idx, 1].plot(iterations, attacker_weight, color="#c23b50", linewidth=1.6)

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

    fig.tight_layout(rect=[0.0, 0.055, 1.0, 0.96])
    save_figure(fig, "HRAC_diagnostics_label_flipping_iid_auxema_sensitivity")
    plt.close(fig)


def main():
    all_runs = []
    for label, path in RUNS:
        if not path.exists():
            raise FileNotFoundError(path)
        accuracy, clients_by_iter = parse_log(path)
        summary = summarise_clients(clients_by_iter)
        all_runs.append((label, accuracy, summary))
        final_iter = max(accuracy)
        print(f"{label}: final_iter={final_iter}, final_acc={accuracy[final_iter]:.4f}")

    plot_accuracy(all_runs)
    plot_diagnostics(all_runs)


if __name__ == "__main__":
    main()
