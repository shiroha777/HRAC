import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ATTACKS = ["label_flipping", "MSA", "HisMSA", "ALIE", "IPM", "poisonedfl"]
PARTITIONS = {
    "iid": "iidPartition",
    "noniid": "LabelSeperation",
}


ITER_RE = re.compile(r"\[HRAC\] Iteration (\d+) Statistics:")
FLOAT = r"[0-9.eE+-]+"
CLIENT_RE = re.compile(
    rf"\]\s+(?P<client_id>\d+)\s+\|\s+"
    rf"(?P<tau>{FLOAT})\s+\|\s+"
    rf"(?P<mu>{FLOAT})\s+\|\s+"
    rf"(?P<nu>{FLOAT})\s+\|\s+"
    rf"(?P<norm_pre>{FLOAT})\s+\|\s+"
    rf"(?P<norm_post>{FLOAT})\s+\|\s+"
    rf"(?P<norm_final>{FLOAT})\s+\|\s+"
    rf"(?P<weight>{FLOAT})\s+\|"
    rf"(?:\s+(?P<rclip>{FLOAT})\s+\|)?\s+"
    rf"(?P<attacker>YES|NO)"
)


def parse_log(path):
    rows = defaultdict(list)
    current_iter = None

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            iter_match = ITER_RE.search(line)
            if iter_match:
                current_iter = int(iter_match.group(1))
                continue

            if current_iter is None:
                continue

            client_match = CLIENT_RE.search(line)
            if not client_match:
                continue

            client_id = int(client_match.group("client_id"))
            nu = float(client_match.group("nu"))
            weight = float(client_match.group("weight"))
            is_attacker = client_match.group("attacker") == "YES"
            rows[current_iter].append(
                {
                    "client_id": client_id,
                    "nu": nu,
                    "weight": weight,
                    "is_attacker": is_attacker,
                }
            )

    return rows


def summarise(rows, bottom_k):
    summary = []
    for iteration, clients in sorted(rows.items()):
        if not clients:
            continue

        benign = [row for row in clients if not row["is_attacker"]]
        attackers = [row for row in clients if row["is_attacker"]]
        if not benign or not attackers:
            continue

        sorted_by_weight = sorted(clients, key=lambda row: row["weight"])
        bottom = sorted_by_weight[:bottom_k]
        attacker_bottom_count = sum(1 for row in bottom if row["is_attacker"])

        summary.append(
            {
                "iteration": iteration,
                "benign_nu": sum(row["nu"] for row in benign) / len(benign),
                "attacker_nu": sum(row["nu"] for row in attackers) / len(attackers),
                "benign_weight": sum(row["weight"] for row in benign) / len(benign),
                "attacker_weight": sum(row["weight"] for row in attackers) / len(attackers),
                "attacker_bottom_count": attacker_bottom_count,
            }
        )

    return summary


def plot_summary(summary, attack, partition_label, bottom_k, output_base):
    iterations = [row["iteration"] for row in summary]
    benign_nu = [row["benign_nu"] for row in summary]
    attacker_nu = [row["attacker_nu"] for row in summary]
    benign_weight = [row["benign_weight"] for row in summary]
    attacker_weight = [row["attacker_weight"] for row in summary]
    bottom_counts = [row["attacker_bottom_count"] for row in summary]

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.4), constrained_layout=True)
    fig.suptitle(f"HRAC diagnostics on {partition_label} CIFAR-10 under {attack} (b={bottom_k})")

    axes[0].plot(iterations, benign_nu, label="benign mean", color="#1f77b4", linewidth=1.8)
    axes[0].plot(iterations, attacker_nu, label="attacker mean", color="#c23b50", linewidth=1.8)
    axes[0].set_title(r"Residual-change statistic $\nu$")
    axes[0].set_xlabel("Communication iteration")
    axes[0].set_ylabel(r"Mean $\nu_i$")
    axes[0].legend(frameon=False)

    axes[1].plot(iterations, benign_weight, label="benign mean", color="#1f77b4", linewidth=1.8)
    axes[1].plot(iterations, attacker_weight, label="attacker mean", color="#c23b50", linewidth=1.8)
    axes[1].set_title("Normalised aggregation weight")
    axes[1].set_xlabel("Communication iteration")
    axes[1].set_ylabel(r"Mean $w_i$")
    axes[1].legend(frameon=False)

    axes[2].plot(iterations, bottom_counts, color="#5c5c5c", linewidth=1.6)
    axes[2].set_title(f"Attackers in bottom-{bottom_k} weights")
    axes[2].set_xlabel("Communication iteration")
    axes[2].set_ylabel("Count")
    axes[2].set_ylim(-0.1, bottom_k + 0.1)

    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.axvline(50, color="0.5", linestyle=":", linewidth=1)

    fig.savefig(output_base.with_suffix(".pdf"))
    fig.savefig(output_base.with_suffix(".png"), dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--b", type=int, required=True)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    record_root = repo_root / "record" / "NeuralNetwork_cifar10" / f"Centralized_n=10_b={args.b}"
    output_dir = repo_root / "ByrdLab" / "paper" / "imgs" / f"hrac_diagnostics_b{args.b}_all_attacks"
    output_dir.mkdir(parents=True, exist_ok=True)

    for attack in ATTACKS:
        for partition_key, partition_dir in PARTITIONS.items():
            log_path = record_root / partition_dir / f"hrac-log-{attack}.txt"
            if not log_path.exists():
                print(f"SKIP missing: {log_path}")
                continue

            rows = parse_log(log_path)
            summary = summarise(rows, bottom_k=args.b)
            if not summary:
                print(f"SKIP empty: {log_path}")
                continue

            partition_label = "IID" if partition_key == "iid" else "Non-IID"
            output_base = output_dir / f"HRAC_diagnostics_{attack}_{partition_key}"
            plot_summary(summary, attack, partition_label, args.b, output_base)
            print(f"WROTE {output_base.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
