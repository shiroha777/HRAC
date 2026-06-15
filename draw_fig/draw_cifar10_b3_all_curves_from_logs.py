"""Draw CIFAR-10 accuracy curves from cached experiment logs.

The first row uses benign b=0 runs. The attack rows use the Byzantine size set
by CIFAR10_BYZANTINE_SIZE, defaulting to b=3. Available centred-clipping records
are displayed simply as "CC" so that the figures do not expose a specific tau
value.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

__FILE_DIR__ = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(__FILE_DIR__)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ByrdLab.library.cache_io import isfile_in_cache, load_file_in_cache, set_cache_path


RECORD_DIR = os.path.join(project_root, "record")
TASK_GRAPH = "NeuralNetwork_cifar10"
BYZANTINE_SIZE = int(os.environ.get("CIFAR10_BYZANTINE_SIZE", "3"))
GRAPH_NAME = f"Centralized_n=10_b={BYZANTINE_SIZE}"
GRAPH_NAME_B0 = "Centralized_n=10_b=0"
FIG_STEM_PREFIX = f"CMomentum_cifar10_b{BYZANTINE_SIZE}"

PARTITIONS = [
    ("iid", "iidPartition"),
    ("noniid", "LabelSeperation"),
]
set_cache_path(RECORD_DIR)

ROW_B0_LABEL = "Baseline (b=0)"
ATTACK_ORDER = [ROW_B0_LABEL, "label_flipping", "MSA", "HisMSA", "ALIE", "IPM", "poisonedfl"]
ATTACK_FILE_PREFIX = {
    "MSA": "CMomentum_MSA_MSA",
    "HisMSA": "CMomentum_HisMSA_HisMSA",
    "baseline": "CMomentum_baseline",
    "label_flipping": "CMomentum_label_flipping",
    "ALIE": "CMomentum_ALIE",
    "IPM": "CMomentum_IPM",
    "poisonedfl": "CMomentum_poisonedfl",
}

AGG_DISPLAY = {
    "mean": "Mean",
    "trimmed_mean": "Trimmed Mean",
    "faba": "FABA",
    "CC_tau=0.1": "CC",
    "CC_tau=0.3": "CC",
    "CC_tau=100": "CC",
    "LFighter": "LFighter",
    "HRAC": "HRAC",
}
EXCLUDED_METHODS = {"Trimmed Mean"}

COLORS = plt.cm.tab10(np.linspace(0, 1, 10))
METHOD_COLORS = {
    "Mean": COLORS[0],
    "Trimmed Mean": COLORS[1],
    "FABA": COLORS[2],
    "CC": (0.9, 0.45, 0.0),
    "LFighter": COLORS[5],
    "HRAC": COLORS[4],
}
PREFERRED_METHOD_ORDER = ["Mean", "FABA", "CC", "LFighter", "HRAC"]


def convert_to_list(data):
    """Convert cached acc_path values to a plain list of floats."""
    if data is None:
        return None
    try:
        import torch

        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy() if data.is_cuda else data.detach().numpy()
    except ImportError:
        pass
    if isinstance(data, np.ndarray):
        data = data.tolist()
    if isinstance(data, list):
        data = [float(x) if not isinstance(x, (list, tuple)) else float(x[0]) for x in data]
    return data


def parse_cache_filename(fname):
    """Parse a cached filename into (attack_key, aggregation_key, priority)."""
    if not fname or fname.endswith(".txt") or fname.endswith(".md"):
        return None, None, None
    for attack_key in sorted(ATTACK_FILE_PREFIX.keys(), key=lambda k: -len(ATTACK_FILE_PREFIX[k])):
        prefix = ATTACK_FILE_PREFIX[attack_key]
        if fname.startswith(prefix + "_"):
            agg_key = fname[len(prefix) + 1 :]
            priority = 1
            if agg_key.endswith("_twoStageLR"):
                agg_key = agg_key[: -len("_twoStageLR")]
                priority = 2
            return attack_key, agg_key, priority
    return None, None, None


def collect_from_cache():
    """Return data[(row_label, partition)][method_display] = (iters, accs)."""
    data = {}
    priorities = {}
    for graph_name, row_label in [(GRAPH_NAME_B0, ROW_B0_LABEL), (GRAPH_NAME, None)]:
        path_list_base = [TASK_GRAPH, graph_name]
        for part_key, part_dir in PARTITIONS:
            path_list = path_list_base + [part_dir]
            full_dir = os.path.join(RECORD_DIR, *path_list)
            if not os.path.isdir(full_dir):
                continue
            for fname in os.listdir(full_dir):
                if fname.endswith(".txt") or fname.endswith(".md"):
                    continue
                attack_key, agg_key, priority = parse_cache_filename(fname)
                if attack_key is None:
                    continue
                if agg_key.startswith("HSM"):
                    continue
                if "ablation" in agg_key:
                    continue
                key = (row_label if row_label is not None else attack_key, part_key)
                try:
                    if not isfile_in_cache(fname, path_list):
                        continue
                    record = load_file_in_cache(fname, path_list)
                except Exception:
                    continue
                if not isinstance(record, dict) or "acc_path" not in record:
                    continue
                acc_path = convert_to_list(record.get("acc_path"))
                if not acc_path:
                    continue
                interval = record.get("display_interval", 100)
                iters = [k * interval for k in range(len(acc_path))]
                method_display = AGG_DISPLAY.get(agg_key, agg_key)
                if method_display in EXCLUDED_METHODS:
                    continue
                data_key = (key, method_display)
                if priority < priorities.get(data_key, -1):
                    continue
                priorities[data_key] = priority
                data.setdefault(key, {})[method_display] = (iters, acc_path)
    return data


def _ordered_methods(curves):
    order = [m for m in PREFERRED_METHOD_ORDER if m in curves]
    order += sorted(k for k in curves if k not in order)
    return order


def draw_all_curves(data, save_dir):
    """Draw the full 7x2 figure used for inspection."""
    n_rows = len(ATTACK_ORDER)
    fig, axes = plt.subplots(n_rows, 2, figsize=(10, 4 * n_rows), sharex=True)
    part_titles = {"iid": "IID", "noniid": "Non-IID"}
    for i, row_label in enumerate(ATTACK_ORDER):
        for j, (part_key, _) in enumerate(PARTITIONS):
            ax = axes[i, j]
            curves = data.get((row_label, part_key), {})
            for idx, method in enumerate(_ordered_methods(curves)):
                iters, accs = curves[method]
                color = METHOD_COLORS.get(method, COLORS[idx % len(COLORS)])
                ax.plot(iters, accs, label=method, color=color, linewidth=2, alpha=0.9)
            ax.set_ylabel("Accuracy", fontsize=11)
            ax.set_ylim(0.0, 1.0)
            ax.set_xlim(0, 20000)
            ax.grid(True, alpha=0.3)
            ax.set_facecolor("#fafafa")
            ax.set_title(f"{row_label} | {part_titles.get(part_key, part_key)}", fontsize=11)
            if i == n_rows - 1:
                ax.set_xlabel("Iterations", fontsize=11)
            if curves:
                ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    out_path = os.path.join(save_dir, f"{FIG_STEM_PREFIX}_all_curves.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


def main():
    save_dir = os.path.join(__FILE_DIR__, "pic")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Loading b=0 and b={BYZANTINE_SIZE} data from cache.")
    data = collect_from_cache()
    if not data:
        print(f"No data found under Centralized_n=10_b=0 and Centralized_n=10_b={BYZANTINE_SIZE}.")
        return
    for (row_label, part), curves in sorted(data.items()):
        print(f"  {row_label} | {part}: {list(curves.keys())}")
    draw_all_curves(data, save_dir)
    print("Done.")


if __name__ == "__main__":
    main()
