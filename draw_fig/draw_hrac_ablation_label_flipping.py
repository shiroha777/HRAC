"""Plot HRAC ablation results for CIFAR-10 (b=3).

Run the experiments first:
  python run_hrac_ablation_label_flipping_cifar10.py

Then draw:
  python draw_fig/draw_hrac_ablation_label_flipping.py --attack alie --partitions noniid
"""
import argparse
import os
import pickle
import re
import struct
import sys
import types

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
RECORD_DIR = os.path.join(PROJECT_ROOT, "record")
TASK_GRAPH = "NeuralNetwork_cifar10"
GRAPH_NAME = "Centralized_n=10_b=3"
PARTITIONS_ALL = [
    ("iid", "iidPartition", "IID"),
    ("noniid", "LabelSeperation", "Non-IID"),
]

VARIANTS = [
    ("HRAC_ablation_full", "Full HRAC"),
    ("HRAC_ablation_no_global_cap", "w/o global cap"),
    ("HRAC_ablation_no_residual_clip", "w/o residual clipping"),
    ("HRAC_ablation_no_nu_weighting", "w/o nu weighting"),
]


def attack_record_name(attack):
    return {
        "alie": "ALIE",
        "ipm": "IPM",
        "hismsa": "HisMSA",
        "msa": "MSA_MSA",
        "label_flipping": "label_flipping",
        "bit_flipping": "bit_flipping",
        "bf": "bit_flipping",
    }.get(attack, attack)


def attack_display_name(attack):
    return {
        "alie": "ALIE",
        "ipm": "IPM",
        "hismsa": "HisMSA",
        "msa": "MSA",
        "label_flipping": "label flipping",
        "bit_flipping": "bit flipping",
        "bf": "bit flipping",
    }.get(attack, attack.replace("_", " "))


def _to_list(data):
    try:
        import torch
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
    except Exception:
        pass
    if hasattr(data, "tolist"):
        data = data.tolist()
    return [float(x[0] if isinstance(x, (list, tuple)) else x) for x in data]


def load_record(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise

    original_modules = {name: sys.modules.get(name) for name in ("torch", "torch._utils", "torch.storage")}

    torch_mod = types.ModuleType("torch")
    utils_mod = types.ModuleType("torch._utils")
    storage_mod = types.ModuleType("torch.storage")

    def _load_from_bytes(data):
        return data

    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        # Stored accuracy/loss entries are scalar tensors. The final eight bytes
        # of the serialized storage contain the little-endian double value.
        return struct.unpack("<d", storage[-8:])[0]

    storage_mod._load_from_bytes = _load_from_bytes
    utils_mod._rebuild_tensor_v2 = _rebuild_tensor_v2
    torch_mod._utils = utils_mod
    torch_mod.storage = storage_mod

    sys.modules["torch"] = torch_mod
    sys.modules["torch._utils"] = utils_mod
    sys.modules["torch.storage"] = storage_mod
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    finally:
        for name, module in original_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def log_attack_name(attack):
    return {
        "alie": "ALIE",
        "ipm": "IPM",
        "hismsa": "HisMSA",
        "msa": "MSA",
        "label_flipping": "label_flipping",
        "bit_flipping": "bit_flipping",
        "bf": "bit_flipping",
    }.get(attack, attack)


def log_path_for_variant(base, attack, variant_key):
    attack_name = log_attack_name(attack)
    if variant_key == "HRAC_ablation_full":
        return os.path.join(base, f"hrac-log-{attack_name}.txt")
    prefix = "HRAC_ablation_"
    if variant_key.startswith(prefix):
        variant = variant_key[len(prefix):]
        return os.path.join(base, f"hrac-ablation-{variant}-log-{attack_name}.txt")
    return None


def load_curve_from_log(path):
    if not path or not os.path.isfile(path):
        return None
    accs = []
    pattern = re.compile(r"Accuracy:\s*([0-9.]+)")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                accs.append(float(match.group(1)))
    if not accs:
        return None
    iters = [i * 100 for i in range(len(accs))]
    return iters, accs


def collect_curves(attack, partitions):
    curves = {}
    attack_prefix = "CMomentum_" + attack_record_name(attack) + "_"
    for part_key, part_dir, _ in partitions:
        base = os.path.join(RECORD_DIR, TASK_GRAPH, GRAPH_NAME, part_dir)
        part_curves = {}
        for variant_key, label in VARIANTS:
            log_curve = load_curve_from_log(log_path_for_variant(base, attack, variant_key))
            if log_curve is not None:
                part_curves[label] = log_curve
                continue

            fname = attack_prefix + variant_key
            path = os.path.join(base, fname)
            if not os.path.isfile(path):
                path = os.path.join(base, fname + "_twoStageLR")
            if not os.path.isfile(path) and variant_key == "HRAC_ablation_full":
                path = os.path.join(base, attack_prefix + "HRAC")
            if not os.path.isfile(path) and variant_key == "HRAC_ablation_full":
                path = os.path.join(base, attack_prefix + "HRAC_twoStageLR")
            if not os.path.isfile(path):
                continue
            record = load_record(path)
            acc_path = _to_list(record["acc_path"])
            interval = int(record.get("display_interval", 100))
            iters = [i * interval for i in range(len(acc_path))]
            part_curves[label] = (iters, acc_path)
        curves[part_key] = part_curves
    return curves


def draw(curves, attack, partitions):
    fig_width = 4.8 if len(partitions) == 1 else 9.0
    fig, axes = plt.subplots(1, len(partitions), figsize=(fig_width, 3.4), sharey=True)
    if len(partitions) == 1:
        axes = [axes]
    colors = plt.cm.tab10(range(len(VARIANTS)))
    attack_title = attack_display_name(attack)
    for ax, (part_key, _, title) in zip(axes, partitions):
        part_curves = curves.get(part_key, {})
        for idx, (_, label) in enumerate(VARIANTS):
            if label not in part_curves:
                continue
            iters, accs = part_curves[label]
            ax.plot(iters, accs, label=label, linewidth=1.8, color=colors[idx])
        ax.set_title(f"{attack_title} | {title}", fontsize=10)
        ax.set_xlabel("Iterations")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Accuracy")
    fig.tight_layout()

    out_dir = os.path.join(THIS_DIR, "pic")
    os.makedirs(out_dir, exist_ok=True)
    partition_tag = "_".join(part_key for part_key, _, _ in partitions)
    png_path = os.path.join(out_dir, f"CMomentum_cifar10_b3_{attack_record_name(attack)}_{partition_tag}_hrac_ablation.png")
    pdf_path = png_path.replace(".png", ".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    paper_img_dir = os.path.join(PROJECT_ROOT, "ByrdLab", "paper", "imgs")
    if os.path.isdir(paper_img_dir):
        import shutil
        shutil.copy2(png_path, os.path.join(paper_img_dir, os.path.basename(png_path)))
        shutil.copy2(pdf_path, os.path.join(paper_img_dir, os.path.basename(pdf_path)))

    print(f"Saved: {png_path}")


def main():
    parser = argparse.ArgumentParser(description="Draw CIFAR-10 HRAC ablation curves.")
    parser.add_argument("--attack", default="label_flipping",
                        help="Attack name used in the experiment, e.g. label_flipping or alie.")
    parser.add_argument("--partitions", nargs="+", default=["iid", "noniid"],
                        choices=[x[0] for x in PARTITIONS_ALL],
                        help="Partitions to draw.")
    args = parser.parse_args()

    partitions = [item for item in PARTITIONS_ALL if item[0] in args.partitions]
    curves = collect_curves(args.attack, partitions)
    if not any(curves.values()):
        print("No HRAC ablation records found. Run run_hrac_ablation_label_flipping_cifar10.py first.")
        return 1
    draw(curves, args.attack, partitions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
