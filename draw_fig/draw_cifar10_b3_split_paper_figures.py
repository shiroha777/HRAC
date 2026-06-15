"""Generate split CIFAR-10 b=3 figures for the dissertation paper.

This script reuses the cache-loading logic from
draw_cifar10_b3_all_curves_from_logs.py, but saves smaller grouped figures that
are easier to place and discuss in the paper.
"""
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from draw_cifar10_b3_all_curves_from_logs import (
    ATTACK_ORDER,
    BYZANTINE_SIZE,
    COLORS,
    FIG_STEM_PREFIX,
    METHOD_COLORS,
    PARTITIONS,
    PREFERRED_METHOD_ORDER,
    ROW_B0_LABEL,
    __FILE_DIR__,
    collect_from_cache,
)


PART_TITLES = {"iid": "IID", "noniid": "Non-IID"}
ROW_TITLES = {
    ROW_B0_LABEL: "Baseline (b=0)",
    "label_flipping": "Label flipping",
    "MSA": "MSA",
    "HisMSA": "HisMSA",
    "ALIE": "ALIE",
    "IPM": "IPM",
    "poisonedfl": "PoisonedFL",
}

FIGURE_GROUPS = [
    (
        f"{FIG_STEM_PREFIX}_baseline_label_flipping",
        [ROW_B0_LABEL, "label_flipping"],
        "Benign training and label-flipping attack",
    ),
    (
        f"{FIG_STEM_PREFIX}_msa_hismsa",
        ["MSA", "HisMSA"],
        "Model-shuffling and history-aware attacks",
    ),
    (
        f"{FIG_STEM_PREFIX}_alie_ipm",
        ["ALIE", "IPM"],
        "ALIE and IPM attacks",
    ),
    (
        f"{FIG_STEM_PREFIX}_poisonedfl",
        ["poisonedfl"],
        "PoisonedFL attack",
    ),
]


def _ordered_methods(curves):
    order = [m for m in PREFERRED_METHOD_ORDER if m in curves]
    order += sorted(k for k in curves if k not in order)
    return order


def draw_group(data, save_dir, stem, row_labels, title):
    n_rows = len(row_labels)
    fig_height = 2.45 * n_rows
    fig, axes = plt.subplots(n_rows, 2, figsize=(7.2, fig_height), sharex=True)
    if n_rows == 1:
        axes = [axes]

    for i, row_label in enumerate(row_labels):
        for j, (part_key, _) in enumerate(PARTITIONS):
            ax = axes[i][j]
            curves = data.get((row_label, part_key), {})
            for idx, method in enumerate(_ordered_methods(curves)):
                iters, accs = curves[method]
                color = METHOD_COLORS.get(method, COLORS[idx % len(COLORS)])
                ax.plot(iters, accs, label=method, color=color, linewidth=1.5, alpha=0.92)

            row_title = ROW_TITLES.get(row_label, row_label)
            ax.set_title(f"{row_title} | {PART_TITLES.get(part_key, part_key)}", fontsize=8.5)
            ax.set_ylabel("Accuracy", fontsize=8)
            ax.set_ylim(0.0, 1.0)
            ax.set_xlim(0, 20000)
            ax.grid(True, alpha=0.28)
            ax.set_facecolor("#fafafa")
            ax.tick_params(axis="both", labelsize=7.5)
            if i == n_rows - 1:
                ax.set_xlabel("Iterations", fontsize=8)
            if curves:
                ax.legend(loc="best", fontsize=5.8, framealpha=0.9)

    fig.tight_layout()

    png_path = os.path.join(save_dir, f"{stem}.png")
    pdf_path = os.path.join(save_dir, f"{stem}.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")
    return png_path, pdf_path


def copy_to_paper(paths):
    project_root = os.path.dirname(__FILE_DIR__)
    paper_img_dir = os.path.join(project_root, "ByrdLab", "paper", "imgs")
    os.makedirs(paper_img_dir, exist_ok=True)
    for path in paths:
        dst = os.path.join(paper_img_dir, os.path.basename(path))
        shutil.copy2(path, dst)
        print(f"Copied to paper: {dst}")


def main():
    save_dir = os.path.join(__FILE_DIR__, "pic")
    os.makedirs(save_dir, exist_ok=True)

    print(f"Generating split figures for b={BYZANTINE_SIZE}.")
    data = collect_from_cache()
    missing_rows = [row for row in ATTACK_ORDER if not any((row, p[0]) in data for p in PARTITIONS)]
    if missing_rows:
        print(f"Warning: no curves found for rows: {missing_rows}")

    generated = []
    for stem, row_labels, title in FIGURE_GROUPS:
        generated.extend(draw_group(data, save_dir, stem, row_labels, title))
    copy_to_paper(generated)


if __name__ == "__main__":
    main()
