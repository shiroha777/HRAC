"""
Plot training curves from byzantine-robust-optimizer run (stats file).
Usage (in LPA repo, from project root or draw_fig):
  python draw_fig/draw_paper_repo_curves.py
  python draw_fig/draw_paper_repo_curves.py --stats "G:/FYP/LPA/byzantine-robust-optimizer/outputs/cifar10-all/f5_BF_cp_m0.99_seed0_noniid/stats"
If matplotlib is not installed, writes epoch,loss,acc to a CSV next to --out.
"""
import argparse
import ast
import os

def load_stats(path):
    epochs, eval_loss, eval_top1 = [], [], []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = ast.literal_eval(line)
            if d.get("_meta", {}).get("type") == "validation":
                epochs.append(d["E"])
                eval_loss.append(d["Loss"])
                eval_top1.append(d["top1"])
    return epochs, eval_loss, eval_top1


def main():
    parser = argparse.ArgumentParser()
    default_stats = os.path.join(
        os.path.dirname(__file__), "..", "byzantine-robust-optimizer",
        "outputs", "cifar10-all", "f5_BF_cp_m0.99_seed0_noniid", "stats"
    )
    parser.add_argument("--stats", type=str, default=default_stats, help="Path to stats file")
    parser.add_argument("--out", type=str, default=None, help="Output .png path")
    parser.add_argument("--title", type=str, default=None)
    args = parser.parse_args()

    path = os.path.normpath(args.stats)
    if not os.path.isfile(path):
        print("Stats file not found:", path)
        return
    epochs, eval_loss, eval_top1 = load_stats(path)
    if not epochs:
        print("No validation records in", path)
        return

    title = args.title or "BF+CC Non-IID (paper repo)"
    out_path = args.out
    if out_path is None:
        out_path = os.path.join(os.path.dirname(__file__), "pic", "paper_repo_BF_CC_noniid_curves.png")
    out_path = os.path.normpath(out_path)
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)

    # Always write CSV for backup / Excel
    csv_path = out_path.replace(".png", ".csv")
    with open(csv_path, "w") as f:
        f.write("epoch,test_loss,test_acc\n")
        for e, L, A in zip(epochs, eval_loss, eval_top1):
            f.write(f"{e},{L},{A}\n")
    print("Saved CSV:", csv_path)

    try:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.plot(epochs, eval_top1, "b-", linewidth=2)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Test Accuracy (%)")
        ax1.set_title(f"{title}\nTest Accuracy")
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 105)

        ax2.plot(epochs, eval_loss, "r-", linewidth=2)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Test Loss")
        ax2.set_title(f"{title}\nTest Loss")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved plot:", out_path)
    except ImportError:
        print("matplotlib not found. Install with: pip install matplotlib")
        print("You can open the CSV in Excel to plot:", csv_path)


if __name__ == "__main__":
    main()
