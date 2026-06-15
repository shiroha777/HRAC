"""Run HRAC ablations on CIFAR-10 (b=3).

This launcher runs the five ablation variants used in the dissertation:

1. full
2. no_global_cap
3. no_residual_clip
4. no_nu_weighting
5. global_cap_only

By default, each variant is evaluated on both IID and label-separation non-IID
partitions under label flipping. Use --attack and --partitions to switch the
environment, e.g. --attack alie --partitions noniid.
Results are saved under distinct cache names such as
CMomentum_ALIE_HRAC_ablation_no_global_cap, so normal HRAC runs are
not overwritten.
"""
import argparse
import os
import subprocess
import sys


VARIANTS = [
    "full",
    "no_global_cap",
    "no_residual_clip",
    "no_nu_weighting",
    "global_cap_only",
]

PARTITIONS = ["iid", "noniid"]


def main():
    parser = argparse.ArgumentParser(description="Run CIFAR-10 HRAC ablations.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use for each run.")
    parser.add_argument("--seed", type=int, default=100, help="Seed passed to main CMomentum.py.")
    parser.add_argument("--cpu", action="store_true", help="Pass --cpu to each run.")
    parser.add_argument("--attack", default="label_flipping",
                        help="Attack passed to main CMomentum.py, e.g. label_flipping, alie, ipm.")
    parser.add_argument("--partitions", nargs="+", default=PARTITIONS, choices=PARTITIONS,
                        help="Data partitions to run. Use: --partitions noniid for ALIE non-IID ablation.")
    parser.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS,
                        help="Ablation variants to run.")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    main_script = os.path.join(project_root, "main CMomentum.py")
    if not os.path.isfile(main_script):
        raise FileNotFoundError(main_script)

    total = len(args.variants) * len(args.partitions)
    current = 0
    for variant in args.variants:
        for partition in args.partitions:
            current += 1
            cmd = [
                args.python,
                main_script,
                "--attack", args.attack,
                "--aggregation", "hrac",
                "--data-partition", partition,
                "--seed", str(args.seed),
                "--hrac-ablation-experiment",
                "--hrac-ablation", variant,
            ]
            if args.cpu:
                cmd.append("--cpu")

            printable = " ".join(f'"{x}"' if " " in x else x for x in cmd)
            print(f"\n[{current}/{total}] {printable}")
            sys.stdout.flush()
            if args.dry_run:
                continue

            ret = subprocess.run(cmd, cwd=project_root)
            if ret.returncode != 0:
                print(f"Warning: command failed with code {ret.returncode}: {printable}")

    print(f"\nDone. Total ablation runs: {total}")


if __name__ == "__main__":
    main()
