"""Aggregate per-seed eval metrics into a single mean ± std table.

R9: never report a single-seed number. After all 3 seeds of an experiment
finish, run this to produce a CSV + a Markdown table that the Phase-1 / Phase-2
report drops in verbatim.

Usage:
    python scripts/aggregate_seeds.py --exp-id sft_baseline --eval-dataset fpb
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--exp-id", required=True)
    p.add_argument("--eval-dataset", default="fpb")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--out-md", default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7])
    args = p.parse_args()

    rows = []
    for seed in args.seeds:
        path = Path(args.results_dir) / f"exp_{args.exp_id}_seed{seed}" / f"eval_{args.eval_dataset}.json"
        if not path.exists():
            print(f"[WARN] missing {path}")
            continue
        m = json.loads(path.read_text())
        rows.append({
            "seed": seed,
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "n": m["n"],
            "n_unparseable": m["n_unparseable"],
        })

    if not rows:
        print("No runs found.")
        return 1

    print(f"\n## {args.exp_id} on {args.eval_dataset} test (n={rows[0]['n']})\n")
    print("| Seed | Accuracy | Macro F1 | Unparseable |")
    print("|------|----------|----------|-------------|")
    for r in rows:
        print(f"| {r['seed']} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} | {r['n_unparseable']} |")

    acc = [r["accuracy"] for r in rows]
    f1 = [r["macro_f1"] for r in rows]
    print(f"| **mean ± std** | "
          f"**{statistics.mean(acc):.4f} ± {statistics.stdev(acc) if len(acc) > 1 else 0:.4f}** | "
          f"**{statistics.mean(f1):.4f} ± {statistics.stdev(f1) if len(f1) > 1 else 0:.4f}** | — |")

    if args.out_md:
        Path(args.out_md).write_text(
            "\n".join([
                f"## {args.exp_id} on {args.eval_dataset} test (n={rows[0]['n']})",
                "",
                "| Seed | Accuracy | Macro F1 | Unparseable |",
                "|------|----------|----------|-------------|",
                *(f"| {r['seed']} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} | {r['n_unparseable']} |" for r in rows),
                f"| **mean ± std** | "
                f"**{statistics.mean(acc):.4f} ± {statistics.stdev(acc) if len(acc) > 1 else 0:.4f}** | "
                f"**{statistics.mean(f1):.4f} ± {statistics.stdev(f1) if len(f1) > 1 else 0:.4f}** | — |",
                "",
            ])
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
