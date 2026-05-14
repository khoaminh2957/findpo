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
            # weighted_f1 is the paper's headline metric (FinDPO Table 2);
            # fall back to None for old eval json that didn't have it.
            "weighted_f1": m.get("weighted_f1"),
            "n": m["n"],
            "n_unparseable": m["n_unparseable"],
        })

    if not rows:
        print("No runs found.")
        return 1

    print(f"\n## {args.exp_id} on {args.eval_dataset} test (n={rows[0]['n']})\n")
    print("| Seed | Accuracy | Macro F1 | Weighted F1 | Unparseable |")
    print("|------|----------|----------|-------------|-------------|")
    for r in rows:
        wf1 = f"{r['weighted_f1']:.4f}" if r['weighted_f1'] is not None else "-"
        print(f"| {r['seed']} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} | {wf1} | {r['n_unparseable']} |")

    def stat_line(vals):
        if not vals: return "-"
        m = statistics.mean(vals)
        s = statistics.stdev(vals) if len(vals) > 1 else 0
        return f"**{m:.4f} ± {s:.4f}**"

    acc = [r["accuracy"] for r in rows]
    f1 = [r["macro_f1"] for r in rows]
    wf1 = [r["weighted_f1"] for r in rows if r["weighted_f1"] is not None]
    print(f"| **mean ± std** | {stat_line(acc)} | {stat_line(f1)} | {stat_line(wf1)} | — |")

    if args.out_md:
        lines = [
            f"## {args.exp_id} on {args.eval_dataset} test (n={rows[0]['n']})",
            "",
            "| Seed | Accuracy | Macro F1 | Weighted F1 | Unparseable |",
            "|------|----------|----------|-------------|-------------|",
        ]
        for r in rows:
            wf1_str = f"{r['weighted_f1']:.4f}" if r['weighted_f1'] is not None else "-"
            lines.append(f"| {r['seed']} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} | {wf1_str} | {r['n_unparseable']} |")
        lines.append(f"| **mean ± std** | {stat_line(acc)} | {stat_line(f1)} | {stat_line(wf1)} | — |")
        lines.append("")
        Path(args.out_md).write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
