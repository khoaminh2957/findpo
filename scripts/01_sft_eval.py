"""Phase 1 — SFT eval (delegates to findpo.evaluate.run_eval)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from findpo.evaluate import run_eval  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--eval-dataset", default="fpb")
    p.add_argument("--splits-dir", default="data/splits")
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()
    m = run_eval(Path(args.run_dir), args.eval_dataset, Path(args.splits_dir),
                 batch_size=args.batch_size)
    print(json.dumps({k: v for k, v in m.items() if k != "per_class"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
