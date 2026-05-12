"""Phase 0 — download datasets, compute SHA256 manifest (R5).

Run on REMOTE_GPU after 00_setup_check.py passes.
Writes data/splits/<dataset>_<split>.arrow and data/manifest_phase0.json.

The manifest is then committed; it's the data fingerprint future runs verify.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from findpo.data import (  # noqa: E402
    DATASET_FIELD_MAP,
    SplitFingerprint,
    deterministic_split,
    label_dist,
    load_one,
    normalize_dataset,
    sha256_of_dataset,
    write_manifest,
    assert_no_overlap,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Any config that lists datasets:.")
    parser.add_argument("--out", default="data/splits")
    parser.add_argument("--manifest", default="data/manifest_phase0.json")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_seed = int(cfg["split"]["split_seed"])
    train_ratio = float(cfg["split"]["train_ratio"])

    fingerprints: list[SplitFingerprint] = []

    for source_tag, ds_cfg in cfg["datasets"].items():
        repo = ds_cfg["repo"]
        if repo.startswith("TODO"):
            print(f"[SKIP] {source_tag}: repo is TODO ({repo})")
            continue
        revision = ds_cfg.get("revision")
        config_name = ds_cfg.get("config")

        print(f"\n--- {source_tag} ({repo}) ---")
        raw = load_one(repo, revision, config=config_name)
        # Pick a sensible split — most HF classification datasets ship "train"
        if hasattr(raw, "keys"):
            split_key = "train" if "train" in raw else next(iter(raw.keys()))
            raw_split = raw[split_key]
        else:
            raw_split = raw

        norm = normalize_dataset(raw_split, repo=repo, source_tag=source_tag)
        train, test, train_idx, test_idx = deterministic_split(norm, train_ratio, split_seed)
        assert_no_overlap(train, test)

        train_path = out_dir / f"{source_tag}_train"
        test_path = out_dir / f"{source_tag}_test"
        train.save_to_disk(str(train_path))
        test.save_to_disk(str(test_path))

        for split_name, split_ds in (("train", train), ("test", test)):
            fp = SplitFingerprint(
                name=f"{source_tag}_{split_name}",
                repo=repo,
                revision=revision,
                config=config_name,
                num_rows=len(split_ds),
                sha256=sha256_of_dataset(split_ds),
                label_distribution=label_dist(split_ds),
            )
            fingerprints.append(fp)
            print(f"  {fp.name}: n={fp.num_rows}, dist={fp.label_distribution}, sha256={fp.sha256[:16]}…")

        # Save split index arrays so the split can be reconstructed without re-running
        (out_dir / f"{source_tag}_train_idx.json").write_text(json.dumps(train_idx.tolist()))
        (out_dir / f"{source_tag}_test_idx.json").write_text(json.dumps(test_idx.tolist()))

    write_manifest(Path(args.manifest), fingerprints, extras={"split_seed": split_seed,
                                                                "train_ratio": train_ratio})
    print(f"\nWrote manifest → {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
