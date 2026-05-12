"""Dataset loading, splitting, SHA256 manifest (R5).

Every dataset is pinned by HuggingFace revision (commit hash). After loading,
splits are saved to disk and SHA256-hashed. The manifest is checked into the
run directory so a third party can verify they loaded the same bytes.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset

from .labels import canonicalize_label


# ----------------------------- field schemas --------------------------------
# Each dataset has its own text/label column names. Map to the canonical
# {"text", "label", "source"} schema we use downstream.

DATASET_FIELD_MAP: dict[str, dict[str, str]] = {
    "takala/financial_phrasebank": {"text": "sentence", "label": "label"},
    "zeroshot/twitter-financial-news-sentiment": {"text": "text", "label": "label"},
    # gpt_news repo TBD — fill once verified against paper.
}


def _normalize_example(ex: dict, fields: dict[str, str], source: str) -> dict:
    return {
        "text": ex[fields["text"]],
        "label": canonicalize_label(ex[fields["label"]]),
        "source": source,
    }


def load_one(repo: str, revision: str | None, config: str | None = None,
             split: str | None = None) -> Dataset:
    """Load a single HF dataset with explicit revision pin."""
    if revision is None:
        # Allow null during prep, but warn when actually fetching.
        print(f"[WARN] revision=None for {repo} — pin after first download (R5).")
    kwargs: dict[str, Any] = {}
    if revision is not None:
        kwargs["revision"] = revision
    if config is not None:
        kwargs["name"] = config
    if split is not None:
        kwargs["split"] = split
    ds = load_dataset(repo, **kwargs)
    if isinstance(ds, DatasetDict):
        return ds
    return ds


def normalize_dataset(ds: Dataset, repo: str, source_tag: str) -> Dataset:
    fields = DATASET_FIELD_MAP.get(repo)
    if fields is None:
        raise KeyError(
            f"No field mapping for {repo}; add it to DATASET_FIELD_MAP in findpo/data.py."
        )
    return ds.map(lambda ex: _normalize_example(ex, fields, source_tag),
                  remove_columns=ds.column_names)


def deterministic_split(ds: Dataset, train_ratio: float, seed: int) -> tuple[Dataset, Dataset, np.ndarray, np.ndarray]:
    """80/20 split reproducible from seed alone. Returns (train, test, train_idx, test_idx)."""
    n = len(ds)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    cut = int(round(train_ratio * n))
    train_idx, test_idx = perm[:cut], perm[cut:]
    return ds.select(train_idx.tolist()), ds.select(test_idx.tolist()), train_idx, test_idx


# ----------------------------- manifest -------------------------------------

@dataclass
class SplitFingerprint:
    name: str
    repo: str
    revision: str | None
    config: str | None
    num_rows: int
    sha256: str
    label_distribution: dict[str, int]


def sha256_of_dataset(ds: Dataset) -> str:
    """Hash the normalised (text, label, source) bytes of every example.
    Order-sensitive — call after split is materialized."""
    h = hashlib.sha256()
    for ex in ds:
        h.update(ex["text"].encode("utf-8"))
        h.update(b"\x00")
        h.update(ex["label"].encode("utf-8"))
        h.update(b"\x00")
        h.update(ex["source"].encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def label_dist(ds: Dataset) -> dict[str, int]:
    from collections import Counter
    return dict(Counter(ds["label"]))


def write_manifest(manifest_path: Path, splits: list[SplitFingerprint],
                   extras: dict | None = None) -> None:
    payload = {"splits": [asdict(s) for s in splits]}
    if extras:
        payload.update(extras)
    manifest_path.write_text(json.dumps(payload, indent=2))


def assert_no_overlap(train: Dataset, test: Dataset) -> None:
    """R8.d — hash every text + label, assert empty intersection."""
    def keyset(ds: Dataset) -> set[str]:
        return {
            hashlib.sha256(f"{ex['text']}\x00{ex['label']}".encode()).hexdigest()
            for ex in ds
        }
    overlap = keyset(train) & keyset(test)
    if overlap:
        raise AssertionError(
            f"Train/test overlap: {len(overlap)} examples share (text,label). "
            "Split is broken — refuse to train."
        )
