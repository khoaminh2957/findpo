"""Label canonicalisation + prompt formatting — single source of truth.

Every dataset uses different integer/string label encodings for sentiment.
To avoid the tokenizer-alignment / off-by-one bugs called out in R8.e and
R8.f, ALL training and eval code goes through this module — never inline a
label string.

Per-dataset int→canonical maps live in `INT_LABEL_MAPS` because the integer
encodings differ across datasets (e.g. TFNS uses {0:Bearish, 1:Bullish,
2:Neutral} — NOT the same order as FPB).
"""
from __future__ import annotations

from typing import Iterable

LABELS: tuple[str, ...] = ("negative", "neutral", "positive")
LABEL2ID: dict[str, int] = {lbl: i for i, lbl in enumerate(LABELS)}
ID2LABEL: dict[int, str] = {i: lbl for i, lbl in enumerate(LABELS)}

# Per-dataset integer-label encodings — verified against dataset cards
# (FinDPO repro audit 2026-05-13, FinDPO paper section 4.1.1).
INT_LABEL_MAPS: dict[str, dict[int, str]] = {
    # FPB: ClassLabel(names=["negative", "neutral", "positive"]) — natural order.
    "takala/financial_phrasebank": {0: "negative", 1: "neutral", 2: "positive"},
    # TFNS: dataset card lists LABEL_0=Bearish, LABEL_1=Bullish, LABEL_2=Neutral.
    "zeroshot/twitter-financial-news-sentiment": {0: "negative", 1: "positive", 2: "neutral"},
    # NWGI uses STRING labels (handled by _STR_LABEL_MAP), no int map needed.
}

# String labels — includes NWGI's 7-class scheme merged to 3 classes per
# FinDPO paper §4.1.1: "strongly and mildly negative classes were combined
# into a single negative class, and similarly... positive". Empirically NWGI
# also has 'moderately' tier (paper says 5 labels but dataset has 7);
# we treat any */negative as negative, */positive as positive.
_STR_LABEL_MAP: dict[str, str] = {
    "negative": "negative", "neutral": "neutral", "positive": "positive",
    "bearish": "negative", "bullish": "positive",
    # NWGI 7-class → 3-class
    "strong negative": "negative",
    "moderately negative": "negative",
    "mildly negative": "negative",
    "mildly positive": "positive",
    "moderately positive": "positive",
    "strong positive": "positive",
}


def canonicalize_label(raw, repo: str | None = None) -> str:
    """Map dataset-specific encoding → canonical {negative, neutral, positive}.

    `repo` is required when `raw` is an int (encodings differ per dataset).
    For strings the mapping is unambiguous so `repo` is optional.
    """
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in _STR_LABEL_MAP:
            return _STR_LABEL_MAP[s]
        raise ValueError(f"Unknown string label: {raw!r}")
    if isinstance(raw, (int, bool)):
        if repo is None:
            raise ValueError(
                f"Integer label {raw!r} given without repo. Cannot canonicalize — "
                "encodings differ per dataset (TFNS swaps positive/neutral)."
            )
        m = INT_LABEL_MAPS.get(repo)
        if m is None:
            raise KeyError(
                f"No integer→label map for {repo}. Add it to INT_LABEL_MAPS in findpo/labels.py."
            )
        return m[int(raw)]
    raise TypeError(f"Unsupported label type: {type(raw)}")


def format_prompt_messages(system: str, user_text: str) -> list[dict]:
    """Return a chat-template messages list. Apply tokenizer.apply_chat_template
    downstream — never hand-craft the template (R8.e)."""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_text},
    ]


def format_assistant_message(label: str) -> dict:
    if label not in LABEL2ID:
        raise ValueError(f"label must be one of {LABELS}, got {label!r}")
    return {"role": "assistant", "content": label}


def class_distribution(labels: Iterable[str]) -> dict[str, int]:
    out = {lbl: 0 for lbl in LABELS}
    for lbl in labels:
        # All values in `labels` should already be canonical strings; if not,
        # this will raise — surface the bug rather than silently miscount.
        out[canonicalize_label(lbl)] += 1
    return out
