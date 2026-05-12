"""Label canonicalisation + prompt formatting — single source of truth.

Every dataset uses different integer/string labels for sentiment. To avoid
the tokenizer-alignment bug called out in R8.e, ALL training and eval code
goes through this module — never inline a label string.
"""
from __future__ import annotations

from typing import Iterable

LABELS: tuple[str, ...] = ("negative", "neutral", "positive")
LABEL2ID: dict[str, int] = {lbl: i for i, lbl in enumerate(LABELS)}
ID2LABEL: dict[int, str] = {i: lbl for i, lbl in enumerate(LABELS)}


def canonicalize_label(raw) -> str:
    """Map dataset-specific label encodings → canonical string in LABELS."""
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in LABEL2ID:
            return s
        # TFNS uses LABEL_0/1/2 in some HF mirrors
        if s in ("label_0", "0", "bearish"):
            return "negative"
        if s in ("label_1", "1", "neutral"):
            return "neutral"
        if s in ("label_2", "2", "bullish"):
            return "positive"
        raise ValueError(f"Unknown string label: {raw!r}")
    if isinstance(raw, (int, bool)):
        # FPB: 0=neg, 1=neu, 2=pos. Confirm per-dataset before relying on this.
        return ID2LABEL[int(raw)]
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
        out[canonicalize_label(lbl)] += 1
    return out
