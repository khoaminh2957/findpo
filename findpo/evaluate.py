"""Shared classification eval — used by both SFT and DPO eval scripts.

The trained artifact is always a base model + LoRA adapter at <run_dir>/model.
This module is the single source of truth so SFT vs DPO numbers are computed
identically (Pineau et al. 2021 — "report the same metric the same way").
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from .labels import LABELS, format_prompt_messages
from .tokenizer_setup import render_chat, setup_tokenizer


def _parse_label(raw: str) -> str | None:
    """Parse model output to a canonical label.

    Two passes so the FIRST word the model emitted wins over any other label
    that may also appear later in a verbose response. Without the two-pass
    structure, an output like "positive or negative" would match "negative"
    first (alphabetical order of LABELS) via the `in split` branch — wrong;
    the model's actual answer was "positive".
    """
    s = raw.strip().lower()
    # Pass 1: prefer the label the model started with.
    for lbl in LABELS:
        if s.startswith(lbl):
            return lbl
    # Pass 2: fall back to any label appearing as a whole word.
    for lbl in LABELS:
        if lbl in s.split():
            return lbl
    return None


def run_eval(run_dir: Path, eval_dataset: str, splits_dir: Path,
             batch_size: int = 8, max_new_tokens: int = 4) -> dict:
    import torch
    from datasets import load_from_disk
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from sklearn.metrics import (accuracy_score, classification_report,
                                  confusion_matrix, f1_score)

    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    sys_prompt = cfg["prompt"]["system"]
    test = load_from_disk(str(splits_dir / f"{eval_dataset}_test"))

    base_name = cfg["model"]["name"]
    tok = AutoTokenizer.from_pretrained(base_name)
    setup_tokenizer(tok, padding_side="left")   # generation → left pad

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    requested_attn = cfg["model"].get("attn_implementation")
    try:
        base = AutoModelForCausalLM.from_pretrained(
            base_name, quantization_config=bnb, device_map="auto",
            attn_implementation=requested_attn,
        )
    except (ImportError, ValueError) as e:
        if requested_attn != "sdpa":
            print(f"[WARN] attn_implementation={requested_attn!r} failed ({e}); "
                  "falling back to 'sdpa'.")
            base = AutoModelForCausalLM.from_pretrained(
                base_name, quantization_config=bnb, device_map="auto",
                attn_implementation="sdpa",
            )
        else:
            raise
    model = PeftModel.from_pretrained(base, str(run_dir / "model"))
    model.eval()

    preds = []
    n_unparseable = 0
    for i in range(0, len(test), batch_size):
        batch = test.select(range(i, min(i + batch_size, len(test))))
        prompts = [
            render_chat(tok, format_prompt_messages(sys_prompt, ex["text"]),
                        add_generation_prompt=True)
            for ex in batch
        ]
        max_seq = cfg["train"].get("max_seq_length") or cfg["train"].get("max_length", 512)
        # add_special_tokens=False — prompts already contain the literal
        # <|begin_of_text|> from the chat template. Auto-BOS would duplicate it.
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_seq, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        decoded = tok.batch_decode(gen, skip_special_tokens=True)
        for ex, raw in zip(batch, decoded):
            pred = _parse_label(raw)
            if pred is None:
                n_unparseable += 1
                pred = "neutral"
            preds.append({"text": ex["text"], "true": ex["label"], "pred": pred, "raw": raw})

    y_true = [d["true"] for d in preds]
    y_pred = [d["pred"] for d in preds]
    metrics = {
        "eval_dataset": eval_dataset,
        "n": len(preds),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=list(LABELS), average="macro"),
        "n_unparseable": n_unparseable,
        "per_class": classification_report(y_true, y_pred, labels=list(LABELS),
                                            output_dict=True, zero_division=0),
        "confusion_matrix": {
            "labels": list(LABELS),
            "matrix": confusion_matrix(y_true, y_pred, labels=list(LABELS)).tolist(),
        },
    }
    (run_dir / f"eval_{eval_dataset}.json").write_text(json.dumps(metrics, indent=2))
    with (run_dir / f"predictions_{eval_dataset}.jsonl").open("w") as f:
        for d in preds:
            f.write(json.dumps(d) + "\n")
    return metrics
