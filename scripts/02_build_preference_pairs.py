"""Phase 2 — build preference pairs (offline).

Strategy A: rejected = uniform random from the two non-true classes.
Strategy B: rejected = top non-true class predicted by the SFT model
            (loaded from --sft-run-dir/model).

Writes a HuggingFace Dataset at data/preference_pairs/<strategy>_<seed>/ with
columns {prompt, chosen, rejected}, plus pairs_manifest_<strategy>_<seed>.json.

Strategy B is the default — captures where the SFT baseline is *confused*,
which is the intent of DPO. Run once per seed; pairs are reused across DPO
training seeds (the DPO seed only affects the training loop, not the pairs).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from findpo.labels import LABELS, format_prompt_messages  # noqa: E402
from findpo.sanity import check_preference_pair_format  # noqa: E402
from findpo.seeding import set_seed  # noqa: E402
from findpo.tokenizer_setup import render_chat, setup_tokenizer  # noqa: E402


def _strategy_a_rejected(true_label: str, rng: random.Random) -> str:
    return rng.choice([l for l in LABELS if l != true_label])


def _predict_sft(texts: list[str], system: str, sft_dir: Path, base_model: str,
                  attn_impl: str | None, batch_size: int = 8) -> list[list[float]]:
    """Run the SFT model over a list of texts, return logits over the 3 label tokens
    at the first generated position. Used by strategy B."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(base_model)
    # Right-padding: this function does a forward pass only (no generation),
    # so we want the LAST REAL token's logits at `attention_mask.sum(-1) - 1`.
    setup_tokenizer(tok, padding_side="right")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=bnb, device_map="auto",
        attn_implementation=attn_impl,
    )
    model = PeftModel.from_pretrained(base, str(sft_dir))
    model.eval()

    # Probe: take FIRST token id of each label string with a leading space
    # (since the chat template puts the assistant turn after a newline + role tag,
    # the next BPE token of "positive" may or may not have a leading space — try
    # both encodings and pick whichever the model places highest probability on
    # at the first generation step on a sample).
    candidate_ids: dict[str, list[int]] = {}
    for lbl in LABELS:
        ids_no_space = tok.encode(lbl, add_special_tokens=False)
        ids_with_space = tok.encode(" " + lbl, add_special_tokens=False)
        # First subword id of each variant — pick the more common encoding at runtime.
        candidate_ids[lbl] = list({ids_no_space[0], ids_with_space[0]})

    all_logits: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        prompts = [
            render_chat(tok, format_prompt_messages(system, t),
                        add_generation_prompt=True)
            for t in batch_texts
        ]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=512).to(model.device)
        with torch.no_grad():
            out = model(**enc)
        # logits at the last non-pad position of each row
        last_idx = enc["attention_mask"].sum(dim=1) - 1
        for r, idx in enumerate(last_idx.tolist()):
            logits = out.logits[r, idx]
            row = []
            for lbl in LABELS:
                # max over the candidate first-token ids
                row.append(float(logits[candidate_ids[lbl]].max().item()))
            all_logits.append(row)
    return all_logits


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="DPO config (datasets + prompt only used).")
    p.add_argument("--strategy", choices=("A", "B"), default="B")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--splits-dir", default="data/splits")
    p.add_argument("--out-dir", default="data/preference_pairs")
    p.add_argument("--sft-run-dir", help="Required for strategy B; path to SFT run dir (contains model/).")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(args.seed)
    rng = random.Random(args.seed)
    sys_prompt = cfg["prompt"]["system"]

    # Load merged train set
    from datasets import concatenate_datasets, load_from_disk, Dataset
    parts = []
    for src in cfg["datasets"]:
        if cfg["datasets"][src]["repo"].startswith("TODO"):
            continue
        parts.append(load_from_disk(str(Path(args.splits_dir) / f"{src}_train")))
    train = concatenate_datasets(parts)

    texts = list(train["text"])
    true_labels = list(train["label"])

    if args.strategy == "A":
        rejecteds = [_strategy_a_rejected(t, rng) for t in true_labels]
    else:
        if not args.sft_run_dir:
            raise SystemExit("Strategy B requires --sft-run-dir.")
        sft_model_dir = Path(args.sft_run_dir) / "model"
        logits = _predict_sft(
            texts, sys_prompt, sft_model_dir,
            base_model=cfg["model"]["name"],
            attn_impl=cfg["model"].get("attn_implementation"),
        )
        # rejected = argmax over non-true classes
        rejecteds = []
        for true_lbl, row in zip(true_labels, logits):
            row_by_label = dict(zip(LABELS, row))
            cand = [(l, v) for l, v in row_by_label.items() if l != true_lbl]
            cand.sort(key=lambda kv: kv[1], reverse=True)
            rejecteds.append(cand[0][0])

    # Build records — prompt is the rendered chat template up to assistant role,
    # chosen/rejected are bare label strings (TRL handles concatenation).
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    setup_tokenizer(tok, padding_side="right")   # rendering only — padding unused

    records: list[dict] = []
    for t, true_lbl, rej in zip(texts, true_labels, rejecteds):
        prompt = render_chat(tok, format_prompt_messages(sys_prompt, t),
                              add_generation_prompt=True)
        records.append({"prompt": prompt, "chosen": true_lbl, "rejected": rej})

    ok, det = check_preference_pair_format(records)
    if not ok:
        raise SystemExit(f"Preference pairs malformed: {det['issues']}")
    print("Sample pairs (first 3):")
    print(json.dumps(records[:3], indent=2)[:1500], "…")

    pairs_ds = Dataset.from_list(records)
    out = Path(args.out_dir) / f"strategy{args.strategy}_seed{args.seed}"
    pairs_ds.save_to_disk(str(out))

    h = hashlib.sha256()
    for r in records:
        h.update(r["prompt"].encode())
        h.update(r["chosen"].encode())
        h.update(r["rejected"].encode())
    manifest = {
        "strategy": args.strategy,
        "seed": args.seed,
        "n_pairs": len(records),
        "sha256": h.hexdigest(),
        "rejected_distribution": {
            lbl: sum(1 for r in records if r["rejected"] == lbl) for lbl in LABELS
        },
        "chosen_distribution": {
            lbl: sum(1 for r in records if r["chosen"] == lbl) for lbl in LABELS
        },
        "sft_run_dir": args.sft_run_dir,
    }
    (out.parent / f"manifest_strategy{args.strategy}_seed{args.seed}.json").write_text(
        json.dumps(manifest, indent=2)
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
