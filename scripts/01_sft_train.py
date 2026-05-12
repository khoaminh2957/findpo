"""Phase 1 — SFT baseline training (QLoRA, Llama-3.1-8B-Instruct).

Usage:
    python scripts/01_sft_train.py --config configs/sft_baseline.yaml --seed 42
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from findpo.data import sha256_of_dataset, label_dist, SplitFingerprint, write_manifest  # noqa: E402
from findpo.env_info import assert_clean_tree, write_env_and_git  # noqa: E402
from findpo.labels import LABELS, format_prompt_messages  # noqa: E402
from findpo.paths import run_paths  # noqa: E402
from findpo.sanity import check_class_balance, check_train_test_no_overlap, check_tokenizer_alignment  # noqa: E402
from findpo.seeding import set_seed  # noqa: E402
from findpo.tokenizer_setup import render_chat, setup_tokenizer  # noqa: E402
from findpo.tracking import DualLogger, WandbCfg  # noqa: E402


def _load_splits(cfg: dict, splits_dir: Path):
    from datasets import concatenate_datasets, load_from_disk
    train_parts, test_parts = [], []
    for src in cfg["datasets"]:
        if cfg["datasets"][src]["repo"].startswith("TODO"):
            continue
        train_parts.append(load_from_disk(str(splits_dir / f"{src}_train")))
        test_parts.append(load_from_disk(str(splits_dir / f"{src}_test")))
    return concatenate_datasets(train_parts), concatenate_datasets(test_parts)


def _format_for_sft(example: dict, tokenizer, system: str) -> dict:
    """Apply Llama-3 chat template; only the assistant turn carries the label.

    Returns the chat-template text VERBATIM (including the literal
    `<|begin_of_text|>` at position 0). The SFTConfig's
    `dataset_kwargs={"add_special_tokens": False}` ensures SFTTrainer
    tokenizes WITHOUT auto-prepending BOS, so the final sequence has
    exactly one BOS — coming from the literal in the rendered text.
    """
    msgs = format_prompt_messages(system, example["text"]) + [
        {"role": "assistant", "content": example["label"]},
    ]
    text = render_chat(tokenizer, msgs, add_generation_prompt=False)
    # Defensive: a healthy Llama-3 template ends each assistant turn with
    # <|eot_id|>. If it doesn't, the template is broken (e.g. a third-party
    # mirror that hard-codes a trailing generation prompt) — refuse to train
    # because completion-only masking would compute loss on the wrong tokens.
    if not text.rstrip().endswith(tokenizer.eos_token):
        raise RuntimeError(
            "Chat template did not end the assistant turn with eos. "
            "This usually means the tokenizer's chat_template appends a "
            "trailing generation prompt regardless of add_generation_prompt. "
            "Refusing to produce broken training data. "
            f"Last 80 chars: {text[-80:]!r}"
        )
    return {"text": text}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--splits-dir", default="data/splits")
    p.add_argument("--allow-dirty", action="store_true")
    p.add_argument("--skip-wandb", action="store_true")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(args.seed)

    assert_clean_tree(REPO_ROOT, allow_dirty=args.allow_dirty)

    paths = run_paths(cfg["train"]["output_dir_base"], cfg["experiment"]["id"], args.seed)
    paths.ensure()
    shutil.copy(args.config, paths.config_yaml)
    write_env_and_git(REPO_ROOT, paths.env_info, paths.git_info)

    # -------- data
    splits_dir = Path(args.splits_dir)
    train_ds, test_ds = _load_splits(cfg, splits_dir)

    ok, det = check_train_test_no_overlap(train_ds, test_ds)
    assert ok, det
    ok, det = check_class_balance(train_ds, "train"); print("class_balance(train):", det)
    ok, det = check_class_balance(test_ds, "test"); print("class_balance(test):", det)

    fingerprints = [
        SplitFingerprint("merged_train", "merged", None, None, len(train_ds),
                         sha256_of_dataset(train_ds), label_dist(train_ds)),
        SplitFingerprint("merged_test", "merged", None, None, len(test_ds),
                         sha256_of_dataset(test_ds), label_dist(test_ds)),
    ]
    write_manifest(paths.data_manifest, fingerprints,
                   extras={"split_seed": cfg["split"]["split_seed"]})

    # -------- model + tokenizer
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig)
    from peft import LoraConfig, prepare_model_for_kbit_training
    from trl import DataCollatorForCompletionOnlyLM, SFTTrainer, SFTConfig

    model_name = cfg["model"]["name"]
    tok = AutoTokenizer.from_pretrained(model_name, revision=cfg["model"].get("revision"))
    setup_tokenizer(tok, padding_side="right")   # SFT training → right pad

    ok, det = check_tokenizer_alignment(tok, cfg["prompt"]["system"], "test text")
    assert ok, f"tokenizer chat template broken: {det}"

    bnb = BitsAndBytesConfig(
        load_in_4bit=cfg["quantization"]["load_in_4bit"],
        bnb_4bit_quant_type=cfg["quantization"]["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, cfg["quantization"]["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=cfg["quantization"]["bnb_4bit_use_double_quant"],
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=cfg["model"].get("revision"),
        quantization_config=bnb,
        device_map="auto",
        attn_implementation=cfg["model"].get("attn_implementation"),
    )
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=cfg["train"]["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model.config.use_cache = False

    peft_cfg = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        bias=cfg["lora"]["bias"],
        task_type=cfg["lora"]["task_type"],
        target_modules=cfg["lora"]["target_modules"],
    )

    # -------- format datasets for SFT
    sys_prompt = cfg["prompt"]["system"]
    train_fmt = train_ds.map(lambda ex: _format_for_sft(ex, tok, sys_prompt),
                              remove_columns=train_ds.column_names)
    test_fmt = test_ds.map(lambda ex: _format_for_sft(ex, tok, sys_prompt),
                            remove_columns=test_ds.column_names)

    # -------- tracking
    wandb_cfg_obj = WandbCfg(
        project=cfg["wandb"]["project"],
        tags=cfg["wandb"]["tags"] + [f"seed-{args.seed}"],
        enabled=(not args.skip_wandb) and ("wandb" in cfg["train"]["report_to"]),
    )
    logger = DualLogger.start(paths, config=cfg, wandb_cfg=wandb_cfg_obj,
                              run_name=f"{cfg['experiment']['id']}_seed{args.seed}")

    # -------- trainer
    sft_args = SFTConfig(
        output_dir=str(paths.checkpoints_dir),
        seed=args.seed,
        num_train_epochs=cfg["train"]["num_train_epochs"],
        per_device_train_batch_size=cfg["train"]["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["train"]["gradient_accumulation_steps"],
        per_device_eval_batch_size=cfg["train"]["per_device_eval_batch_size"],
        learning_rate=cfg["train"]["learning_rate"],
        lr_scheduler_type=cfg["train"]["lr_scheduler_type"],
        warmup_ratio=cfg["train"]["warmup_ratio"],
        weight_decay=cfg["train"]["weight_decay"],
        max_seq_length=cfg["train"]["max_seq_length"],
        bf16=cfg["train"]["bf16"],
        gradient_checkpointing=cfg["train"]["gradient_checkpointing"],
        optim=cfg["train"]["optim"],
        logging_steps=cfg["train"]["logging_steps"],
        save_strategy=cfg["train"]["save_strategy"],
        save_total_limit=cfg["train"]["save_total_limit"],
        eval_strategy=cfg["train"]["eval_strategy"],
        load_best_model_at_end=cfg["train"]["load_best_model_at_end"],
        metric_for_best_model=cfg["train"]["metric_for_best_model"],
        greater_is_better=cfg["train"]["greater_is_better"],
        report_to=cfg["train"]["report_to"] if not args.skip_wandb else "none",
        run_name=f"{cfg['experiment']['id']}_seed{args.seed}",
        packing=False,
        dataset_text_field="text",
        # Disable the tokenizer's auto-BOS post-processor. Our chat-template
        # text already contains the literal `<|begin_of_text|>` at position 0,
        # and DPOTrainer for decoder-only models hard-codes
        # add_special_tokens=False — passing False here keeps SFT and DPO
        # sequences bit-identical in their prompt prefix.
        # See findpo/tokenizer_setup.py (footgun #2) for the full reasoning.
        dataset_kwargs={"add_special_tokens": False},
    )

    # Completion-only masking: train CE loss ONLY on assistant tokens, not on
    # system+user prompt. The response template is the exact Llama-3 assistant
    # header — DataCollatorForCompletionOnlyLM finds its token sequence in each
    # row and sets labels to -100 for everything before it.
    response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    response_template_ids = tok.encode(response_template, add_special_tokens=False)
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template_ids,
        tokenizer=tok,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_fmt,
        eval_dataset=test_fmt,
        peft_config=peft_cfg,
        processing_class=tok,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(str(paths.model_dir))
    tok.save_pretrained(str(paths.model_dir))

    # final eval is invoked from 01_sft_eval.py — keep this script focused on training
    logger.finish({"phase": "sft-train-complete",
                    "n_train": len(train_ds),
                    "n_test": len(test_ds)})
    print(f"SFT done → {paths.model_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
