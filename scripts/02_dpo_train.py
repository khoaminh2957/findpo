"""Phase 2 — DPO training (QLoRA path).

Usage:
    python scripts/02_dpo_train.py --config configs/dpo_repro_qlora.yaml \\
                                    --seed 42 \\
                                    --pairs-dir data/preference_pairs/strategyB_seed42

When peft_config is passed and ref_model is None, TRL DPOTrainer disables the
adapter to compute reference log-probs. This is the Zephyr / alignment-handbook
recipe; do not pass a separate ref_model for QLoRA.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from findpo.env_info import assert_clean_tree, write_env_and_git  # noqa: E402
from findpo.paths import run_paths  # noqa: E402
from findpo.sanity import check_preference_pair_format, check_tokenizer_alignment  # noqa: E402
from findpo.seeding import set_seed  # noqa: E402
from findpo.tracking import DualLogger, WandbCfg  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--pairs-dir", required=True)
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

    # -------- pairs
    from datasets import load_from_disk
    pairs = load_from_disk(args.pairs_dir)
    sample = [pairs[i] for i in range(min(10, len(pairs)))]
    ok, det = check_preference_pair_format(sample)
    assert ok, det
    # 90/10 train/eval split on pairs (eval used for monitoring DPO reward acc)
    pairs = pairs.train_test_split(test_size=0.1, seed=args.seed)
    pairs_train, pairs_eval = pairs["train"], pairs["test"]

    # -------- model + tokenizer
    import torch
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import DPOConfig, DPOTrainer

    model_name = cfg["model"]["name"]
    tok = AutoTokenizer.from_pretrained(model_name, revision=cfg["model"].get("revision"))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

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
        model, use_gradient_checkpointing=cfg["train"]["gradient_checkpointing"],
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

    # -------- tracking
    wandb_cfg_obj = WandbCfg(
        project=cfg["wandb"]["project"],
        tags=cfg["wandb"]["tags"] + [f"seed-{args.seed}"],
        enabled=(not args.skip_wandb) and ("wandb" in cfg["train"]["report_to"]),
    )
    logger = DualLogger.start(paths, config=cfg, wandb_cfg=wandb_cfg_obj,
                              run_name=f"{cfg['experiment']['id']}_seed{args.seed}")

    dpo_args = DPOConfig(
        output_dir=str(paths.checkpoints_dir),
        seed=args.seed,
        beta=cfg["train"]["beta"],
        loss_type=cfg["train"]["loss_type"],
        label_smoothing=cfg["train"]["label_smoothing"],
        num_train_epochs=cfg["train"]["num_train_epochs"],
        per_device_train_batch_size=cfg["train"]["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["train"]["gradient_accumulation_steps"],
        per_device_eval_batch_size=cfg["train"]["per_device_eval_batch_size"],
        learning_rate=cfg["train"]["learning_rate"],
        lr_scheduler_type=cfg["train"]["lr_scheduler_type"],
        warmup_ratio=cfg["train"]["warmup_ratio"],
        weight_decay=cfg["train"]["weight_decay"],
        max_length=cfg["train"]["max_length"],
        max_prompt_length=cfg["train"]["max_prompt_length"],
        bf16=cfg["train"]["bf16"],
        gradient_checkpointing=cfg["train"]["gradient_checkpointing"],
        optim=cfg["train"]["optim"],
        logging_steps=cfg["train"]["logging_steps"],
        save_steps=cfg["train"]["save_steps"],
        save_total_limit=cfg["train"]["save_total_limit"],
        eval_strategy=cfg["train"]["eval_strategy"],
        load_best_model_at_end=cfg["train"]["load_best_model_at_end"],
        report_to=cfg["train"]["report_to"] if not args.skip_wandb else "none",
        run_name=f"{cfg['experiment']['id']}_seed{args.seed}",
        generate_during_eval=cfg["train"]["generate_during_eval"],
        precompute_ref_log_probs=cfg["train"]["precompute_ref_log_probs"],
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,         # peft_config implies adapter-disable as reference
        args=dpo_args,
        train_dataset=pairs_train,
        eval_dataset=pairs_eval,
        peft_config=peft_cfg,
        processing_class=tok,
    )

    trainer.train()
    trainer.save_model(str(paths.model_dir))
    tok.save_pretrained(str(paths.model_dir))

    logger.finish({"phase": "dpo-train-complete",
                    "n_pairs_train": len(pairs_train),
                    "n_pairs_eval": len(pairs_eval)})
    print(f"DPO done → {paths.model_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
