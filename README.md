# FinDPO reproduction

Reproduction of **FinDPO** (Iacovides, Zhou & Mandic 2025; [arXiv:2507.18417](https://arxiv.org/abs/2507.18417)) — 3-class financial sentiment classifier trained with DPO on top of Llama-3.

This repo is the baseline scaffold for downstream research on alternative label sources (multi-LLM annotation replacing FPB labels).

## Scope

| Item | Choice | Notes |
|---|---|---|
| Base model | `meta-llama/Meta-Llama-3.1-8B-Instruct` | Paper used 3.0; 3.1 has same architecture, deviation logged |
| Training path | QLoRA (NF4 4-bit + LoRA r=16) | No full fine-tune |
| Phases | 0 → 1 (SFT) → 2 (DPO) → 4 (docs) | Phase 3 (trading sim) out of scope |
| Seeds | 42, 123, 7 (mean ± std) | Three runs per condition |
| Tracking | W&B + JSONL fallback | Dual logger |

## Repo layout

```
findpo/                  Importable package (utilities)
  seeding.py             R2  deterministic seeding + param-hash
  paths.py               R10 output dir structure
  env_info.py            R1+R6 env / git capture, dirty-tree guard
  data.py                R5 load + split + SHA256 manifest
  labels.py              Canonical label set + chat-template helpers
  tracking.py            R4+R9 dual W&B + JSONL logger
  sanity.py              R8 sanity tests
  evaluate.py            Shared classification eval (SFT and DPO call this)

scripts/                 Entry points (run from REMOTE_GPU)
  00_setup_check.py      Phase 0 smoke test
  00_download_datasets.py Phase 0 data fetch + manifest
  01_sft_train.py        Phase 1 SFT (QLoRA)
  01_sft_eval.py         Phase 1 eval wrapper
  02_build_preference_pairs.py  Phase 2 preference pair construction
  02_dpo_train.py        Phase 2 DPO (QLoRA, ref=adapter-disabled)
  02_dpo_eval.py         Phase 2 eval wrapper
  aggregate_seeds.py     Mean ± std across 3 seeds

configs/                 YAML configs (no hardcoded hyperparams in scripts)
  sft_baseline.yaml
  dpo_repro_qlora.yaml

data/                    Datasets (gitignored except manifest)
results/                 Run outputs (gitignored)
logs/                    tmux/training logs (gitignored)
reports/                 Phase summary reports + final REPRODUCTION_REPORT.md
```

## Setup (REMOTE_GPU)

1. `conda env create -f environment.yml && conda activate findpo`
   - PyTorch is pinned to 2.5.1 with CUDA 12.4. Adjust `environment.yml` if the
     remote has a different CUDA runtime.
2. `pip install -r requirements.txt` (no-op after step 1).
3. `pip install flash-attn==2.7.2.post1 --no-build-isolation`
4. `huggingface-cli login`  — paste HF token with read access; the Llama-3.1
   repo must be approved on your HF account.
5. `wandb login` — paste API key from <https://wandb.ai/authorize>.
6. `python scripts/00_setup_check.py` — must exit 0.
7. `pip freeze > requirements.lock.txt && git add requirements.lock.txt`

## Run a phase

### Phase 0 — data fingerprint
```bash
python scripts/00_download_datasets.py --config configs/sft_baseline.yaml
git add data/manifest_phase0.json data/splits/*_idx.json
git commit -m "phase 0: data manifest"
git tag phase-0-complete
```

### Phase 1 — SFT baseline (one seed per GPU, 3 seeds in parallel)
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/01_sft_train.py --config configs/sft_baseline.yaml --seed 42 &
CUDA_VISIBLE_DEVICES=1 python scripts/01_sft_train.py --config configs/sft_baseline.yaml --seed 123 &
CUDA_VISIBLE_DEVICES=2 python scripts/01_sft_train.py --config configs/sft_baseline.yaml --seed 7 &
wait
for s in 42 123 7; do
  python scripts/01_sft_eval.py --run-dir results/exp_sft_baseline_seed${s} --eval-dataset fpb
done
python scripts/aggregate_seeds.py --exp-id sft_baseline --eval-dataset fpb \
  --out-md reports/phase_1_sft_table.md
```

**Pass criterion**: mean accuracy on FPB test ≥ 80%.

### Phase 2 — DPO repro
```bash
# 1. Build preference pairs once per seed (strategy B uses the seed-matched SFT model).
for s in 42 123 7; do
  python scripts/02_build_preference_pairs.py \
    --config configs/dpo_repro_qlora.yaml \
    --strategy B --seed ${s} \
    --sft-run-dir results/exp_sft_baseline_seed${s}
done

# 2. Train (parallel across 3 GPUs).
CUDA_VISIBLE_DEVICES=0 python scripts/02_dpo_train.py --config configs/dpo_repro_qlora.yaml --seed 42 \
  --pairs-dir data/preference_pairs/strategyB_seed42 &
CUDA_VISIBLE_DEVICES=1 python scripts/02_dpo_train.py --config configs/dpo_repro_qlora.yaml --seed 123 \
  --pairs-dir data/preference_pairs/strategyB_seed123 &
CUDA_VISIBLE_DEVICES=2 python scripts/02_dpo_train.py --config configs/dpo_repro_qlora.yaml --seed 7 \
  --pairs-dir data/preference_pairs/strategyB_seed7 &
wait

# 3. Eval + aggregate.
for s in 42 123 7; do
  python scripts/02_dpo_eval.py --run-dir results/exp_dpo_repro_qlora_seed${s} --eval-dataset fpb
done
python scripts/aggregate_seeds.py --exp-id dpo_repro_qlora --eval-dataset fpb \
  --out-md reports/phase_2_dpo_table.md
```

**Pass criterion**: mean DPO accuracy − mean SFT accuracy ≥ +8% on FPB
(paper claims +11%; allow margin).

## Reproducibility guardrails

- Every training script refuses to launch with a dirty working tree
  (`git status --porcelain` non-empty). Pass `--allow-dirty` only for
  exploratory runs that will not appear in the report.
- `data_manifest.json`, `env_info.json`, `git_info.json`, `config.yaml` are
  written into every run directory.
- All metrics are double-logged: W&B + `results/exp_*/train_log.jsonl`.
- `aggregate_seeds.py` is the only sanctioned way to produce a headline
  number in a report — never cite a single-seed result.

## Outstanding TODOs

- [ ] Confirm the GPT-labeled financial news dataset repo against the paper
  PDF and fill the `gpt_news.repo` field in both configs (currently
  `TODO_VERIFY_FROM_PAPER`).
- [ ] After first download on remote, pin `revision` commit hashes for every
  HF dataset and the base model.
- [ ] Decide whether Phase 3 (trading sim) is added back after Phase 2.
- [ ] Llama-3.3-70B extension experiment (deferred).
