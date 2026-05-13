# FinDPO reproduction

Reproduction of **FinDPO** (Iacovides, Zhou & Mandic 2025; [arXiv:2507.18417](https://arxiv.org/abs/2507.18417)) — 3-class financial sentiment classifier trained with DPO on top of Llama-3.

This repo is the baseline scaffold for downstream research on alternative label sources (multi-LLM annotation replacing FPB labels).

## Scope

| Item | Choice | Notes |
|---|---|---|
| Base model | `unsloth/Meta-Llama-3.1-8B-Instruct` | Paper used 3.0; 3.1 has same architecture, deviation logged |
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

**Hardware:** target is RTX 5090 / RTX PRO 6000 Blackwell (sm_120) on
Vast.ai. The version floors below are forced by Blackwell support —
earlier `bitsandbytes` and `flash-attn` releases ship no sm_120 kernels
and will crash at model load.

**Vast.ai Docker template (verified 2026-05-13 via Docker Hub):**

```
pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel
```

This image matches the pinned `torch==2.7.0` + CUDA 12.8 (Blackwell-capable)
exactly. `devel` variant ships nvcc + CUDA headers needed to build
flash-attn from source. Image size ~8GB; allocate ≥100GB disk on Vast.ai.

DO NOT use `vastai/pytorch` (outdated to PyTorch 1.0 / CUDA 10.0).

Alternative if you want flash-attn pre-built (saves 30-60min build):
`axolotlai/axolotl-cloud-term:main-py3.11-cu128-2.9.1` — but ships
PyTorch 2.9.1 (not exact match) and 14GB; you'll need to
`pip install -r requirements.txt --force-reinstall` to overwrite
the Axolotl-installed library versions with our pinned ones.

**Pip-first install (recommended for cloud rentals like Vast.ai):**

```bash
# 1. Verify GPU + driver. Need driver ≥ 570.x for Blackwell.
nvidia-smi
# 2. Fresh venv (Vast.ai images usually have python 3.10+).
python -m venv ~/findpo-env && source ~/findpo-env/bin/activate
# 3. PyTorch first, from the cu128 channel.
pip install --extra-index-url https://download.pytorch.org/whl/cu128 torch==2.7.0
# 4. All other pinned deps (transformers, trl, peft, bitsandbytes 0.49.2, …).
pip install -r requirements.txt
# 5. flash-attn — empirically verified on 2026-05-13 via pip / PyPI API:
#    flash-attn 2.7.4.post1 through 2.8.3 ship ONLY a source tarball on PyPI
#    (no precompiled wheels). `pip install flash-attn==2.8.3 --no-build-isolation`
#    will COMPILE FROM SOURCE — 30-60 min on first install, needs ninja + g++ +
#    CUDA toolkit in PATH + ~16GB RAM during build. Don't panic when CPU sits
#    at 100% for an hour.
pip install flash-attn==2.8.3 --no-build-isolation
#    To skip the build, grab a prebuilt community wheel matching torch+cu128+sm_120:
#      https://huggingface.co/lldacing/flash-attention-prebuild-wheels
#    Pick the file name matching your stack (e.g. flash_attn-2.7.4...-cp310-...whl)
#    and `pip install <url>`.
#    Last-resort fallback: switch configs to attn_implementation: "sdpa" —
#    setup_check warns + all four model-load sites have try/except auto-fallback.
# 6. HF auth.
huggingface-cli login           # paste your HF token (Llama-3.1-8B-Instruct must be granted)
# 7. W&B.
wandb login                     # paste API key from https://wandb.ai/authorize
# 8. Phase-0 smoke test.
python scripts/00_setup_check.py    # must exit 0
# 9. Freeze the exact versions for reproducibility.
pip freeze > requirements.lock.txt && git add requirements.lock.txt
```

`environment.yml` (conda) is provided as an alternative install path and
pins the same versions.

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

**Paper-grounded pass criterion** (FinDPO Table 2, FinSFT weighted F1):
- FPB ≥ 0.80 (paper FinSFT = 0.829)
- TFNS ≥ 0.82 (paper FinSFT = 0.850)
- NWGI ≥ 0.68 (paper FinSFT = 0.708)

### Phase 2 — DPO repro
```bash
# 1. Build preference pairs once per seed (strategy B uses the seed-matched SFT model).
for s in 42 123 7; do
  python scripts/02_build_preference_pairs.py \
    --config configs/dpo_repro_qlora.yaml \
    --strategy AB --seed ${s} \
    --sft-run-dir results/exp_sft_baseline_seed${s}
done

# 2. Train (parallel across 3 GPUs).
CUDA_VISIBLE_DEVICES=0 python scripts/02_dpo_train.py --config configs/dpo_repro_qlora.yaml --seed 42 \
  --pairs-dir data/preference_pairs/strategyAB_seed42 &
CUDA_VISIBLE_DEVICES=1 python scripts/02_dpo_train.py --config configs/dpo_repro_qlora.yaml --seed 123 \
  --pairs-dir data/preference_pairs/strategyAB_seed123 &
CUDA_VISIBLE_DEVICES=2 python scripts/02_dpo_train.py --config configs/dpo_repro_qlora.yaml --seed 7 \
  --pairs-dir data/preference_pairs/strategyAB_seed7 &
wait

# 3. Eval + aggregate.
for s in 42 123 7; do
  python scripts/02_dpo_eval.py --run-dir results/exp_dpo_repro_qlora_seed${s} --eval-dataset fpb
done
python scripts/aggregate_seeds.py --exp-id dpo_repro_qlora --eval-dataset fpb \
  --out-md reports/phase_2_dpo_table.md
```

**Paper-grounded pass criterion** (FinDPO Table 2, weighted F1 mean ± std):
- FPB FinDPO − FinSFT: paper +3.6% (0.865 − 0.829) → repro ≥ +2%
- TFNS FinDPO − FinSFT: paper +2.2% → repro ≥ +1%
- NWGI FinDPO − FinSFT: paper +12.5% → repro ≥ +8% (dominant driver)
- Average: paper +9.7% → repro ≥ +6%

Note: paper's headline "+11%" is FinDPO vs FinGPT v3.3 (NOT vs FinSFT).
Vs FinSFT (paper's own same-base SFT), gap is +9.7% average.

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
