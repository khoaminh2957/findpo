# FinDPO reproduction

Reproduction of **FinDPO** (Iacovides, Zhou & Mandic 2025; [arXiv:2507.18417](https://arxiv.org/abs/2507.18417)): a 3-class financial sentiment classifier trained with DPO on Llama-3.

The repo is a baseline scaffold for downstream research on alternative label sources (multi-LLM annotation replacing FPB labels).

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

Target hardware is RTX 5090 / RTX PRO 6000 Blackwell (sm_120) on Vast.ai.
The pinned versions of `bitsandbytes` and `flash-attn` are the lowest
releases that ship sm_120 kernels; older releases crash at model load.

Vast.ai Docker template (verified 2026-05-13 on Docker Hub):

```
pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel
```

This matches the pinned `torch==2.7.0` + CUDA 12.8. The `devel` variant
ships nvcc + CUDA headers, which are needed to build flash-attn from
source. Image is ~8 GB; allocate >= 100 GB disk on Vast.ai.

Avoid `vastai/pytorch` (stuck on PyTorch 1.0 / CUDA 10.0).

If you want a prebuilt flash-attn (saves 30-60 min of build time), use
`axolotlai/axolotl-cloud-term:main-py3.11-cu128-2.9.1`. That image ships
PyTorch 2.9.1 (not an exact match) and is 14 GB, so you also need
`pip install -r requirements.txt --force-reinstall` to overwrite the
Axolotl-installed versions with the pinned ones.

Install (pip path, what I use on Vast.ai):

```bash
# Driver >= 570.x for Blackwell
nvidia-smi

python -m venv ~/findpo-env && source ~/findpo-env/bin/activate

# PyTorch from the cu128 channel
pip install --extra-index-url https://download.pytorch.org/whl/cu128 torch==2.7.0

# transformers, trl, peft, bitsandbytes 0.49.2, ...
pip install -r requirements.txt

# flash-attn 2.7.4.post1 to 2.8.3 only ship a source tarball on PyPI,
# so this compiles from source (30-60 min on first install; needs ninja,
# g++, CUDA toolkit on PATH, and ~16 GB RAM during the build).
pip install flash-attn==2.8.3 --no-build-isolation
# To skip the build, grab a prebuilt community wheel matching
# torch+cu128+sm_120 from:
#   https://huggingface.co/lldacing/flash-attention-prebuild-wheels
# If that also fails, set attn_implementation: "sdpa" in the configs;
# setup_check warns and all four model-load sites already auto-fallback.

huggingface-cli login           # HF token; Llama-3.1-8B-Instruct must be granted
wandb login                     # API key from https://wandb.ai/authorize

python scripts/00_setup_check.py    # must exit 0
pip freeze > requirements.lock.txt && git add requirements.lock.txt
```

`environment.yml` is the conda alternative and pins the same versions.

## Run a phase

### Phase 0: data fingerprint
```bash
python scripts/00_download_datasets.py --config configs/sft_baseline.yaml
git add data/manifest_phase0.json data/splits/*_idx.json
git commit -m "phase 0: data manifest"
git tag phase-0-complete
```

### Phase 1: SFT baseline (one seed per GPU, 3 seeds in parallel)
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

Pass criterion (FinDPO paper Table 2, FinSFT weighted F1):
- FPB >= 0.80 (paper FinSFT = 0.829)
- TFNS >= 0.82 (paper FinSFT = 0.850)
- NWGI >= 0.68 (paper FinSFT = 0.708)

### Phase 2: DPO repro
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

Pass criterion (FinDPO paper Table 2, weighted F1 mean ± std), FinDPO vs FinSFT:
- FPB: paper +3.6% (0.865 vs 0.829); repro target +2%
- TFNS: paper +2.2%; repro target +1%
- NWGI: paper +12.5%; repro target +8% (dominant driver)
- Simple mean of the three: +6.1%; repro target +4%

The paper's headline "+11%" is FinDPO vs FinGPT v3.3, not vs FinSFT.
Against the paper's own same-base SFT, the simple mean of the per-dataset
deltas above is +6.1%; the paper may use a different averaging (e.g.
weighted by dataset size) for its +11% figure, which Table 2 alone does
not let me reproduce.

## Reproducibility

- Training scripts refuse to launch with a dirty working tree
  (`git status --porcelain` non-empty); pass `--allow-dirty` only for
  exploratory runs that will not appear in the report.
- `data_manifest.json`, `env_info.json`, `git_info.json`, and `config.yaml`
  are written into every run directory.
- Metrics are double-logged to W&B and `results/exp_*/train_log.jsonl`.
- `aggregate_seeds.py` is the only path to a headline number in a report;
  do not cite a single-seed result.

## TODO

- Confirm the GPT-labeled financial news dataset against the paper PDF
  and fill the `gpt_news.repo` field in both configs (currently
  `TODO_VERIFY_FROM_PAPER`).
- After the first download on the remote box, pin `revision` commit
  hashes for every HF dataset and for the base model.
- Decide whether Phase 3 (trading sim) gets added back after Phase 2.
- Llama-3.3-70B extension experiment (deferred).
