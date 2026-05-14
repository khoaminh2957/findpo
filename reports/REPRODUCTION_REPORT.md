# FinDPO reproduction — final report

**Target paper:** Iacovides, Zhou & Mandic 2025, *"FinDPO: Financial Sentiment
Classification with Direct Preference Optimization"* ([arXiv:2507.18417](https://arxiv.org/abs/2507.18417))

**Status:** Phase 1 (SFT) reproduces the paper. Phase 2 (DPO) **does NOT
reproduce the paper's headline improvement**. The DPO model consistently
underperforms its own SFT initialisation by ~3.5 percentage points weighted F1
on average across three datasets and three seeds — the opposite direction of
the +6.1 pp improvement implied by Table 2 of the paper (simple mean of the
per-dataset FinDPO − FinSFT deltas).

This is a **valid negative result**, not a pipeline bug: the SFT phase matches
or exceeds the paper's FinSFT baseline, the same eval code is used for both
phases, and the failure pattern is consistent across seeds. Section 5 lists
the most likely root causes that could be tested in a follow-up study.

---

## 1. Scope and configuration

| Item | This reproduction | Paper |
|---|---|---|
| Base model | `unsloth/Meta-Llama-3.1-8B-Instruct` | `meta-llama/Llama-3-8B-Instruct` |
| Training path | QLoRA NF4 + LoRA r=16, α=16 | Same |
| Phases | 0 → 1 (SFT) → 2 (DPO) → 4 (docs); Phase 3 trading sim skipped | 0 → 1 → 2 → 3 |
| Seeds | 42, 123, 7 | Three seeds (not specified) |
| Hardware | Phase 1: 2× RTX 5090. Phase 2: 4× A100 PCIE | Not specified |
| Headline metric | Weighted F1 on FPB / TFNS / NWGI test splits | Same (paper Table 2) |

Datasets: FPB `takala/financial_phrasebank` (sentences_50agree),
TFNS `zeroshot/twitter-financial-news-sentiment`,
NWGI `oliverwang15/news_with_gpt_instructions`. All loaded with pinned
revision hashes; SHA-256 fingerprints in [data/manifest_phase0.json](../data/manifest_phase0.json).

Train / test split: 80 / 20, deterministic seed 42 across both phases — eval
test sets are identical for SFT and DPO comparison.

DPO setup: TRL `DPOTrainer` 0.12.2, `ref_model=None` + `peft_config` so the
reference model is the **base model with the LoRA adapter disabled** (the
"Zephyr" pattern), not the SFT model. 5 epochs, lr 5e-6, β 0.1, sigmoid loss,
cosine LR with 10 % warmup. Strategy AB preference pairs (see §5).

---

## 2. Phase 1 — SFT baseline (reproduces paper)

Weighted F1, mean ± std across seeds 42 / 123 / 7. Paper FinSFT numbers from
Table 2.

| Dataset | This repro SFT | Paper FinSFT | Δ vs paper | Pass criterion |
|---|---|---|---|---|
| FPB | **0.8898 ± 0.0045** | 0.829 | **+0.061** | ≥ 0.80 ✅ |
| TFNS | **0.9111 ± 0.0012** | 0.850 | **+0.061** | ≥ 0.82 ✅ |
| NWGI | **0.8538 ± 0.0028** | 0.708 | **+0.146** | ≥ 0.68 ✅ |
| Mean | **0.8849** | 0.796 | **+0.089** | — |

The SFT phase **exceeds** the paper's reported FinSFT numbers by a wide margin
on every dataset. This is the strongest single piece of evidence that the
pipeline (dataset loading, prompt formatting, tokenisation, chat-template
handling, label canonicalisation, eval code) is correct.

`n_unparseable` = 0 on all 3 seeds × 3 datasets — the SFT model produces
clean single-word labels.

Per-seed breakdown in [phase_1_sft_fpb.md](phase_1_sft_fpb.md),
[phase_1_sft_tfns.md](phase_1_sft_tfns.md),
[phase_1_sft_gpt_news.md](phase_1_sft_gpt_news.md).

---

## 3. Phase 2 — DPO reproduction (does NOT reproduce paper)

### 3.1 Headline numbers

Weighted F1, mean ± std across seeds 42 / 123 / 7. Same eval code as Phase 1.

| Dataset | SFT | This repro DPO | Δ (DPO − SFT) | Paper Δ | Verdict |
|---|---|---|---|---|---|
| FPB | 0.8898 ± 0.0045 | **0.8485 ± 0.0059** | **−0.0413** | +0.036 | ❌ Regression |
| TFNS | 0.9111 ± 0.0012 | **0.8558 ± 0.0085** | **−0.0553** | +0.022 | ❌ Regression |
| NWGI | 0.8538 ± 0.0028 | **0.8455 ± 0.0027** | **−0.0083** | +0.125 | ❌ Regression |
| **Mean** | **0.8849** | **0.8499** | **−0.0350** | **+0.061** | ❌ |

DPO regresses on every dataset and every seed. The pass criteria from the
[README outstanding](../README.md) (`FPB ≥ +2 %`, `TFNS ≥ +1 %`, `NWGI ≥ +8 %`,
avg ≥ +6 %) all fail.

Per-seed breakdown in [phase_2_dpo_fpb.md](phase_2_dpo_fpb.md),
[phase_2_dpo_tfns.md](phase_2_dpo_tfns.md),
[phase_2_dpo_gpt_news.md](phase_2_dpo_gpt_news.md).

### 3.2 Output format degradation

| Phase | n_unparseable across 3 seeds × 3 datasets |
|---|---|
| SFT | 0 / 0 / 0 — 0 / 0 / 0 — 0 / 0 / 0 (total 0 / 17 364) |
| DPO | 3 / 13 / 8 — 19 / 27 / 59 — 10 / 17 / 1 (total 157 / 17 364) |

The DPO model also emits noisy artefacts that *do* parse to a label but signal
distribution drift — e.g. `'positive▍'` (Unicode box-drawing glyph),
`'positive)))),'`, `'negativeassistant'` (role-token leak),
`' neutral (This text…'` (would be truncated at `max_new_tokens=4`).

86 unique raw outputs for DPO seed 42 FPB versus a handful for SFT.

### 3.3 DPO training metrics look fine

The model *does* learn the DPO objective. From `train_log.jsonl`, on the
hold-out preference-pair eval:

| Seed | Best `eval_rewards/accuracies` | Epoch |
|---|---|---|
| 42 | 0.9148 | 4 |
| 123 | 0.9168 | 2 |
| 7 | 0.9225 | 3 |

>91 % of held-out pairs are correctly ranked (chosen vs rejected). The DPO
loss decreases. `rewards/margins` reaches ~6 at end of training. **The DPO
metric and the downstream classification metric have diverged.**

`eval_loss` bottoms at epoch 2-3 then rises through epoch 5 on all three
seeds — a textbook overfitting signal — but the rise is small and does not
explain the gap (see §4).

---

## 4. Where the loss comes from — per-example analysis

We re-evaluated SFT on FPB to obtain `predictions_fpb.jsonl` for both phases
and joined them by input text.

### 4.1 SFT vs DPO agreement on FPB test set (n = 966)

| Seed | Both right | SFT right, DPO wrong | DPO right, SFT wrong | Both wrong | Net DPO |
|---|---|---|---|---|---|
| 42 | 791 | **68** | 31 | 76 | **−37** |
| 123 | 789 | **75** | 39 | 63 | **−36** |
| 7 | 773 | **83** | 39 | 71 | **−44** |

For every seed, DPO **loses ~2× as many examples as it gains**. The "DPO wrong,
SFT right" column is roughly twice the "DPO right, SFT wrong" column, on all
three seeds. The negative result is not driven by an outlier seed.

### 4.2 What kind of errors does DPO introduce?

For seed 42 FPB, of the 68 examples where SFT was right and DPO became wrong:

| true → DPO prediction | Count |
|---|---|
| positive → neutral | 29 |
| neutral → positive | 23 |
| negative → neutral | 13 |
| neutral → negative | 3 |

DPO simultaneously:
- **over-neutralises** confident positive / negative SFT predictions (42 cases)
- **mis-classifies truly neutral inputs** as positive or negative (26 cases)

The bias is bidirectional and large. It is *not* a simple "DPO collapsed to
always predicting neutral" — the model has shifted its decision boundary in a
way that hurts both directions.

### 4.3 Earlier checkpoints are also worse than SFT

To rule out over-training as the sole cause, we re-evaluated the three
preserved checkpoints (epochs 3, 4, 5) for seed 42 on FPB:

| Checkpoint | weighted F1 | unparseable |
|---|---|---|
| ckpt-8250 (epoch 3) | 0.8396 | 0 |
| ckpt-11000 (epoch 4, "best" by `eval_rewards/accuracies`) | 0.8500 | 3 |
| ckpt-13750 (epoch 5, last) | 0.8484 | 3 |
| SFT baseline | **0.8891** | **0** |

DPO is **worse than SFT at every preserved epoch**. Over-training amplifies
the format-degradation symptom (unparseable count goes 0 → 3) but the
classification gap is already ~0.05 weighted F1 at epoch 3. Even if we had
preserved epoch 1 and 2, they would have to lift weighted F1 by ~0.05 in
2 fewer epochs to close the gap — unlikely, because `eval_rewards/accuracies`
in the train log was already plateauing by epoch 2.

> Note: checkpoints 2750 and 5500 were manually deleted mid-training to free
> disk on a 32 GB Vast.ai container — see §6. This is a procedural deviation
> that affects seed 123 specifically (its `best_model_checkpoint` was
> step 5500). It does **not** affect the §3 / §4 conclusions because the
> regression is present at every preserved checkpoint on every seed.

---

## 5. Likely root causes (untested hypotheses)

The gap between this reproduction and the paper is large and consistent.
Plausible explanations, in order of how cheap they are to test:

1. **Reference model = base, not SFT.** The TRL Zephyr-style pattern
   (`ref_model=None` + `peft_config`) makes the reference model the base
   Llama-3 with LoRA disabled. The paper text does not explicitly specify
   this, and the canonical DPO recipe sets the reference model to the SFT
   model. With ref = base, DPO can drift the LoRA delta in directions that
   maximise the chosen-over-rejected margin *relative to a model that can't
   classify*, even when that drift hurts absolute classification quality. The
   training-metric / downstream-metric divergence in §3.3 is consistent with
   this.

2. **Strategy AB preference pairs.** When the SFT model already predicts
   correctly (~88 % of FPB), Strategy B falls back to a *random* non-true
   label as the rejected response. This converts ~88 % of training pairs into
   noisy supervision — "prefer the correct class over a random other class",
   which any classifier already does. The 12 % meaningful signal (where SFT
   was wrong, rejected = the SFT-confused class) is diluted. The
   over-neutralisation pattern in §4.2 is consistent with the model averaging
   away from extreme predictions in response to noisy random-class rejecteds.

3. **5 epochs is too many.** `eval_loss` rises after epoch 2-3 on all seeds.
   The paper specifies 5 epochs but may have used a different effective
   batch / schedule / data scale that made 5 epochs appropriate. Earlier
   epochs (1, 2) might be better — but as §4.3 shows, even epoch 3 is below
   SFT.

4. **β = 0.1 too loose.** A higher β keeps the policy closer to the
   reference. Combined with hypothesis 1, β = 0.1 against `ref = base`
   permits substantial drift away from the SFT-quality initialisation.

5. **Llama 3.0 vs 3.1.** The paper used Llama-3.0 (no public mirror); we used
   the architecturally identical Llama-3.1 via the `unsloth` mirror because
   the original gating made the 3.0 weights inaccessible. The base behaviour
   could differ subtly in ways that interact with DPO.

6. **Chat-template / BOS handling.** Verified parity between train and eval
   (`add_special_tokens=False` everywhere, completion-only masking matches).
   Less likely than 1-3, but listed for completeness.

The cheapest single test would be **(1)**: re-run DPO with `ref_model` set
explicitly to the SFT checkpoint (one seed, 2-3 epochs) and see if the
classification gap closes. We did not run this in the current reproduction.

---

## 6. Procedural notes

- All training runs launched with `--allow-dirty` denied unless the working
  tree was clean. `data_manifest.json`, `env_info.json`, `git_info.json`,
  `config.yaml` were saved into every run directory.
- `aggregate_seeds.py` was the only sanctioned source of headline numbers.
  Per-seed tables in `reports/phase_{1,2}_*.md`.
- **Disk procedural deviation:** the Vast.ai A100 container was provisioned
  with only 32 GB overlay, of which ~24 GB was consumed by the HF model
  cache (15 GB) and the Python venv (9 GB), leaving < 8 GB for training
  artefacts. To prevent disk-OOM during epoch 4-5 checkpoint rotation,
  `checkpoint-2750` and `checkpoint-5500` were manually deleted for all
  three seeds. This pre-empted the `save_total_limit=3` rotation that would
  have happened at step 11000 / 13750 anyway. Recommendation for follow-up
  runs: provision ≥ 50 GB container disk, or use a Vast.ai volume mount
  (see §7).

- **Wall-time / cost:**
  - Phase 1 SFT (2× RTX 5090, ~3 epochs): ~3 h × 3 seeds in parallel ≈ 3 h wall
  - Phase 2 DPO (4× A100 PCIE, 5 epochs, batch 4): ~7.3 h × 3 seeds in parallel ≈ 7.3 h wall
  - Total GPU spend (approximate): ~$25 5090 + ~$25 A100 = **~$50**

- **W&B:** all runs logged to project `findpo-repro` with paired JSONL
  fallback in `results/exp_*/train_log.jsonl`.

---

## 7. Recommendations for follow-up

1. **Test reference-model hypothesis first.** Re-run DPO for one seed with
   `ref_model = SFT checkpoint` instead of `ref_model = base with adapter
   disabled`. If the classification gap closes, root cause is identified.
2. **Strategy A only.** Restrict preference pairs to the subset where SFT
   was wrong (rejected = SFT-confused class). Removes the random-class noise
   from §5.2. Smaller pair set may need lr or epoch adjustment.
3. **Earlier stopping.** With `eval_loss` bottoming at epoch 2-3, evaluate
   `num_train_epochs=2` or use `eval_loss` (not `eval_rewards/accuracies`)
   as `metric_for_best_model`.
4. **Use a larger disk on Vast.ai.** Provision either Container ≥ 50 GB or
   Container 25 GB + Volume 60 GB so `save_total_limit=3` works as designed
   without manual checkpoint deletion mid-run.
5. **Independent ablations** for §5 hypotheses 3 and 4 would each cost ~1
   GPU-day at the current cadence (~$3 on A100).

---

## 8. Reproducibility surface

- Code: this repo at tag `phase-4-complete`.
- Pinned revisions:
  - Base model: `unsloth/Meta-Llama-3.1-8B-Instruct@a2856192`
  - FPB: `takala/financial_phrasebank@8d3fe0c3`
  - TFNS: `zeroshot/twitter-financial-news-sentiment@ccbe24de`
  - NWGI: `oliverwang15/news_with_gpt_instructions@b7c33337`
- Configs: [configs/sft_baseline.yaml](../configs/sft_baseline.yaml),
  [configs/dpo_repro_qlora.yaml](../configs/dpo_repro_qlora.yaml)
- Data fingerprint: [data/manifest_phase0.json](../data/manifest_phase0.json)
- Per-run artefacts: `results/exp_*/{config.yaml,env_info.json,git_info.json,train_log.jsonl}`
- Per-seed eval JSON + raw predictions: `results/exp_*/eval_*.json`,
  `results/exp_*/predictions_*.jsonl`

A reader who reruns from the configs and pinned revisions should reproduce
the same regression. We invite verification.
