"""Tokenizer setup — single source of truth for pad-token + padding-side +
truncation-side + chat-template rendering.

This module exists because three Llama-3 footguns can silently corrupt training:

1. EOS-as-pad pitfall. `pad_token = eos_token` is the default many recipes
   reach for, but when a label-masking collator masks pad positions the
   model never learns to emit EOS → generation never stops. We pin
   `<|finetune_right_pad_id|>` (ID 128004) which exists in the Llama-3
   vocab specifically for this.

2. Double-BOS / zero-BOS asymmetry between SFT and DPO. The Llama-3 chat
   template emits a literal `<|begin_of_text|>` at the start. The HF fast
   tokenizer also has a `TemplateProcessing` post-processor that
   prepends BOS whenever `add_special_tokens=True`.

   - TRL 0.12.2 `SFTTrainer` tokenizes the text field with the tokenizer's
     default `add_special_tokens=True` → BOS added by post-processor.
   - TRL 0.12.2 `DPOTrainer.tokenize_row` for decoder-only models hard-
     codes `add_special_tokens=False` and does NOT manually prepend BOS
     (the BOS-prepend branch is gated on `is_encoder_decoder`, which is
     False for Llama).

   If we leave BOS in the chat-template text AND let SFTTrainer add BOS
   too, SFT sequences get TWO BOS while DPO sequences get ONE — silent
   divergence. If we strip BOS from the text, SFT sequences get ONE and
   DPO sequences get ZERO — different silent divergence.

   The fix that's consistent across BOTH trainers is: **keep BOS in the
   chat-template text** (so DPO's add_special_tokens=False path produces
   single-BOS via the literal) AND **force SFT to also use
   add_special_tokens=False** (so SFT doesn't add a second BOS on top).
   See 01_sft_train.py — passes `dataset_kwargs={"add_special_tokens":
   False}` to SFTConfig. Result: SFT and DPO sequences are bit-identical
   in their prompt prefix, both starting with one BOS at position 0.

   `tok.add_bos_token = False` is a no-op on `PreTrainedTokenizerFast`
   (verified empirically: attribute is set but ignored — BOS insertion
   is hard-coded in the Rust post-processor).

3. Truncation-side default. transformers defaults `truncation_side =
   "right"`. For an instruction-tuning sequence whose response sits at
   the END, right-truncation chops the response — silently breaking
   completion-only masking (template needle disappears) or losing the
   label. Pin `truncation_side = "left"`.
"""
from __future__ import annotations

LLAMA3_PAD_TOKEN = "<|finetune_right_pad_id|>"
LLAMA3_PAD_TOKEN_ID = 128004


def setup_tokenizer(tok, padding_side: str):
    """Set pad_token / pad_token_id / padding_side / truncation_side.

    padding_side ∈ {"left", "right"}:
        right — SFT training, forward-only scoring (last real token sits at
                attention_mask.sum() - 1)
        left  — batched generation / eval (so the last position is at
                seq_len - 1 across the batch and generation continues from
                the end of every row)

    truncation_side is always "left" — see module docstring (footgun #3).
    """
    if padding_side not in ("left", "right"):
        raise ValueError(f"padding_side must be left or right, got {padding_side!r}")
    tok.pad_token = LLAMA3_PAD_TOKEN
    tok.pad_token_id = LLAMA3_PAD_TOKEN_ID
    tok.padding_side = padding_side
    tok.truncation_side = "left"
    return tok


def render_chat(tokenizer, messages, add_generation_prompt: bool = False) -> str:
    """Render a Llama-3 chat template. Returns text INCLUDING the literal
    `<|begin_of_text|>` at position 0.

    Downstream tokenization MUST use `add_special_tokens=False` so that the
    tokenizer does NOT also auto-prepend BOS — otherwise the sequence ends
    up with [BOS, BOS, ...]. The single-BOS comes from the literal string
    in the rendered text. See module docstring (footgun #2).

    All call sites in this repo follow this convention:
        - `01_sft_train.py` passes `dataset_kwargs={"add_special_tokens":
          False}` to SFTConfig.
        - `02_build_preference_pairs.py` and `findpo/evaluate.py` pass
          `add_special_tokens=False` explicitly to `tokenizer(...)`.
        - TRL 0.12.2 `DPOTrainer.tokenize_row` already hard-codes
          `add_special_tokens=False` for decoder-only models.
    """
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
