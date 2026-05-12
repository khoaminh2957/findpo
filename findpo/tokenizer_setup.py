"""Tokenizer setup — single source of truth for pad-token + padding-side +
truncation-side + chat-template rendering.

This module exists because three Llama-3 footguns can silently corrupt training:

1. EOS-as-pad pitfall. `pad_token = eos_token` is the default many recipes
   reach for, but when a label-masking collator masks pad positions the
   model never learns to emit EOS → generation never stops. We pin
   `<|finetune_right_pad_id|>` (ID 128004) which exists in the Llama-3
   vocab specifically for this.

2. Double-BOS. The Llama-3 chat template renders the literal string
   `<|begin_of_text|>` at the start. The HF fast tokenizer's
   `TemplateProcessing` post-processor ALSO prepends BOS during
   tokenization (whenever `add_special_tokens=True`, which is the
   default for HF Trainer / TRL pipelines). Without intervention every
   sequence ends up with `[BOS, BOS, ...]`. Note: `tok.add_bos_token =
   False` is a no-op on `PreTrainedTokenizerFast` — BOS insertion is
   hard-coded in the Rust post-processor. The reliable fix is to strip
   the literal BOS from the rendered chat-template text before passing
   it to the trainer (`render_chat` below). Empirically verified on
   transformers==4.46.3 + the official Llama-3.1 tokenizer.

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

    truncation_side is always "left" — see module docstring.
    """
    if padding_side not in ("left", "right"):
        raise ValueError(f"padding_side must be left or right, got {padding_side!r}")
    tok.pad_token = LLAMA3_PAD_TOKEN
    tok.pad_token_id = LLAMA3_PAD_TOKEN_ID
    tok.padding_side = padding_side
    tok.truncation_side = "left"
    return tok


def render_chat(tokenizer, messages, add_generation_prompt: bool = False) -> str:
    """Render a chat template and strip the leading BOS string (see module
    docstring, footgun #2). Use this everywhere instead of calling
    `tokenizer.apply_chat_template(..., tokenize=False)` directly.

    Returns text WITHOUT the literal `<|begin_of_text|>` at the start; the
    downstream tokenizer's post-processor will add exactly one BOS token at
    tokenization time, yielding the canonical single-BOS sequence.
    """
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    bos = tokenizer.bos_token
    if bos and text.startswith(bos):
        text = text[len(bos):]
    return text
