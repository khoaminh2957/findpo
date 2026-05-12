"""Tokenizer setup — single source of truth for pad-token + padding-side.

Llama-3 vocab includes a dedicated `<|finetune_right_pad_id|>` (ID 128004).
Using it as pad_token avoids the EOS-masking pitfall (when pad==eos and the
collator masks pad → model never learns to emit EOS → generation fails to
stop). See:
  https://discuss.huggingface.co/t/how-to-set-the-pad-token-for-meta-llama-llama-3-models/103418
"""
from __future__ import annotations

LLAMA3_PAD_TOKEN = "<|finetune_right_pad_id|>"
LLAMA3_PAD_TOKEN_ID = 128004


def setup_tokenizer(tok, padding_side: str):
    """Set pad_token / pad_token_id / padding_side. Returns the tokenizer.

    padding_side ∈ {"left", "right"}:
        right — SFT training, forward-only scoring (next-token logits sit at
                attention_mask.sum() - 1)
        left  — batched generation / eval (so the last generated position
                is at seq_len - 1 across the batch)
    """
    if padding_side not in ("left", "right"):
        raise ValueError(f"padding_side must be left or right, got {padding_side!r}")
    tok.pad_token = LLAMA3_PAD_TOKEN
    tok.pad_token_id = LLAMA3_PAD_TOKEN_ID
    tok.padding_side = padding_side
    return tok
