"""Phase 0 smoke test (run on REMOTE_GPU).

Verifies:
    * GPU is visible and bf16-capable
    * pinned library versions match requirements.txt
    * HF Hub auth works (Llama-3.1-8B-Instruct is gated)
    * tokenizer loads and chat template renders with Llama-3 tokens (R8.e)
    * model loads in 4-bit and a single forward pass returns sane logits
    * W&B login works (or warn and continue)

Exit code 0 = all checks pass. Non-zero = phase 0 blocked.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from findpo.env_info import collect_env_info  # noqa: E402
from findpo.labels import format_prompt_messages  # noqa: E402
from findpo.sanity import check_tokenizer_alignment  # noqa: E402


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--skip-model", action="store_true",
                        help="Skip model load (use for non-GPU dry runs).")
    parser.add_argument("--skip-wandb", action="store_true")
    args = parser.parse_args()

    failures: list[str] = []

    section("Environment")
    env = collect_env_info()
    print(json.dumps({k: env[k] for k in ("python_version", "torch_version",
                                           "torch_cuda_available", "cuda_runtime",
                                           "gpu_count", "transformers_version",
                                           "trl_version", "peft_version",
                                           "bitsandbytes_version")
                      if k in env}, indent=2))
    if not env.get("torch_cuda_available"):
        failures.append("CUDA not available")
    if env.get("gpu_count", 0) < 1:
        failures.append("No GPU detected")

    section("BF16 capability")
    try:
        import torch
        bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        print(f"bf16_supported = {bf16_ok}")
        if not bf16_ok:
            failures.append("BF16 not supported on this GPU")
    except Exception as e:
        failures.append(f"bf16 check error: {e}")

    section("HF Hub auth")
    try:
        from huggingface_hub import whoami
        info = whoami()
        print(f"HF user: {info.get('name')}, type: {info.get('type')}")
    except Exception as e:
        failures.append(f"HF auth failed (Llama-3 is gated): {e}")

    section("Tokenizer + chat template (R8.e)")
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        ok, details = check_tokenizer_alignment(
            tok,
            system="You are a financial sentiment classifier.",
            user_text="The company reported record profits.",
        )
        print(json.dumps(details, indent=2))
        if not ok:
            failures.append(f"Tokenizer alignment failed: {details['missing']}")
    except Exception as e:
        failures.append(f"Tokenizer load error: {e}")

    if not args.skip_model:
        section("Model load (4-bit) + forward pass")
        try:
            import torch
            from transformers import AutoModelForCausalLM, BitsAndBytesConfig
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                quantization_config=bnb,
                device_map="auto",
                attn_implementation="flash_attention_2",
            )
            tok = AutoTokenizer.from_pretrained(args.model)
            msgs = format_prompt_messages(
                system="You are a financial sentiment classifier.",
                user_text="The company reported record profits.",
            )
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            ids = tok(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model(**ids)
            print(f"logits shape: {tuple(out.logits.shape)}, dtype: {out.logits.dtype}")
            assert out.logits.shape[-1] == model.config.vocab_size
        except Exception as e:
            failures.append(f"Model load / forward failed: {e}")
    else:
        section("Model load (SKIPPED)")

    if not args.skip_wandb:
        section("W&B")
        try:
            import wandb
            api = wandb.Api()
            _ = api.viewer
            print(f"W&B user: {api.viewer.get('username')}")
        except Exception as e:
            print(f"[WARN] W&B login not set up: {e}")
            print("       Run `wandb login` on this machine.")

    section("Summary")
    if failures:
        print("FAIL — issues found:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("PASS — Phase 0 environment ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
