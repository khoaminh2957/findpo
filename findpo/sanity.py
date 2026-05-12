"""R8 sanity tests — required to pass before Phase 2 (DPO).

Each function returns (passed: bool, details: dict). Caller decides whether
a failure is a hard stop. The R8 contract is that the canonical pipeline
runs every check before launching the real training.
"""
from __future__ import annotations

from typing import Any, Callable

from datasets import Dataset

from .data import assert_no_overlap
from .labels import LABELS, class_distribution
from .seeding import hash_module_params


def check_train_test_no_overlap(train: Dataset, test: Dataset) -> tuple[bool, dict[str, Any]]:
    try:
        assert_no_overlap(train, test)
        return True, {"train_n": len(train), "test_n": len(test)}
    except AssertionError as e:
        return False, {"error": str(e)}


def check_class_balance(ds: Dataset, name: str, min_per_class: int = 20) -> tuple[bool, dict[str, Any]]:
    dist = class_distribution(ds["label"])
    ok = all(c >= min_per_class for c in dist.values())
    return ok, {"name": name, "distribution": dist, "min_per_class": min_per_class}


def check_preference_pair_format(pairs: list[dict], n_print: int = 10) -> tuple[bool, dict[str, Any]]:
    """R8.c — print first n_print pairs, assert chosen != rejected, prompt non-empty."""
    issues: list[str] = []
    for i, p in enumerate(pairs[:n_print]):
        if "prompt" not in p or not p["prompt"]:
            issues.append(f"row {i}: empty prompt")
        if p.get("chosen") == p.get("rejected"):
            issues.append(f"row {i}: chosen == rejected ({p.get('chosen')!r})")
        if p.get("chosen") not in LABELS:
            issues.append(f"row {i}: chosen not in {LABELS}: {p.get('chosen')!r}")
        if p.get("rejected") not in LABELS:
            issues.append(f"row {i}: rejected not in {LABELS}: {p.get('rejected')!r}")
    return (not issues), {"sample": pairs[:n_print], "issues": issues}


def check_ref_model_frozen(initial_hash: str, current_module) -> tuple[bool, dict[str, Any]]:
    """R8.b — after N DPO steps, ref model bytes must be unchanged."""
    now = hash_module_params(current_module)
    ok = now == initial_hash
    return ok, {"initial_sha256": initial_hash, "current_sha256": now}


def overfit_test(train_step_fn: Callable[[int], float], batch_size: int = 32,
                 num_steps: int = 200, target_loss: float = 0.05) -> tuple[bool, dict[str, Any]]:
    """R8.a — train 32 random pairs for 200 steps, loss must drop near zero.

    train_step_fn(step) -> loss; caller wires it to a stripped-down trainer
    with no regularization. If loss does not drop below `target_loss` the
    DPO setup is broken — do NOT proceed to real training.
    """
    losses: list[float] = []
    for step in range(num_steps):
        losses.append(train_step_fn(step))
    final = sum(losses[-10:]) / 10
    ok = final < target_loss
    return ok, {
        "batch_size": batch_size,
        "num_steps": num_steps,
        "first_10_mean": sum(losses[:10]) / 10,
        "last_10_mean": final,
        "target_loss": target_loss,
    }


def check_tokenizer_alignment(tokenizer, system: str, user_text: str,
                              expected_substrings: tuple[str, ...] = (
                                  "<|begin_of_text|>",
                                  "<|start_header_id|>system<|end_header_id|>",
                                  "<|start_header_id|>user<|end_header_id|>",
                              )) -> tuple[bool, dict[str, Any]]:
    """R8.e — applied chat template must match Llama-3 format end-to-end."""
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_text},
    ]
    rendered = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    missing = [s for s in expected_substrings if s not in rendered]
    return (not missing), {"rendered_head": rendered[:300], "missing": missing}
