"""Deterministic seeding (R2).

DPO + large-model training has irreducible non-determinism from CUDA atomics
and mixed precision. The seed pins everything we *can* pin so that variance
across seeds is interpretable.
"""
from __future__ import annotations

import os
import random
import hashlib
import json
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_rng_state() -> dict[str, Any]:
    return {
        "torch_cpu": torch.get_rng_state().tolist(),
        "torch_cuda": [s.tolist() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def hash_module_params(module: torch.nn.Module) -> str:
    """SHA256 of concatenated parameter bytes. Used by R8.b (frozen ref model check)."""
    h = hashlib.sha256()
    for name, p in sorted(module.named_parameters()):
        h.update(name.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
