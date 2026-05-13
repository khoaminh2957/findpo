"""Deterministic seeding (R2).

DPO + large-model training has irreducible non-determinism from CUDA atomics
and mixed precision. The seed pins everything we *can* pin so that variance
across seeds is interpretable.

torch is imported lazily so the rest of the findpo package (in particular
findpo.sanity and findpo.data) can be imported on machines without torch
installed (e.g. a local Windows box that only does scaffolding work).
"""
from __future__ import annotations

import os
import random
import hashlib
from typing import Any

import numpy as np


def set_seed(seed: int) -> None:
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_rng_state() -> dict[str, Any]:
    import torch
    return {
        "torch_cpu": torch.get_rng_state().tolist(),
        "torch_cuda": [s.tolist() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def hash_module_params(module) -> str:
    """SHA256 of concatenated parameter bytes. Used by R8.b (frozen ref model check).

    Handles bfloat16 (which numpy lacks native support for) by casting to fp32
    before serializing.
    """
    import torch
    h = hashlib.sha256()
    for name, p in sorted(module.named_parameters()):
        h.update(name.encode())
        arr = p.detach().cpu()
        if arr.dtype == torch.bfloat16:
            arr = arr.to(torch.float32)
        h.update(arr.contiguous().numpy().tobytes())
    return h.hexdigest()
