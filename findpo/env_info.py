"""Capture environment + git state (R1, R6).

Writes env_info.json (versions, CUDA, GPU) and git_info.json (commit, dirty flag)
to a run directory. Refuses dirty working tree unless allow_dirty=True.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as e:
        return f"<error: {e}>"


def collect_env_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["torch_cuda_available"] = torch.cuda.is_available()
        info["cuda_runtime"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu_count"] = torch.cuda.device_count()
            info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except ImportError:
        info["torch_version"] = None

    for pkg in ("transformers", "trl", "peft", "accelerate", "bitsandbytes",
                "datasets", "tokenizers", "wandb", "numpy", "pandas"):
        try:
            mod = __import__(pkg)
            info[f"{pkg}_version"] = getattr(mod, "__version__", "unknown")
        except ImportError:
            info[f"{pkg}_version"] = None

    info["nvidia_smi"] = _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                "--format=csv,noheader"])
    info["nvcc_version"] = _run(["nvcc", "--version"])
    return info


def collect_git_info(repo_root: Path) -> dict[str, Any]:
    return {
        "commit": _run(["git", "-C", str(repo_root), "rev-parse", "HEAD"]),
        "branch": _run(["git", "-C", str(repo_root), "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty_files": _run(["git", "-C", str(repo_root), "status", "--porcelain"]),
        "remote": _run(["git", "-C", str(repo_root), "remote", "-v"]),
    }


def assert_clean_tree(repo_root: Path, allow_dirty: bool = False) -> None:
    """Refuse to launch training with uncommitted changes (R6)."""
    dirty = _run(["git", "-C", str(repo_root), "status", "--porcelain"])
    if dirty and not allow_dirty:
        raise RuntimeError(
            f"Working tree is dirty. Commit or pass --allow-dirty.\n{dirty}"
        )


def write_env_and_git(repo_root: Path, env_path: Path, git_path: Path) -> None:
    env_path.write_text(json.dumps(collect_env_info(), indent=2))
    git_path.write_text(json.dumps(collect_git_info(repo_root), indent=2))
