"""Output directory layout (R10).

results/
  exp_<id>_seed<N>/
    config.yaml
    data_manifest.json
    git_info.json
    env_info.json
    train_log.jsonl
    eval_log.jsonl
    final_metrics.json
    checkpoints/
    model/
    wandb_run_id.txt
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunPaths:
    root: Path
    config_yaml: Path
    data_manifest: Path
    git_info: Path
    env_info: Path
    train_log: Path
    eval_log: Path
    final_metrics: Path
    checkpoints_dir: Path
    model_dir: Path
    wandb_run_id: Path

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)


def run_paths(output_dir_base: str | Path, exp_id: str, seed: int) -> RunPaths:
    root = Path(output_dir_base) / f"exp_{exp_id}_seed{seed}"
    return RunPaths(
        root=root,
        config_yaml=root / "config.yaml",
        data_manifest=root / "data_manifest.json",
        git_info=root / "git_info.json",
        env_info=root / "env_info.json",
        train_log=root / "train_log.jsonl",
        eval_log=root / "eval_log.jsonl",
        final_metrics=root / "final_metrics.json",
        checkpoints_dir=root / "checkpoints",
        model_dir=root / "model",
        wandb_run_id=root / "wandb_run_id.txt",
    )
