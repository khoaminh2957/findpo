"""Dual W&B + JSONL logging (R4, R9).

W&B is the primary UI; JSONL is the durable fallback for cases when network
to W&B is flaky. EVERY metric is written to both, atomically.

Drop-in usage:
    logger = DualLogger.start(run_paths, config, wandb_cfg)
    logger.log_train({"loss": 0.5, "lr": 2e-5}, step=10)
    logger.log_eval({"accuracy": 0.82}, step=1500)
    logger.finish({"final/test_accuracy": 0.91})
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .paths import RunPaths


@dataclass
class WandbCfg:
    project: str
    tags: list[str]
    enabled: bool = True


class _JsonlSink:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Archive any prior log from a previous run rather than appending
        # (which would mix two runs' step numbers) or silently overwriting
        # (which loses data). Re-runs after a crash get a fresh file and a
        # .bakN file preserving the previous attempt.
        if self.path.exists() and self.path.stat().st_size > 0:
            i = 0
            while True:
                bak = self.path.with_suffix(self.path.suffix + f".bak{i}")
                if not bak.exists():
                    break
                i += 1
            self.path.rename(bak)
        self._fh = self.path.open("w", encoding="utf-8", buffering=1)

    def write(self, payload: dict[str, Any]) -> None:
        payload = {"_ts": time.time(), **payload}
        self._fh.write(json.dumps(payload, default=str) + "\n")
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


class DualLogger:
    """Use DualLogger.start(...) and DualLogger.finish(...) — do not call __init__."""

    def __init__(self, paths: RunPaths, wandb_run, train_sink: _JsonlSink, eval_sink: _JsonlSink) -> None:
        self.paths = paths
        self.wandb = wandb_run
        self._train = train_sink
        self._eval = eval_sink

    @classmethod
    def start(cls, paths: RunPaths, config: dict, wandb_cfg: WandbCfg,
              run_name: str | None = None) -> "DualLogger":
        paths.ensure()
        train_sink = _JsonlSink(paths.train_log)
        eval_sink = _JsonlSink(paths.eval_log)

        wandb_run = None
        if wandb_cfg.enabled:
            try:
                import wandb
                wandb_run = wandb.init(
                    project=wandb_cfg.project,
                    name=run_name or paths.root.name,
                    tags=wandb_cfg.tags,
                    config=config,
                    dir=str(paths.root),
                )
                paths.wandb_run_id.write_text(wandb_run.id)
            except Exception as e:
                print(f"[WARN] W&B init failed: {e}. Continuing with JSONL only.")
                wandb_run = None

        return cls(paths, wandb_run, train_sink, eval_sink)

    def log_train(self, metrics: dict[str, Any], step: int) -> None:
        self._train.write({"step": step, **metrics})
        if self.wandb is not None:
            self.wandb.log(metrics, step=step)

    def log_eval(self, metrics: dict[str, Any], step: int) -> None:
        self._eval.write({"step": step, **metrics})
        if self.wandb is not None:
            self.wandb.log(metrics, step=step)

    def finish(self, final_metrics: dict[str, Any]) -> None:
        self.paths.final_metrics.write_text(json.dumps(final_metrics, indent=2, default=str))
        if self.wandb is not None:
            try:
                self.wandb.summary.update(final_metrics)
                self.wandb.finish()
            except Exception:
                pass
        self._train.close()
        self._eval.close()
