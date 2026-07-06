#!/usr/bin/env python3
"""模型 PK 入口脚本。

用法: python scripts/run_model_pk.py --config configs/project.yaml [--profile default]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yolo_iter.config import load_yaml, model_pk_config_from_project
from yolo_iter.logging_utils import setup_logger
from yolo_iter.model_pk import run_model_pk


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run model PK using champion predictions as pseudo-GT.")
    ap.add_argument("--config", default="configs/project.yaml")
    ap.add_argument("--profile", default="default", help="model_pk profile in the unified project config")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = model_pk_config_from_project(load_yaml(args.config), profile=args.profile)
    preview_dir = Path(cfg.get("output_root", "runs"))
    preview_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("model_pk", preview_dir / "model_pk_latest.log")
    logger.info("Using config: %s", Path(args.config).resolve())
    run_dir = run_model_pk(args.config, profile=args.profile, logger=logger)
    logger.info("Model PK complete: %s", run_dir)


if __name__ == "__main__":
    main()
