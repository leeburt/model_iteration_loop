#!/usr/bin/env python3
"""训练后自动验收入口脚本。

用法: python scripts/run_acceptance.py --config configs/project.yaml [--profile full]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yolo_iter.acceptance import run_acceptance
from yolo_iter.config import acceptance_config_from_project, load_yaml
from yolo_iter.logging_utils import setup_logger


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run YOLO acceptance evaluation.")
    ap.add_argument("--config", default="configs/project.yaml")
    ap.add_argument("--profile", default="full", help="acceptance profile in the unified project config")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    # Create a temporary logger first; run_acceptance owns the real run dir.
    cfg = acceptance_config_from_project(load_yaml(args.config), profile=args.profile)
    preview_dir = Path(cfg.get("output_root", "runs"))
    preview_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("acceptance", preview_dir / "acceptance_latest.log")
    logger.info("Using config: %s", Path(args.config).resolve())
    run_dir = run_acceptance(args.config, profile=args.profile, logger=logger)
    logger.info("Acceptance complete: %s", run_dir)


if __name__ == "__main__":
    main()
