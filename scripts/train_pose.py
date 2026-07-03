#!/usr/bin/env python3
"""YOLO pose 训练入口脚本。

用法: python scripts/train_pose.py --config configs/project.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yolo_iter.config import load_yaml, train_config_from_project
from yolo_iter.logging_utils import setup_logger
from yolo_iter.manifest import build_dataset_manifest, write_json
from yolo_iter.training import train_pose


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train YOLO pose model from YAML config.")
    ap.add_argument("--config", default="configs/project.yaml")
    ap.add_argument("--log-dir", default="runs/train_logs")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = train_config_from_project(load_yaml(args.config))
    run_name = str(cfg.get("name", "train_pose"))
    log_file = Path(args.log_dir) / f"{run_name}.log"
    logger = setup_logger("train_pose", log_file)
    logger.info("Using config: %s", Path(args.config).resolve())

    manifest = build_dataset_manifest(cfg["data"], dataset_name="train_data")
    manifest_path = Path(args.log_dir) / f"{run_name}_dataset_manifest.json"
    write_json(manifest_path, manifest)
    logger.info("Dataset manifest written: %s", manifest_path)
    logger.info("Dataset summary: %s", json.dumps({
        "dataset_version": manifest["dataset_version"],
        "image_count": manifest["image_count"],
        "object_count": manifest["object_count"],
        "leakage_count": len(manifest.get("leakage", [])),
    }, ensure_ascii=False))

    train_pose(cfg, logger=logger)


if __name__ == "__main__":
    main()
