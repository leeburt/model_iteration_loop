#!/usr/bin/env python3
"""Train a YOLO detect model with the shared training wrapper."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yolo_iter.logging_utils import setup_logger
from yolo_iter.manifest import build_dataset_manifest, write_json
from yolo_iter.training import build_training_args, train_yolo


DEFAULT_MODEL = "/data/liuyuhao/junction_detection/runs/all_junctions/junction_v16_v9_filtered/weights/best.pt"
DEFAULT_DATA =""
DEFAULT_PROJECT = "./train_resule"
DEFAULT_NAME = "junction_v17_xuxian_yinying"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO detect model.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Initial detect weights used when no last.pt is found.")
    parser.add_argument("--data", default=DEFAULT_DATA, help="YOLO detect data.yaml path.")
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="Ultralytics project output directory.")
    parser.add_argument("--name", default=DEFAULT_NAME, help="Ultralytics run name.")
    parser.add_argument("--resume", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="5")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr0", type=float, default=1e-4)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--log-dir", default="runs/train_logs")
    parser.add_argument("--dry-run", action="store_true", help="Build config and model object without starting training.")
    parser.add_argument("--skip-manifest", action="store_true", help="Skip dataset manifest generation.")
    parser.add_argument("--no-cache", action="store_true", help="Disable Ultralytics dataset cache.")
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP training.")
    return parser.parse_args()


def resume_value(raw: str) -> str | bool:
    if raw == "true":
        return True
    if raw == "false":
        return False
    return "auto"


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    """Build the train config consumed by yolo_iter.training."""
    return {
        "data": str(Path(args.data).expanduser()),
        "initial_weights": str(Path(args.model).expanduser()),
        "project": str(Path(args.project).expanduser()),
        "name": args.name,
        "resume": resume_value(args.resume),
        "args": {
            "task": "detect",
            "mode": "train",
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": str(args.device),
            "workers": args.workers,
            "val": True,
            "optimizer": args.optimizer,
            "lr0": args.lr0,
            "lrf": args.lrf,
            "patience": args.patience,
            "close_mosaic": args.close_mosaic,
            "amp": not args.no_amp,
            "plots": True,
            "save": True,
            "exist_ok": True,
            "cache": not args.no_cache,
        },
    }


def validate_paths(config: dict[str, Any]) -> None:
    """Fail fast for missing detect data or initial weights."""
    data = Path(config["data"]).expanduser()
    weights = Path(config["initial_weights"]).expanduser()
    if not data.is_file():
        raise FileNotFoundError(f"data.yaml does not exist: {data}")
    if not weights.is_file():
        raise FileNotFoundError(f"initial weights do not exist: {weights}")


def main() -> None:
    args = parse_args()
    config = build_config(args)
    validate_paths(config)

    log_file = Path(args.log_dir) / f"{config['name']}.log"
    logger = setup_logger("train", log_file)
    logger.info("Using detect train config: %s", json.dumps(config, ensure_ascii=False))

    if not args.skip_manifest:
        manifest = build_dataset_manifest(config["data"], dataset_name="train_data")
        manifest_path = Path(args.log_dir) / f"{config['name']}_dataset_manifest.json"
        write_json(manifest_path, manifest)
        logger.info("Dataset manifest written: %s", manifest_path)
        logger.info(
            "Dataset summary: %s",
            json.dumps(
                {
                    "dataset_version": manifest["dataset_version"],
                    "image_count": manifest["image_count"],
                    "object_count": manifest["object_count"],
                    "leakage_count": len(manifest.get("leakage", [])),
                },
                ensure_ascii=False,
            ),
        )

    if args.dry_run:
        _model, train_args, metadata = build_training_args(config)
        print(json.dumps({"dry_run": True, "metadata": metadata, "train_args": train_args}, indent=2, ensure_ascii=False))
        return

    _results, metadata = train_yolo(config, logger=logger)
    print("\nTraining done!")
    print(f"Task: {metadata['task']}")
    print(f"Best weights: {Path(config['project']) / config['name'] / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
