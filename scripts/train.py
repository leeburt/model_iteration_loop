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

from yolo_iter.config import load_yaml, train_config_from_project
from yolo_iter.logging_utils import setup_logger
from yolo_iter.manifest import build_dataset_manifest, write_json
from yolo_iter.training import build_training_args, train_yolo


DEFAULT_MODEL = "/data/liuyuhao/junction_detection/runs/all_junctions/junction_v16_v9_filtered/weights/best.pt"
DEFAULT_DATA =""
DEFAULT_PROJECT = "./train_resule"
DEFAULT_NAME = "junction_v17_xuxian_yinying"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO detect model.")
    parser.add_argument("--config", default=None, help="Project YAML config; uses its train section when provided.")
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


def merge_train_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Apply explicit CLI train overrides on top of YAML config."""
    merged = dict(config)
    train_args = dict(merged.get("args") or {})
    merged["args"] = train_args
    if args.model != DEFAULT_MODEL:
        merged["initial_weights"] = str(Path(args.model).expanduser())
    if args.data != DEFAULT_DATA:
        merged["data"] = str(Path(args.data).expanduser())
    if args.project != DEFAULT_PROJECT:
        merged["project"] = str(Path(args.project).expanduser())
    if args.name != DEFAULT_NAME:
        merged["name"] = args.name
    if args.resume != "auto":
        merged["resume"] = resume_value(args.resume)

    overrides = {
        "epochs": args.epochs if args.epochs != 20 else None,
        "imgsz": args.imgsz if args.imgsz != 1280 else None,
        "batch": args.batch if args.batch != 4 else None,
        "device": str(args.device) if str(args.device) != "5" else None,
        "workers": args.workers if args.workers != 8 else None,
        "patience": args.patience if args.patience != 15 else None,
        "optimizer": args.optimizer if args.optimizer != "AdamW" else None,
        "lr0": args.lr0 if args.lr0 != 1e-4 else None,
        "lrf": args.lrf if args.lrf != 0.01 else None,
        "close_mosaic": args.close_mosaic if args.close_mosaic != 10 else None,
    }
    for key, value in overrides.items():
        if value is not None:
            train_args[key] = value
    if args.no_amp:
        train_args["amp"] = False
    if args.no_cache:
        train_args["cache"] = False
    return merged


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    """Build the train config consumed by yolo_iter.training."""
    if args.config:
        config = train_config_from_project(load_yaml(args.config))
        return merge_train_overrides(config, args)
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
