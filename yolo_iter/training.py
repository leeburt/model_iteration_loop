"""YOLO 模型训练封装。

支持从 YAML 配置驱动训练，自动处理：
- 断点续训检测（resume: auto）
- 训练参数组装
- 训练元数据记录
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO

from .manifest import file_sha256

os.environ["OPENCV_IO_IGNORE_WARNING"] = "1"
warnings.filterwarnings("ignore", ".*iCCP.*")
warnings.filterwarnings("ignore", category=UserWarning)


def get_resume_checkpoint(project_name: str, name: str, task: str = "pose") -> Path | None:
    """查找可续训的 last.pt 检查点，找不到返回 None。"""
    task = str(task or "pose")
    candidates = [
        Path(project_name) / name / "weights" / "last.pt",
        Path("runs") / task / project_name / name / "weights" / "last.pt",
    ]
    for path in candidates:
        if path.exists():
            return path.resolve()
    return None


def build_training_args(config: dict[str, Any]) -> tuple[YOLO, dict[str, Any], dict[str, Any]]:
    """根据配置构建 YOLO 模型和训练参数。

    resume 模式说明：
    - "auto": 自动查找 last.pt，找到则续训，否则从头开始
    - True: 强制续训（ultralytics 从 run 目录找 last.pt）
    - 其他: 从头开始

    Returns:
        (YOLO model, train_args dict, metadata dict)
    """
    data_path = Path(config["data"]).expanduser().resolve()
    project = str(config["project"])
    name = str(config["name"])
    args = dict(config.get("args") or {})
    args["data"] = str(data_path)
    args["project"] = project
    args["name"] = name

    resume_mode = config.get("resume", "auto")
    task = str(args.get("task") or "pose")
    resume_checkpoint = get_resume_checkpoint(project, name, task=task) if resume_mode == "auto" else None
    if resume_checkpoint:
        model_path = resume_checkpoint
        args["resume"] = True
    else:
        model_path = Path(config["initial_weights"]).expanduser().resolve()
        args["resume"] = bool(config.get("resume") is True)

    model = YOLO(str(model_path))
    metadata = {
        "data": str(data_path),
        "project": project,
        "name": name,
        "model_path": str(model_path),
        "model_hash": file_sha256(model_path),
        "resume": bool(args.get("resume")),
        "task": task,
        "cuda_device_count": torch.cuda.device_count(),
    }
    return model, args, metadata


def train_yolo(config: dict[str, Any], logger=None):
    """执行 YOLO 训练，返回 (results, metadata)。"""
    model, args, metadata = build_training_args(config)
    if logger:
        logger.info("Starting YOLO %s training", metadata["task"])
        logger.info("Training metadata: %s", metadata)
        logger.info("Training args: %s", args)
    results = model.train(**args)
    save_dir = getattr(results, "save_dir", None) if results is not None else None
    if save_dir is None:
        save_dir = Path(str(args["project"])) / str(args["name"])
    if logger:
        logger.info("Training complete. save_dir=%s", save_dir)
    return results, metadata


def train_pose(config: dict[str, Any], logger=None):
    """Backward-compatible wrapper for older pose training scripts."""
    return train_yolo(config, logger=logger)
