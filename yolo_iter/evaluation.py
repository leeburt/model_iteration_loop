"""Evaluation backend selection for pose and detect tasks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .detect_io import read_detect_txt
from .detect_match import (
    DetectMatchConfig,
    build_visualization_plan as detect_build_visualization_plan,
    detect_config_from_dict,
    detect_match,
    evaluate_split as detect_evaluate_split,
    generate_comparison_visualizations as detect_generate_comparison_visualizations,
    predict_to_labels as detect_predict_to_labels,
    save_visualization_sample as detect_save_visualization_sample,
)
from .pose_io import read_pose_txt
from .pose_tiny_match import (
    TinyMatchConfig,
    build_visualization_plan as pose_build_visualization_plan,
    evaluate_split as pose_evaluate_split,
    generate_comparison_visualizations as pose_generate_comparison_visualizations,
    predict_to_labels as pose_predict_to_labels,
    save_visualization_sample as pose_save_visualization_sample,
    tiny_config_from_dict,
    tiny_match,
)


@dataclass(frozen=True)
class EvaluationBackend:
    """Task-specific functions needed by acceptance, recheck, and model PK."""

    task: str
    cfg: Any
    evaluate_split: Callable[..., dict[str, Any]]
    generate_comparison_visualizations: Callable[..., None]
    predict_to_labels: Callable[..., None]
    read_txt: Callable[[Path, int, int, bool], list[Any]]
    match_items: Callable[[list[Any], list[Any], Any], list[Any]]
    build_visualization_plan: Callable[..., dict[str, set[str]]]
    save_visualization_sample: Callable[..., None]
    match_summary: Callable[[Any], dict[str, Any]]


def pose_match_summary(cfg: TinyMatchConfig) -> dict[str, Any]:
    """Return serializable pose matching protocol summary."""
    return {
        "type": "keypoint_or_center_with_padded_iou",
        "kp_px": cfg.kp_px,
        "kp_box_ratio": cfg.kp_box_ratio,
        "center_px": cfg.center_px,
        "center_box_ratio": cfg.center_box_ratio,
        "pad_px": cfg.pad_px,
        "min_padded_iou": cfg.min_padded_iou,
    }


def detect_match_summary(cfg: DetectMatchConfig) -> dict[str, Any]:
    """Return serializable detect matching protocol summary."""
    return {
        "type": "bbox_iou",
        "match_iou": cfg.match_iou,
    }


def pose_backend(cfg: TinyMatchConfig) -> EvaluationBackend:
    """Build a pose evaluation backend."""
    return EvaluationBackend(
        task="pose",
        cfg=cfg,
        evaluate_split=pose_evaluate_split,
        generate_comparison_visualizations=pose_generate_comparison_visualizations,
        predict_to_labels=pose_predict_to_labels,
        read_txt=read_pose_txt,
        match_items=tiny_match,
        build_visualization_plan=pose_build_visualization_plan,
        save_visualization_sample=pose_save_visualization_sample,
        match_summary=pose_match_summary,
    )


def detect_backend(cfg: DetectMatchConfig) -> EvaluationBackend:
    """Build a detect evaluation backend."""
    return EvaluationBackend(
        task="detect",
        cfg=cfg,
        evaluate_split=detect_evaluate_split,
        generate_comparison_visualizations=detect_generate_comparison_visualizations,
        predict_to_labels=detect_predict_to_labels,
        read_txt=read_detect_txt,
        match_items=detect_match,
        build_visualization_plan=detect_build_visualization_plan,
        save_visualization_sample=detect_save_visualization_sample,
        match_summary=detect_match_summary,
    )


def backend_from_eval_protocol(data: dict[str, Any]) -> EvaluationBackend:
    """Select an evaluation backend from eval_protocol."""
    task = str(data.get("task") or "").strip().lower()
    metric = str(data.get("metric") or "").strip().lower()
    if task == "detect" or metric in {"bbox_iou_f1", "detect_f1"}:
        return detect_backend(detect_config_from_dict(data))
    return pose_backend(tiny_config_from_dict(data))


def backend_from_match_config(cfg: Any | None) -> EvaluationBackend:
    """Select a backend from an already-created config object."""
    if isinstance(cfg, DetectMatchConfig):
        return detect_backend(cfg)
    if isinstance(cfg, TinyMatchConfig):
        return pose_backend(cfg)
    if cfg is None:
        return pose_backend(TinyMatchConfig(show_progress=False))
    if hasattr(cfg, "match_iou"):
        return detect_backend(cfg)
    return pose_backend(cfg)
