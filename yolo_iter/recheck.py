"""基于缓存预测结果的复核评估。

人工修正标签后，复用已缓存的模型预测结果重新计算指标，
无需重新运行模型推理。
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import acceptance_config_from_project, load_yaml
from .manifest import build_dataset_manifest, write_json
from .acceptance import resolve_model_paths, write_comparison_visualizations
from .pose_tiny_match import evaluate_split, tiny_config_from_dict


def find_pred_dir(run_dir: Path, model_role: str, dataset_name: str, split: str) -> Path:
    """定位缓存预测标签目录，不存在则抛异常。"""
    pred_dir = run_dir / "cache" / model_role / dataset_name / split / "_pred_labels"
    if not pred_dir.is_dir():
        raise FileNotFoundError(f"Prediction cache missing: {pred_dir}")
    return pred_dir


def run_review_recheck(
    run_dir: str | Path,
    config_path: str | Path,
    review_name: str,
    profile: str = "full",
    logger=None,
) -> Path:
    """使用原始验收的缓存预测，对修正后的标签重新评估。

    Args:
        run_dir: 原始 acceptance 输出目录。
        config_path: project.yaml 路径。
        review_name: 本次复核的名称（用于输出子目录命名）。
        profile: acceptance profile 名。

    Returns:
        review 输出目录路径。
    """
    run_path = Path(run_dir).expanduser().resolve()
    cfg = acceptance_config_from_project(load_yaml(config_path), profile=profile)
    review_safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in review_name)
    review_dir = run_path / "review" / f"{review_safe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    review_dir.mkdir(parents=True, exist_ok=False)
    if logger:
        logger.info("Review recheck dir: %s", review_dir)

    manifests = []
    for ds in cfg.get("eval_datasets", []):
        manifest = build_dataset_manifest(ds["data"], ds.get("name"))
        manifests.append(manifest)
        write_json(review_dir / "manifests" / f"dataset_manifest_{manifest['dataset_name']}.json", manifest)

    tiny_cfg = tiny_config_from_dict(cfg.get("eval", {}))
    roles = list(resolve_model_paths(cfg).keys())
    if not roles:
        # Review can still re-evaluate existing candidate cache even if current config no longer carries model path.
        if (run_path / "cache" / "candidate").is_dir():
            roles.append("candidate")
        if (run_path / "cache" / "champion").is_dir():
            roles.append("champion")

    results: list[dict[str, Any]] = []
    for role in roles:
        for ds in cfg.get("eval_datasets", []):
            dataset_name = ds["name"]
            data_path = Path(ds["data"]).expanduser().resolve()
            for split in ds.get("splits", ["val"]):
                pred_dir = find_pred_dir(run_path, role, dataset_name, split)
                if logger:
                    logger.info("Rechecking role=%s dataset=%s split=%s", role, dataset_name, split)
                result = evaluate_split(
                    data_path=data_path,
                    dataset_name=dataset_name,
                    split=split,
                    output_dir=review_dir,
                    cfg=tiny_cfg,
                    pred_label_dirs=[pred_dir],
                    model_role=role,
                )
                results.append(result)

    rows = []
    for item in results:
        s = item["summary"]
        totals = s["totals"]
        metrics = s["metrics"]
        rows.append(
            {
                "model_role": s["model_role"],
                "dataset": s["dataset"],
                "split": s["split"],
                "images": totals["images"],
                "gt": totals["gt"],
                "pred": totals["pred"],
                "tp": totals["tp"],
                "fp": totals["fp"],
                "fn": totals["fn"],
                "duplicate_fp": totals["duplicate_fp"],
                "real_fp": totals["real_fp"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
            }
        )
    write_comparison_visualizations(review_dir, results, save_visualizations=bool(tiny_cfg.save_diff), cfg=tiny_cfg)
    write_json(review_dir / "metrics" / "overall_summary.json", rows)
    (review_dir / "metrics").mkdir(parents=True, exist_ok=True)
    with (review_dir / "metrics" / "overall_summary.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys()) if rows else ["model_role", "dataset", "split"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return review_dir
