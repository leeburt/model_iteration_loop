"""Candidate 与 Champion 的模型 PK 流程。

Model PK 不读取人工 GT 作为评价基准，而是把 Champion 预测作为 pseudo-GT，
计算 Candidate 相对 Champion 的 TP/FP/FN，并输出与 acceptance 类似的指标和可视化。
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .acceptance import default_candidate_from_train, make_run_dir, write_run_manifest
from .config import load_yaml, model_pk_config_from_project
from .detect_match import normalize_class_names
from .evaluation import EvaluationBackend, backend_from_eval_protocol, backend_from_match_config
from .logging_utils import add_file_handler
from .manifest import build_dataset_manifest, write_json
from .paths import collect_images, resolve_dataset_paths
from .pose_io import image_size
from .pose_tiny_match import maybe_progress, precision_recall


@dataclass(frozen=True)
class ModelClassContext:
    """Class-name based mapping used to compare two models with different class ids."""

    class_names: dict[int, str]
    candidate_id_to_unified_id: dict[int, int]
    champion_id_to_unified_id: dict[int, int]


def load_yolo_model_names(model_path: Path) -> dict[int, str]:
    """Read class names from a YOLO model without relying on dataset yaml."""
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    return normalize_class_names(getattr(model, "names", None))


def build_model_class_context(
    *,
    configured_names: dict[int, str] | None,
    candidate_model_names: dict[int, str] | None,
    champion_model_names: dict[int, str] | None,
) -> ModelClassContext:
    """Build unified class ids. Explicit configured names keep existing label ids."""
    configured = normalize_class_names(configured_names)
    if configured:
        identity = {idx: idx for idx in configured}
        return ModelClassContext(configured, identity, identity)

    name_to_unified_id: dict[str, int] = {}
    class_names: dict[int, str] = {}

    def register(name: str) -> int:
        if name not in name_to_unified_id:
            unified_id = len(name_to_unified_id)
            name_to_unified_id[name] = unified_id
            class_names[unified_id] = name
        return name_to_unified_id[name]

    champion_names = normalize_class_names(champion_model_names)
    candidate_names = normalize_class_names(candidate_model_names)
    champion_map = {cls_id: register(name) for cls_id, name in sorted(champion_names.items())}
    candidate_map = {cls_id: register(name) for cls_id, name in sorted(candidate_names.items())}
    return ModelClassContext(class_names, candidate_map, champion_map)


def remap_items_to_class_context(items: list[Any], id_map: dict[int, int]) -> list[Any]:
    """Return items with cls_id remapped to unified ids; unknown ids stay unchanged."""
    if not id_map:
        return items
    return [replace(item, cls_id=id_map.get(item.cls_id, item.cls_id)) for item in items]


def collect_image_source(source: str | Path, dataset_name: str) -> list[Path]:
    """Collect a single image or one-level image directory for Model PK."""
    src = Path(source).expanduser()
    if src.is_file():
        if src.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            raise ValueError(f"images is not a supported image: {src}")
        return [src.resolve()]
    if src.is_dir():
        images = collect_images(src)
        if not images:
            raise ValueError(f"No images found for dataset={dataset_name} images={src}")
        return [p.resolve() for p in images]
    raise FileNotFoundError(f"images does not exist: {src}")


def collect_model_pk_sources(ds: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize one model_pk eval_datasets entry into image sources."""
    dataset_name = str(ds.get("name") or "model_pk_images")
    if ds.get("images"):
        return [
            {
                "dataset_name": dataset_name,
                "split": str(ds.get("split") or "images"),
                "images": collect_image_source(ds["images"], dataset_name),
                "data_path": None,
                "class_names": normalize_class_names(ds.get("names")),
            }
        ]

    if not ds.get("data"):
        raise ValueError(f"model_pk eval dataset '{dataset_name}' must set either images or data")
    data_path = Path(ds["data"]).expanduser().resolve()
    data_names = normalize_class_names(load_yaml(data_path).get("names"))
    sources: list[dict[str, Any]] = []
    for split in ds.get("splits", ["val"]):
        image_label_pairs = collect_split_image_label_pairs(data_path, split, dataset_name)
        sources.append(
            {
                "dataset_name": dataset_name,
                "split": split,
                "images": [item[0] for item in image_label_pairs],
                "data_path": data_path,
                "class_names": data_names,
            }
        )
    return sources


def collect_split_image_label_pairs(data_path: Path, split: str, dataset_name: str) -> list[tuple[Path, Path]]:
    """收集 data.yaml 指定 split 下的图片和人工标签路径。

    Model PK 不使用人工标签参与指标计算，但保留 label path 用于目录兼容和可视化 GT 栏位。
    """
    data = load_yaml(data_path)
    pairs = resolve_dataset_paths(data, split)
    if not pairs:
        raise ValueError(f"No split '{split}' found in {data_path}")
    image_label_pairs: list[tuple[Path, Path]] = []
    for image_dir, label_dir in pairs:
        for img_path in collect_images(image_dir):
            image_label_pairs.append((img_path, label_dir / f"{img_path.stem}.txt"))
    if not image_label_pairs:
        raise ValueError(f"No images found for dataset={dataset_name} split={split} data={data_path}")
    return image_label_pairs


def run_model_pk_from_prediction_dirs(
    *,
    data_path: Path | None,
    dataset_name: str,
    split: str,
    images: list[Path] | None = None,
    class_names: dict[int, str] | None = None,
    candidate_model_names: dict[int, str] | None = None,
    champion_model_names: dict[int, str] | None = None,
    output_dir: Path,
    candidate_pred_dir: Path,
    champion_pred_dir: Path,
    cfg: Any | None = None,
    save_visualizations: bool = True,
) -> dict[str, Any]:
    """使用已有预测标签执行 Model PK。

    Champion 预测作为 pseudo-GT，Candidate 预测作为待评估结果。
    """
    backend = backend_from_match_config(cfg)
    cfg = backend.cfg
    class_context = build_model_class_context(
        configured_names=class_names,
        candidate_model_names=candidate_model_names,
        champion_model_names=champion_model_names,
    )
    visual_class_names = class_context.class_names or normalize_class_names(class_names)
    human_gt_by_image: dict[str, Path] = {}
    if images is None:
        if data_path is None:
            raise ValueError("data_path or images is required")
        image_label_pairs = collect_split_image_label_pairs(Path(data_path), split, dataset_name)
        human_gt_by_image = {str(item[0].resolve()): item[1] for item in image_label_pairs}
        images = [item[0] for item in image_label_pairs]
    else:
        images = [Path(p).expanduser().resolve() for p in images]
    totals = {
        "images": 0,
        "champion_pred_as_gt": 0,
        "candidate_pred": 0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
    }
    rows: list[dict[str, Any]] = []

    for img_path in maybe_progress(
        images,
        enabled=cfg.show_progress,
        desc=f"model_pk {dataset_name} {split} match",
        total=len(images),
    ):
        gt_label = human_gt_by_image.get(str(img_path))
        w, h = image_size(img_path)
        champion_label = Path(champion_pred_dir) / f"{img_path.stem}.txt"
        candidate_label = Path(candidate_pred_dir) / f"{img_path.stem}.txt"
        champion_items = backend.read_txt(champion_label, w, h, True)
        candidate_items = backend.read_txt(candidate_label, w, h, True)
        champion_items = remap_items_to_class_context(champion_items, class_context.champion_id_to_unified_id)
        candidate_items = remap_items_to_class_context(candidate_items, class_context.candidate_id_to_unified_id)
        matches = backend.match_items(champion_items, candidate_items, cfg)
        matched_champion = {m.gt_idx for m in matches}
        matched_candidate = {m.pred_idx for m in matches}
        fp = len(candidate_items) - len(matched_candidate)
        fn = len(champion_items) - len(matched_champion)
        tp = len(matches)

        totals["images"] += 1
        totals["champion_pred_as_gt"] += len(champion_items)
        totals["candidate_pred"] += len(candidate_items)
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn

        rows.append(
            {
                "dataset": dataset_name,
                "split": split,
                "image": str(img_path),
                "data_path": str(data_path) if data_path is not None else None,
                "class_names": visual_class_names,
                "candidate_class_id_map": class_context.candidate_id_to_unified_id,
                "champion_class_id_map": class_context.champion_id_to_unified_id,
                "gt_label": str(champion_label),
                "human_gt_label": str(gt_label) if gt_label is not None else "",
                "pred_label": str(candidate_label),
                "champion_pred_label": str(champion_label),
                "gt": len(champion_items),
                "pred": len(candidate_items),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "matches": [m.__dict__ for m in matches],
            }
        )

    metrics = precision_recall(totals["tp"], totals["fp"], totals["fn"])
    summary = {
        "dataset": dataset_name,
        "data": str(Path(data_path).resolve()) if data_path is not None else None,
        "source": "images" if data_path is None else "data_yaml",
        "class_names": visual_class_names,
        "split": split,
        "mode": "model_pk",
        "gt_source": "champion_predictions",
        "match": backend.match_summary(cfg),
        "totals": totals,
        "metrics": metrics,
    }
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    stem = f"model_pk_{dataset_name}_{split}"
    (metrics_dir / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (metrics_dir / f"{stem}_per_image.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    with (metrics_dir / f"{stem}_per_image.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["dataset", "split", "image", "gt", "pred", "tp", "fp", "fn", "human_gt_label", "champion_pred_label", "pred_label"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})

    if save_visualizations:
        generate_model_pk_visualizations(
            output_dir=output_dir,
            dataset_name=dataset_name,
            split=split,
            rows=rows,
            cfg=cfg,
            backend=backend,
        )
    return {"summary": summary, "rows": rows}


def generate_model_pk_visualizations(
    *,
    output_dir: Path,
    dataset_name: str,
    split: str,
    rows: list[dict[str, Any]],
    cfg: Any,
    backend: EvaluationBackend | None = None,
) -> None:
    """生成 Model PK 可视化，Champion 预测同时保存为 labels_gt 和 labels_champion。"""
    backend = backend or backend_from_match_config(cfg)
    plan = backend.build_visualization_plan(rows, None)
    rows_by_image = {str(row["image"]): row for row in rows}
    class_names = None
    if backend.task == "detect" and rows:
        if rows[0].get("data_path"):
            class_names = normalize_class_names(load_yaml(Path(str(rows[0]["data_path"]))).get("names"))
        else:
            class_names = rows[0].get("class_names") or {}
    for image, categories in plan.items():
        row = rows_by_image[image]
        img_path = Path(image)
        champion_label = Path(str(row["champion_pred_label"]))
        candidate_label = Path(str(row["pred_label"]))
        w, h = image_size(img_path)
        champion_items = backend.read_txt(champion_label, w, h, True)
        candidate_items = backend.read_txt(candidate_label, w, h, True)
        champion_items = remap_items_to_class_context(champion_items, row.get("champion_class_id_map") or {})
        candidate_items = remap_items_to_class_context(candidate_items, row.get("candidate_class_id_map") or {})
        for category in categories:
            visual_kwargs = {"class_names": class_names} if class_names is not None else {}
            backend.save_visualization_sample(
                output_dir=output_dir,
                dataset_name=dataset_name,
                split=split,
                category=category,
                img_path=img_path,
                gt_label=champion_label,
                candidate_label=candidate_label,
                candidate_items=candidate_items,
                gt_items=champion_items,
                champion_label=champion_label,
                champion_items=None,
                cfg=cfg,
                **visual_kwargs,
            )


def write_model_pk_overall_summary(run_dir: Path, results: list[dict[str, Any]]) -> None:
    """写 Model PK 汇总指标。"""
    rows = []
    for item in results:
        s = item["summary"]
        totals = s["totals"]
        metrics = s["metrics"]
        rows.append(
            {
                "dataset": s["dataset"],
                "split": s["split"],
                "images": totals["images"],
                "champion_pred_as_gt": totals["champion_pred_as_gt"],
                "candidate_pred": totals["candidate_pred"],
                "tp": totals["tp"],
                "fp": totals["fp"],
                "fn": totals["fn"],
                "precision_vs_champion": metrics["precision"],
                "recall_vs_champion": metrics["recall"],
                "f1_vs_champion": metrics["f1"],
            }
        )
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    write_json(metrics_dir / "overall_summary.json", rows)
    with (metrics_dir / "overall_summary.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "dataset",
            "split",
            "images",
            "champion_pred_as_gt",
            "candidate_pred",
            "tp",
            "fp",
            "fn",
            "precision_vs_champion",
            "recall_vs_champion",
            "f1_vs_champion",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_model_pk(config_path: str | Path, profile: str = "default", logger=None) -> Path:
    """执行完整 Model PK 流程，返回 run_dir。"""
    cfg_path = Path(config_path).expanduser().resolve()
    config = model_pk_config_from_project(load_yaml(cfg_path), profile=profile)
    candidate_raw = str(config.get("candidate_model") or "").strip()
    if candidate_raw:
        candidate_model = Path(candidate_raw).expanduser()
    else:
        default_candidate = default_candidate_from_train(config)
        if default_candidate is None:
            raise ValueError("candidate_model is empty and train.project/train.name is not available")
        candidate_model = default_candidate
    champion_model = Path(str(config.get("champion_model") or "")).expanduser()
    if not candidate_model.is_file():
        raise FileNotFoundError(f"candidate_model does not exist: {candidate_model}")
    if not champion_model.is_file():
        raise FileNotFoundError(f"champion_model does not exist: {champion_model}")

    run_dir = make_run_dir(config)
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    if logger:
        add_file_handler(logger, log_dir / "run.log")
        logger.info("Model PK run dir: %s", run_dir)

    manifests = []
    for ds in config.get("eval_datasets", []):
        if ds.get("images"):
            continue
        manifest = build_dataset_manifest(ds["data"], ds.get("name"))
        manifests.append(manifest)
        write_json(run_dir / "manifests" / f"dataset_manifest_{manifest['dataset_name']}.json", manifest)
        if logger:
            logger.info(
                "Dataset %s version=%s images=%s leakage=%s",
                manifest["dataset_name"],
                manifest["dataset_version"],
                manifest["image_count"],
                len(manifest.get("leakage", [])),
            )

    config["candidate_model"] = str(candidate_model.resolve())
    config["champion_model"] = str(champion_model.resolve())
    write_run_manifest(run_dir, cfg_path, config, manifests)

    backend = backend_from_eval_protocol(config.get("eval", {}))
    eval_cfg = backend.cfg
    candidate_model_names = load_yolo_model_names(candidate_model)
    champion_model_names = load_yolo_model_names(champion_model)
    results = []
    for ds in config.get("eval_datasets", []):
        for source in collect_model_pk_sources(ds):
            dataset_name = source["dataset_name"]
            split = source["split"]
            data_path = source["data_path"]
            images = source["images"]
            candidate_pred_dir = run_dir / "cache" / "candidate" / dataset_name / split / "_pred_labels"
            champion_pred_dir = run_dir / "cache" / "champion" / dataset_name / split / "_pred_labels"
            if logger:
                logger.info("Predicting candidate dataset=%s split=%s", dataset_name, split)
            backend.predict_to_labels(candidate_model, images, candidate_pred_dir, eval_cfg, progress_desc=f"candidate {dataset_name} {split} predict")
            if logger:
                logger.info("Predicting champion dataset=%s split=%s", dataset_name, split)
            backend.predict_to_labels(champion_model, images, champion_pred_dir, eval_cfg, progress_desc=f"champion {dataset_name} {split} predict")
            if logger:
                logger.info("Model PK evaluating dataset=%s split=%s", dataset_name, split)
            result = run_model_pk_from_prediction_dirs(
                data_path=data_path,
                dataset_name=dataset_name,
                split=split,
                images=images,
                class_names=source.get("class_names"),
                candidate_model_names=candidate_model_names,
                champion_model_names=champion_model_names,
                output_dir=run_dir,
                candidate_pred_dir=candidate_pred_dir,
                champion_pred_dir=champion_pred_dir,
                cfg=eval_cfg,
                save_visualizations=bool(eval_cfg.save_diff),
            )
            results.append(result)
            if logger:
                logger.info("Model PK result %s/%s: %s", dataset_name, split, result["summary"]["metrics"])

    write_model_pk_overall_summary(run_dir, results)
    return run_dir
