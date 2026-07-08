"""BBox IoU based YOLO detect evaluation and visualization."""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont

from .config import load_yaml
from .detect_io import DetectItem, image_size, read_detect_txt, result_to_items, write_detect_txt
from .paths import collect_images, resolve_dataset_paths


@dataclass
class DetectMatchConfig:
    """Configuration for detect inference and bbox IoU matching."""

    match_iou: float = 0.50
    device: str = "0"
    imgsz: int = 1280
    batch: int = 1
    conf: float = 0.25
    nms_iou: float = 0.45
    half: bool = True
    save_diff: bool = True
    show_progress: bool = True


@dataclass
class Match:
    """Single GT-prediction bbox match."""

    gt_idx: int
    pred_idx: int
    cls_id: int
    iou: float
    reason: str = "bbox_iou"


def maybe_progress(
    iterable: Iterable[Any],
    *,
    enabled: bool,
    desc: str,
    total: int | None = None,
    tqdm_factory=None,
) -> Iterable[Any]:
    """Wrap an iterable in tqdm when enabled and available."""
    if not enabled:
        return iterable
    if tqdm_factory is None:
        try:
            from tqdm.auto import tqdm as tqdm_factory
        except ImportError:
            return iterable
    return tqdm_factory(iterable, desc=desc, total=total, unit="img", dynamic_ncols=True)


def iou(a: DetectItem, b: DetectItem) -> float:
    """Compute bbox IoU."""
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def detect_match(gt_items: list[DetectItem], pred_items: list[DetectItem], cfg: DetectMatchConfig) -> list[Match]:
    """Greedily match same-class predictions to GT by descending IoU."""
    candidates: list[tuple[float, int, int]] = []
    for gi, gt in enumerate(gt_items):
        for pi, pred in enumerate(pred_items):
            if gt.cls_id != pred.cls_id:
                continue
            value = iou(gt, pred)
            if value >= cfg.match_iou:
                candidates.append((value, gi, pi))
    candidates.sort(reverse=True)

    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matches: list[Match] = []
    for value, gi, pi in candidates:
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        matches.append(Match(gi, pi, gt_items[gi].cls_id, value))
    return matches


def classify_unmatched_predictions(
    gt_items: list[DetectItem],
    pred_items: list[DetectItem],
    matches: list[Match],
    unmatched_pred: list[int],
    cfg: DetectMatchConfig,
) -> tuple[list[int], list[int], dict[int, dict[str, Any]]]:
    """Split unmatched predictions into duplicate FP and real FP."""
    matched_gt = {m.gt_idx for m in matches}
    duplicate_pred: list[int] = []
    real_fp_pred: list[int] = []
    duplicate_info: dict[int, dict[str, Any]] = {}
    for pred_idx in unmatched_pred:
        best_iou = 0.0
        best_gt_idx: int | None = None
        for gt_idx in matched_gt:
            gt = gt_items[gt_idx]
            pred = pred_items[pred_idx]
            if gt.cls_id != pred.cls_id:
                continue
            value = iou(gt, pred)
            if value > best_iou:
                best_iou = value
                best_gt_idx = gt_idx
        if best_gt_idx is not None and best_iou >= cfg.match_iou:
            duplicate_pred.append(pred_idx)
            duplicate_info[pred_idx] = {"matched_gt_idx": best_gt_idx, "iou": best_iou, "reason": "duplicate_bbox_iou"}
        else:
            real_fp_pred.append(pred_idx)
    return duplicate_pred, real_fp_pred, duplicate_info


def precision_recall(tp: int, fp: int, fn: int) -> dict[str, float]:
    """Compute precision, recall, and F1 from counts."""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def predict_to_labels(
    model_path: Path,
    images: list[Path],
    pred_dir: Path,
    cfg: DetectMatchConfig,
    progress_desc: str | None = None,
) -> None:
    """Run YOLO detect inference and write prediction labels to pred_dir."""
    from ultralytics import YOLO

    if pred_dir.exists():
        shutil.rmtree(pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_path))
    results = model.predict(
        source=[str(p) for p in images],
        device=cfg.device,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        conf=cfg.conf,
        iou=cfg.nms_iou,
        half=cfg.half,
        save=False,
        save_txt=False,
        stream=True,
        verbose=False,
    )
    for result in maybe_progress(
        results,
        enabled=cfg.show_progress,
        desc=progress_desc or f"{model_path.stem} predict",
        total=len(images),
    ):
        img_path = Path(result.path)
        items = result_to_items(result)
        img_w, img_h = image_size(img_path)
        write_detect_txt(pred_dir / f"{img_path.stem}.txt", items, img_w, img_h)


def load_font(size: int = 18):
    """Load visualization font."""
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def normalize_class_names(names: Any) -> dict[int, str]:
    """Normalize dataset YAML names into an id-to-name mapping."""
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {idx: str(name) for idx, name in enumerate(names)}
    return {}


def format_item_label(item: DetectItem, class_names: dict[int, str] | None = None, include_conf: bool = False) -> str:
    """Format a visual label as class name/id, optionally followed by confidence."""
    name = (class_names or {}).get(item.cls_id, f"cls_{item.cls_id}")
    if include_conf:
        return f"{name} {item.conf:.2f}"
    return name


def draw_title(img: Image.Image, title: str) -> Image.Image:
    """Draw a small title band above an image."""
    font = load_font(18)
    title_h = 34
    out = Image.new("RGB", (img.width, img.height + title_h), "white")
    out.paste(img, (0, title_h))
    draw = ImageDraw.Draw(out)
    draw.text((10, 8), title, fill="black", font=font)
    return out


def draw_box(draw: ImageDraw.ImageDraw, item: DetectItem, color: str, width: int = 3, pad: int = 2) -> None:
    """Draw a detection bbox."""
    draw.rectangle([item.x1 - pad, item.y1 - pad, item.x2 + pad, item.y2 + pad], outline=color, width=width)


def draw_items(
    draw: ImageDraw.ImageDraw,
    items: list[DetectItem],
    color: str,
    class_names: dict[int, str] | None = None,
    include_conf: bool = False,
) -> None:
    """Draw detection items."""
    font = load_font(14)
    for item in items:
        draw_box(draw, item, color)
        draw.text((item.x1, max(0, item.y1 - 18)), format_item_label(item, class_names, include_conf), fill=color, font=font)


def draw_prediction_item(
    draw: ImageDraw.ImageDraw,
    item: DetectItem,
    color: str,
    class_names: dict[int, str] | None = None,
) -> None:
    """Draw a prediction item with confidence."""
    font = load_font(14)
    draw_box(draw, item, color)
    draw.text((item.x1, max(0, item.y1 - 18)), format_item_label(item, class_names, include_conf=True), fill=color, font=font)


def classify_items_for_visualization(
    gt_items: list[DetectItem],
    pred_items: list[DetectItem],
    cfg: DetectMatchConfig,
) -> dict[str, set[int]]:
    """Classify single-image prediction indices for visualization."""
    matches = detect_match(gt_items, pred_items, cfg)
    tp_pred_indices = {m.pred_idx for m in matches}
    matched_gt_indices = {m.gt_idx for m in matches}
    fp_pred_indices = {i for i in range(len(pred_items)) if i not in tp_pred_indices}
    fn_gt_indices = {i for i in range(len(gt_items)) if i not in matched_gt_indices}
    return {
        "tp_pred_indices": tp_pred_indices,
        "fp_pred_indices": fp_pred_indices,
        "fn_gt_indices": fn_gt_indices,
    }


def render_overlay_panel(
    img_path: Path,
    title: str,
    items: list[DetectItem],
    color: str,
    class_names: dict[int, str] | None = None,
    include_conf: bool = False,
) -> Image.Image:
    """Render one panel with a set of detect items."""
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    draw_items(draw, items, color, class_names=class_names, include_conf=include_conf)
    return draw_title(img, title)


def render_prediction_panel(
    img_path: Path,
    title: str,
    gt_items: list[DetectItem],
    pred_items: list[DetectItem],
    cfg: DetectMatchConfig,
    class_names: dict[int, str] | None = None,
) -> Image.Image:
    """Render predictions with TP green and FP/FN orange."""
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    classified = classify_items_for_visualization(gt_items, pred_items, cfg)
    for idx in classified["tp_pred_indices"]:
        draw_prediction_item(draw, pred_items[idx], "lime", class_names)
    for idx in classified["fp_pred_indices"]:
        draw_prediction_item(draw, pred_items[idx], "orange", class_names)
    for idx in classified["fn_gt_indices"]:
        draw_box(draw, gt_items[idx], "orange", width=4, pad=4)
        draw.text(
            (gt_items[idx].x1, max(0, gt_items[idx].y1 - 18)),
            format_item_label(gt_items[idx], class_names, include_conf=False),
            fill="orange",
            font=load_font(14),
        )
    return draw_title(img, title)


def save_compare_visual(
    out_path: Path,
    img_path: Path,
    gt_items: list[DetectItem],
    candidate_items: list[DetectItem],
    champion_items: list[DetectItem] | None = None,
    cfg: DetectMatchConfig | None = None,
    class_names: dict[int, str] | None = None,
) -> None:
    """Save original / GT / Champion / Candidate comparison image."""
    cfg = cfg or DetectMatchConfig()
    original = draw_title(Image.open(img_path).convert("RGB"), "original")
    gt_panel = render_overlay_panel(img_path, "gt", gt_items, "lime", class_names=class_names, include_conf=False)
    panels = [original, gt_panel]
    if champion_items is not None:
        panels.append(render_prediction_panel(img_path, "champion", gt_items, champion_items, cfg, class_names=class_names))
    panels.append(render_prediction_panel(img_path, "candidate", gt_items, candidate_items, cfg, class_names=class_names))

    width = sum(panel.width for panel in panels)
    height = max(panel.height for panel in panels)
    out = Image.new("RGB", (width, height), "white")
    x = 0
    for panel in panels:
        out.paste(panel, (x, 0))
        x += panel.width
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)


def copy_label_or_empty(src: Path | None, dst: Path) -> None:
    """Copy a label file, or write an empty label if it is missing."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src is not None and src.exists():
        shutil.copy2(src, dst)
    else:
        dst.write_text("", encoding="utf-8")


def save_visualization_sample(
    output_dir: Path,
    dataset_name: str,
    split: str,
    category: str,
    img_path: Path,
    gt_label: Path,
    candidate_label: Path,
    candidate_items: list[DetectItem],
    gt_items: list[DetectItem],
    champion_label: Path | None = None,
    champion_items: list[DetectItem] | None = None,
    cfg: DetectMatchConfig | None = None,
    class_names: dict[int, str] | None = None,
) -> None:
    """Save one visualization sample under visualizations/<dataset>/<split>/<category>."""
    out_dir = output_dir / "visualizations" / dataset_name / split / category
    for sub in ("images", "labels_gt", "labels_candidate", "compare"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    shutil.copy2(img_path, out_dir / "images" / img_path.name)
    copy_label_or_empty(gt_label, out_dir / "labels_gt" / gt_label.name)
    copy_label_or_empty(candidate_label, out_dir / "labels_candidate" / candidate_label.name)
    if champion_label is not None:
        (out_dir / "labels_champion").mkdir(parents=True, exist_ok=True)
        copy_label_or_empty(champion_label, out_dir / "labels_champion" / champion_label.name)
    save_compare_visual(
        out_dir / "compare" / img_path.name,
        img_path,
        gt_items=gt_items,
        candidate_items=candidate_items,
        champion_items=champion_items,
        cfg=cfg,
        class_names=class_names,
    )


def build_visualization_plan(
    candidate_rows: list[dict[str, Any]],
    champion_rows: list[dict[str, Any]] | None = None,
) -> dict[str, set[str]]:
    """Build visualization categories from per-image rows."""
    champion_by_image = {str(row["image"]): row for row in (champion_rows or [])}
    plan: dict[str, set[str]] = {}
    for row in candidate_rows:
        image = str(row["image"])
        categories: set[str] = set()
        cand_fp = int(row.get("fp", 0))
        cand_fn = int(row.get("fn", 0))
        if cand_fp > 0:
            categories.add("fp")
        if cand_fn > 0:
            categories.add("fn")

        champion = champion_by_image.get(image)
        if champion is not None:
            champ_fp = int(champion.get("fp", 0))
            champ_fn = int(champion.get("fn", 0))
            if cand_fp > 0 and champ_fp == 0:
                categories.add("candidate_new_fp")
            if cand_fn > 0 and champ_fn == 0:
                categories.add("candidate_new_fn")
            if cand_fp + cand_fn < champ_fp + champ_fn:
                categories.add("candidate_improved")
        if categories:
            plan[image] = categories
    return plan


def label_for_row(row: dict[str, Any], key: str) -> Path:
    """Read a label path from a per-image row."""
    return Path(str(row[key]))


def generate_comparison_visualizations(
    output_dir: Path,
    dataset_name: str,
    split: str,
    candidate_rows: list[dict[str, Any]],
    champion_rows: list[dict[str, Any]] | None = None,
    cfg: DetectMatchConfig | None = None,
    class_names: dict[int, str] | None = None,
) -> None:
    """Generate detect comparison visualizations from per-image rows."""
    cfg = cfg or DetectMatchConfig()
    if class_names is None and candidate_rows:
        data_path = candidate_rows[0].get("data_path")
        if data_path:
            class_names = normalize_class_names(load_yaml(Path(str(data_path))).get("names"))
    plan = build_visualization_plan(candidate_rows, champion_rows)
    candidate_by_image = {str(row["image"]): row for row in candidate_rows}
    champion_by_image = {str(row["image"]): row for row in (champion_rows or [])}
    for image, categories in plan.items():
        cand_row = candidate_by_image[image]
        champ_row = champion_by_image.get(image)
        img_path = Path(image)
        gt_label = label_for_row(cand_row, "gt_label")
        candidate_label = label_for_row(cand_row, "pred_label")
        w, h = image_size(img_path)
        gt_items = read_detect_txt(gt_label, w, h, has_conf=False)
        candidate_items = read_detect_txt(candidate_label, w, h, has_conf=True)
        champion_label = label_for_row(champ_row, "pred_label") if champ_row is not None else None
        champion_items = read_detect_txt(champion_label, w, h, has_conf=True) if champion_label is not None else None
        for category in categories:
            save_visualization_sample(
                output_dir=output_dir,
                dataset_name=dataset_name,
                split=split,
                category=category,
                img_path=img_path,
                gt_label=gt_label,
                candidate_label=candidate_label,
                candidate_items=candidate_items,
                gt_items=gt_items,
                champion_label=champion_label,
                champion_items=champion_items,
                cfg=cfg,
                class_names=class_names,
            )


def evaluate_split(
    data_path: Path,
    dataset_name: str,
    split: str,
    output_dir: Path,
    cfg: DetectMatchConfig,
    model_path: Path | None = None,
    pred_label_dirs: list[Path] | None = None,
    model_role: str = "candidate",
) -> dict[str, Any]:
    """Evaluate one detect dataset split."""
    data = load_yaml(data_path)
    pairs = resolve_dataset_paths(data, split)
    if not pairs:
        raise ValueError(f"No split '{split}' found in {data_path}")

    image_label_pairs: list[tuple[Path, Path, Path]] = []
    for image_dir, label_dir in pairs:
        for img_path in collect_images(image_dir):
            image_label_pairs.append((img_path, label_dir / f"{img_path.stem}.txt", image_dir))
    if not image_label_pairs:
        raise ValueError(f"No images found for dataset={dataset_name} split={split} data={data_path}")

    pred_root = output_dir / "cache" / model_role / dataset_name / split / "_pred_labels"
    if pred_label_dirs is None:
        if model_path is None:
            raise ValueError("model_path or pred_label_dirs is required")
        predict_to_labels(
            model_path,
            [p[0] for p in image_label_pairs],
            pred_root,
            cfg,
            progress_desc=f"{model_role} {dataset_name} {split} predict",
        )
        pred_label_dirs = [pred_root]
    else:
        pred_label_dirs = [Path(p) for p in pred_label_dirs]

    totals = {
        "images": 0,
        "gt": 0,
        "pred": 0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "duplicate_fp": 0,
        "real_fp": 0,
    }
    rows: list[dict[str, Any]] = []

    for img_path, gt_label, _image_dir in maybe_progress(
        image_label_pairs,
        enabled=cfg.show_progress,
        desc=f"{model_role} {dataset_name} {split} match",
        total=len(image_label_pairs),
    ):
        w, h = image_size(img_path)
        gt_items = read_detect_txt(gt_label, w, h, has_conf=False)
        pred_label = None
        for pred_dir in pred_label_dirs:
            candidate = pred_dir / f"{img_path.stem}.txt"
            if candidate.exists():
                pred_label = candidate
                break
        if pred_label is None:
            pred_label = pred_label_dirs[0] / f"{img_path.stem}.txt"
        pred_items = read_detect_txt(pred_label, w, h, has_conf=True)

        matches = detect_match(gt_items, pred_items, cfg)
        matched_gt = {m.gt_idx for m in matches}
        matched_pred = {m.pred_idx for m in matches}
        unmatched_gt = [i for i in range(len(gt_items)) if i not in matched_gt]
        unmatched_pred = [i for i in range(len(pred_items)) if i not in matched_pred]
        duplicate_pred, real_fp_pred, duplicate_info = classify_unmatched_predictions(
            gt_items, pred_items, matches, unmatched_pred, cfg
        )
        tp = len(matches)
        fp = len(unmatched_pred)
        fn = len(unmatched_gt)

        totals["images"] += 1
        totals["gt"] += len(gt_items)
        totals["pred"] += len(pred_items)
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn
        totals["duplicate_fp"] += len(duplicate_pred)
        totals["real_fp"] += len(real_fp_pred)

        rows.append(
            {
                "dataset": dataset_name,
                "split": split,
                "model_role": model_role,
                "image": str(img_path),
                "data_path": str(data_path),
                "gt_label": str(gt_label),
                "pred_label": str(pred_label),
                "gt": len(gt_items),
                "pred": len(pred_items),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "duplicate_fp": len(duplicate_pred),
                "real_fp": len(real_fp_pred),
                "matches": [asdict(m) for m in matches],
                "duplicate_info": {str(k): v for k, v in duplicate_info.items()},
            }
        )

    metrics = precision_recall(totals["tp"], totals["fp"], totals["fn"])
    adjusted_metrics = precision_recall(totals["tp"], totals["real_fp"], totals["fn"])
    summary = {
        "dataset": dataset_name,
        "data": str(Path(data_path).resolve()),
        "split": split,
        "model_role": model_role,
        "model": str(model_path.resolve()) if model_path else None,
        "pred_label_dirs": [str(p) for p in pred_label_dirs],
        "names": data.get("names"),
        "match": {
            "type": "bbox_iou",
            "match_iou": cfg.match_iou,
        },
        "totals": totals,
        "metrics": metrics,
        "adjusted_metrics_ignore_duplicate_fp": adjusted_metrics,
    }
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{model_role}_{dataset_name}_{split}"
    (metrics_dir / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (metrics_dir / f"{stem}_per_image.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    with (metrics_dir / f"{stem}_per_image.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["dataset", "split", "model_role", "image", "gt", "pred", "tp", "fp", "fn", "duplicate_fp", "real_fp"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})
    return {"summary": summary, "rows": rows}


def detect_config_from_dict(data: dict[str, Any]) -> DetectMatchConfig:
    """Create DetectMatchConfig while ignoring unknown keys."""
    return DetectMatchConfig(**{k: v for k, v in data.items() if k in DetectMatchConfig.__dataclass_fields__})
