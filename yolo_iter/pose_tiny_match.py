"""关键点感知的 GT-Pred 匹配与评估引擎。

核心匹配策略（TinyMatch）：
1. 同类 GT 与 Pred 关键点距离 ≤ 容忍阈值 → 直接匹配（keypoint 匹配）
2. 中心点距离 ≤ 容忍阈值 且 膨胀框 IoU ≥ min_padded_iou → 辅助匹配（center+padded_iou 匹配）
3. 未匹配的 Pred 若与已匹配 GT 存在候选匹配 → 标记为 duplicate_fp，否则 → real_fp

匹配完成后支持：
- 逐图 FP/FN 差异可视化（save_diff_visual）
- FP/FN 样本拷贝（copy_diff，含原图、GT/Pred 标签、JSON 元数据）
- 逐图及汇总指标输出（CSV + JSON）
"""

from __future__ import annotations

import csv
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont

from .config import load_yaml
from .paths import collect_images, resolve_dataset_paths
from .pose_io import PoseItem, image_size, read_pose_txt, result_to_items, write_pose_txt


@dataclass
class TinyMatchConfig:
    """TinyMatch 匹配参数配置。

    关键参数说明：
    - kp_px / kp_box_ratio: 关键点匹配的像素/比例容忍阈值
    - center_px / center_box_ratio: 中心点匹配的像素/比例容忍阈值
    - pad_px: 计算辅助 IoU 时对 bbox 的像素膨胀量（针对极小目标）
    - min_padded_iou: 中心匹配时要求的最低膨胀 IoU
    - baseline_iou: 作为参考的严格 bbox IoU 阈值（非主匹配策略）
    - device / imgsz / batch / conf / nms_iou / half: 推理参数
    - save_diff / save_fp_diff_only / save_duplicate_fp_diff: diff 输出控制
    """

    kp_px: float = 6.0
    kp_box_ratio: float = 0.60
    center_px: float = 8.0
    center_box_ratio: float = 0.75
    pad_px: float = 6.0
    min_padded_iou: float = 0.01
    baseline_iou: float = 0.50
    device: str = "0"
    imgsz: int = 1536
    batch: int = 1
    conf: float = 0.25
    nms_iou: float = 0.45
    half: bool = True
    save_diff: bool = True
    save_fp_diff_only: bool = False
    save_duplicate_fp_diff: bool = False


@dataclass
class Match:
    """单次 GT-Pred 匹配结果。score 越低匹配质量越高。"""

    gt_idx: int
    pred_idx: int
    cls_id: int
    score: float
    iou: float
    padded_iou: float
    center_dist: float
    kp_dist: float | None
    reason: str


def box_center(item: PoseItem) -> tuple[float, float]:
    """返回检测框中心点 (cx, cy)。"""
    return (item.x1 + item.x2) / 2.0, (item.y1 + item.y2) / 2.0


def box_size(item: PoseItem) -> tuple[float, float]:
    """返回检测框宽高 (w, h)。"""
    return max(0.0, item.x2 - item.x1), max(0.0, item.y2 - item.y1)


def visible_keypoint(item: PoseItem) -> tuple[float, float] | None:
    """返回第一个可见关键点的坐标 (v>0)，无可视关键点返回 None。"""
    for x, y, v in item.kpts:
        if v > 0:
            return x, y
    return None


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    """两点欧氏距离。"""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def iou(a: PoseItem, b: PoseItem, pad: float = 0.0) -> float:
    """计算两个检测框的 IoU，可选 pad 像素膨胀（用于极小目标）。"""
    ax1, ay1, ax2, ay2 = a.x1 - pad, a.y1 - pad, a.x2 + pad, a.y2 + pad
    bx1, by1, bx2, by2 = b.x1 - pad, b.y1 - pad, b.x2 + pad, b.y2 + pad
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def candidate_match(gt: PoseItem, pred: PoseItem, cfg: TinyMatchConfig) -> Match | None:
    """判断单个 GT 与 Pred 是否可匹配，返回 Match 或 None。

    匹配优先级：关键点距离匹配 > 中心点+膨胀 IoU 匹配。
    score 越低表示匹配质量越高，用于后续贪心排序。
    """
    if gt.cls_id != pred.cls_id:
        return None

    gt_w, gt_h = box_size(gt)
    gt_diag = math.hypot(gt_w, gt_h)
    center_tol = max(cfg.center_px, cfg.center_box_ratio * max(gt_w, gt_h))
    kp_tol = max(cfg.kp_px, cfg.kp_box_ratio * gt_diag)

    base_iou = iou(gt, pred)
    padded_iou = iou(gt, pred, pad=cfg.pad_px)
    center_dist = distance(box_center(gt), box_center(pred))

    gt_kp = visible_keypoint(gt)
    pred_kp = visible_keypoint(pred)
    kp_dist = None
    if gt_kp is not None and pred_kp is not None:
        kp_dist = distance(gt_kp, pred_kp)
        if kp_dist <= kp_tol:
            score = kp_dist / max(kp_tol, 1e-6) + center_dist / max(center_tol, 1e-6) * 0.25
            return Match(-1, -1, gt.cls_id, score, base_iou, padded_iou, center_dist, kp_dist, "keypoint")

    if center_dist <= center_tol and padded_iou >= cfg.min_padded_iou:
        score = 1.0 + center_dist / max(center_tol, 1e-6) - padded_iou * 0.25
        return Match(-1, -1, gt.cls_id, score, base_iou, padded_iou, center_dist, kp_dist, "center+padded_iou")
    return None


def tiny_match(gt_items: list[PoseItem], pred_items: list[PoseItem], cfg: TinyMatchConfig) -> list[Match]:
    """对一张图的所有 GT 和 Pred 执行贪心匹配。

    生成所有候选匹配 → 按 (score, -padded_iou, -conf) 排序 →
    贪心选取，每个 GT 和 Pred 最多匹配一次。
    """
    candidates: list[Match] = []
    for gi, gt in enumerate(gt_items):
        for pi, pred in enumerate(pred_items):
            cand = candidate_match(gt, pred, cfg)
            if cand is None:
                continue
            cand.gt_idx = gi
            cand.pred_idx = pi
            candidates.append(cand)
    candidates.sort(key=lambda m: (m.score, -m.padded_iou, -pred_items[m.pred_idx].conf))
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matches: list[Match] = []
    for cand in candidates:
        if cand.gt_idx in used_gt or cand.pred_idx in used_pred:
            continue
        used_gt.add(cand.gt_idx)
        used_pred.add(cand.pred_idx)
        matches.append(cand)
    return matches


def classify_unmatched_predictions(
    gt_items: list[PoseItem],
    pred_items: list[PoseItem],
    matches: list[Match],
    unmatched_pred: list[int],
    cfg: TinyMatchConfig,
) -> tuple[list[int], list[int], dict[int, dict[str, Any]]]:
    """将未匹配预测分为 duplicate_fp（匹配已占用 GT）和 real_fp（无任何 GT 匹配）。

    Returns:
        (duplicate_pred_indices, real_fp_pred_indices, duplicate_info_dict)
    """
    matched_gt = {m.gt_idx for m in matches}
    duplicate_pred: list[int] = []
    real_fp_pred: list[int] = []
    duplicate_info: dict[int, dict[str, Any]] = {}
    for pred_idx in unmatched_pred:
        best: Match | None = None
        for gt_idx in matched_gt:
            cand = candidate_match(gt_items[gt_idx], pred_items[pred_idx], cfg)
            if cand is None:
                continue
            cand.gt_idx = gt_idx
            cand.pred_idx = pred_idx
            if best is None or cand.score < best.score:
                best = cand
        if best is None:
            real_fp_pred.append(pred_idx)
        else:
            duplicate_pred.append(pred_idx)
            duplicate_info[pred_idx] = {
                "matched_gt_idx": best.gt_idx,
                "score": best.score,
                "iou": best.iou,
                "padded_iou": best.padded_iou,
                "center_dist": best.center_dist,
                "kp_dist": best.kp_dist,
                "reason": best.reason,
            }
    return duplicate_pred, real_fp_pred, duplicate_info


def strict_iou_match(gt_items: list[PoseItem], pred_items: list[PoseItem], thresh: float) -> int:
    """严格 bbox IoU 贪心匹配，返回 TP 数量。仅作为 baseline 参考指标。"""
    candidates = []
    for gi, gt in enumerate(gt_items):
        for pi, pred in enumerate(pred_items):
            if gt.cls_id == pred.cls_id:
                candidates.append((iou(gt, pred), gi, pi))
    candidates.sort(reverse=True)
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    count = 0
    for val, gi, pi in candidates:
        if val < thresh or gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        count += 1
    return count


def precision_recall(tp: int, fp: int, fn: int) -> dict[str, float]:
    """由 TP/FP/FN 计算 precision、recall、f1。"""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def predict_to_labels(model_path: Path, images: list[Path], pred_dir: Path, cfg: TinyMatchConfig) -> None:
    """用 YOLO 模型对图片列表执行推理，将预测结果写为 YOLO 标签文件到 pred_dir。

    会先清空 pred_dir 再写入。
    """
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
    for result in results:
        img_path = Path(result.path)
        items = result_to_items(result)
        img_w, img_h = image_size(img_path)
        write_pose_txt(pred_dir / f"{img_path.stem}.txt", items, img_w, img_h)


def draw_box(draw: ImageDraw.ImageDraw, item: PoseItem, color: str, width: int = 3, pad: int = 2) -> None:
    """在图像上绘制检测框。"""
    draw.rectangle([item.x1 - pad, item.y1 - pad, item.x2 + pad, item.y2 + pad], outline=color, width=width)


def draw_keypoint(draw: ImageDraw.ImageDraw, item: PoseItem, color: str, radius: int = 5) -> None:
    """在图像上绘制第一个可见关键点（十字+圆）。"""
    kp = visible_keypoint(item) or box_center(item)
    x, y = kp
    draw.ellipse([x - radius, y - radius, x + radius, y + radius], outline=color, fill=color)
    draw.line([x - radius * 2, y, x + radius * 2, y], fill=color, width=2)
    draw.line([x, y - radius * 2, x, y + radius * 2], fill=color, width=2)


def save_diff_visual(
    out_dir: Path,
    img_path: Path,
    gt_items: list[PoseItem],
    pred_items: list[PoseItem],
    matches: list[Match],
    unmatched_gt: list[int],
    duplicate_pred: list[int],
    real_fp_pred: list[int],
) -> None:
    """生成 FP/FN 可视化图。

    颜色约定：TP 绿色，FN 红色，duplicate FP 橙色，real FP 蓝色。
    """
    vis_dir = out_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    for match in matches:
        draw_box(draw, gt_items[match.gt_idx], "lime", width=2, pad=1)
        draw_keypoint(draw, pred_items[match.pred_idx], "lime", radius=4)
    for idx in unmatched_gt:
        gt = gt_items[idx]
        draw_box(draw, gt, "red", width=4, pad=4)
        draw_keypoint(draw, gt, "red", radius=6)
        draw.text((gt.x1, max(0, gt.y1 - 18)), "FN", fill="red", font=font)
    for idx in duplicate_pred:
        pred = pred_items[idx]
        draw_box(draw, pred, "orange", width=3, pad=3)
        draw_keypoint(draw, pred, "orange", radius=5)
        draw.text((pred.x1, pred.y2 + 20), f"DUP {pred.conf:.2f}", fill="orange", font=font)
    for idx in real_fp_pred:
        pred = pred_items[idx]
        draw_box(draw, pred, "dodgerblue", width=4, pad=4)
        draw_keypoint(draw, pred, "cyan", radius=6)
        draw.text((pred.x1, pred.y2 + 20), f"FP {pred.conf:.2f}", fill="dodgerblue", font=font)
    img.save(vis_dir / img_path.name)


def copy_diff(
    out_dir: Path,
    img_path: Path,
    gt_label: Path,
    pred_label: Path,
    unmatched_gt: Iterable[int],
    unmatched_pred: Iterable[int],
    duplicate_pred: Iterable[int],
    real_fp_pred: Iterable[int],
    duplicate_info: dict[int, dict[str, Any]],
) -> None:
    """拷贝 FP/FN 样本的原图、GT 标签、预测标签及 JSON 元数据到 diff 目录。"""
    for sub in ("images", "labels_gt", "labels_pred"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    shutil.copy2(img_path, out_dir / "images" / img_path.name)
    if gt_label.exists():
        shutil.copy2(gt_label, out_dir / "labels_gt" / gt_label.name)
    else:
        (out_dir / "labels_gt" / gt_label.name).write_text("", encoding="utf-8")
    if pred_label.exists():
        shutil.copy2(pred_label, out_dir / "labels_pred" / pred_label.name)
    else:
        (out_dir / "labels_pred" / pred_label.name).write_text("", encoding="utf-8")
    meta = {
        "image": str(img_path),
        "unmatched_gt": list(unmatched_gt),
        "unmatched_pred": list(unmatched_pred),
        "duplicate_pred": list(duplicate_pred),
        "real_fp_pred": list(real_fp_pred),
        "duplicate_info": {str(k): v for k, v in duplicate_info.items()},
    }
    (out_dir / f"{img_path.stem}.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def evaluate_split(
    data_path: Path,
    dataset_name: str,
    split: str,
    output_dir: Path,
    cfg: TinyMatchConfig,
    model_path: Path | None = None,
    pred_label_dirs: list[Path] | None = None,
    model_role: str = "candidate",
) -> dict[str, Any]:
    """对单个数据集的单个 split 执行完整评估。

    流程：加载数据 → 推理（或复用缓存预测）→ 逐图匹配 →
    分类 FP/FN → 输出指标、diff 样本、CSV/JSON 结果。

    Args:
        data_path: data.yaml 路径。
        dataset_name: 数据集名（用于输出目录和报告）。
        split: 目标 split。
        output_dir: 输出根目录。
        cfg: 匹配参数配置。
        model_path: 模型权重路径（与 pred_label_dirs 二选一）。
        pred_label_dirs: 预缓存预测标签目录列表（与 model_path 二选一）。
        model_role: "candidate" 或 "champion"。

    Returns:
        {"summary": {...}, "rows": [...]}
    """
    data = load_yaml(data_path)
    pairs = resolve_dataset_paths(data, split)
    if not pairs:
        raise ValueError(f"No split '{split}' found in {data_path}")

    image_label_pairs: list[tuple[Path, Path, Path]] = []
    for image_dir, label_dir in pairs:
        for img_path in collect_images(image_dir):
            image_label_pairs.append((img_path, label_dir / f"{img_path.stem}.txt", image_dir))

    pred_root = output_dir / "cache" / model_role / dataset_name / split / "_pred_labels"
    if pred_label_dirs is None:
        if model_path is None:
            raise ValueError("model_path or pred_label_dirs is required")
        predict_to_labels(model_path, [p[0] for p in image_label_pairs], pred_root, cfg)
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
        "baseline_tp": 0,
    }
    rows: list[dict[str, Any]] = []
    diff_root = output_dir / "diff" / model_role / dataset_name / split

    for img_path, gt_label, _image_dir in image_label_pairs:
        w, h = image_size(img_path)
        gt_items = read_pose_txt(gt_label, w, h, has_conf=False)
        pred_label = None
        for pred_dir in pred_label_dirs:
            candidate = pred_dir / f"{img_path.stem}.txt"
            if candidate.exists():
                pred_label = candidate
                break
        if pred_label is None:
            pred_label = pred_label_dirs[0] / f"{img_path.stem}.txt"
        pred_items = read_pose_txt(pred_label, w, h, has_conf=True)

        matches = tiny_match(gt_items, pred_items, cfg)
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
        baseline_tp = strict_iou_match(gt_items, pred_items, cfg.baseline_iou)

        totals["images"] += 1
        totals["gt"] += len(gt_items)
        totals["pred"] += len(pred_items)
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn
        totals["duplicate_fp"] += len(duplicate_pred)
        totals["real_fp"] += len(real_fp_pred)
        totals["baseline_tp"] += baseline_tp

        should_save_diff = cfg.save_diff and (fp or fn)
        if should_save_diff and cfg.save_fp_diff_only:
            should_save_diff = len(real_fp_pred) > 0 or (cfg.save_duplicate_fp_diff and len(duplicate_pred) > 0)
        if should_save_diff:
            copy_diff(diff_root, img_path, gt_label, pred_label, unmatched_gt, unmatched_pred, duplicate_pred, real_fp_pred, duplicate_info)
            save_diff_visual(diff_root, img_path, gt_items, pred_items, matches, unmatched_gt, duplicate_pred, real_fp_pred)

        rows.append(
            {
                "dataset": dataset_name,
                "split": split,
                "model_role": model_role,
                "image": str(img_path),
                "gt": len(gt_items),
                "pred": len(pred_items),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "duplicate_fp": len(duplicate_pred),
                "real_fp": len(real_fp_pred),
                "baseline_iou_tp": baseline_tp,
                "matches": [asdict(m) for m in matches],
                "duplicate_info": {str(k): v for k, v in duplicate_info.items()},
            }
        )

    metrics = precision_recall(totals["tp"], totals["fp"], totals["fn"])
    adjusted_metrics = precision_recall(totals["tp"], totals["real_fp"], totals["fn"])
    baseline_fp = totals["pred"] - totals["baseline_tp"]
    baseline_fn = totals["gt"] - totals["baseline_tp"]
    baseline_metrics = precision_recall(totals["baseline_tp"], baseline_fp, baseline_fn)
    summary = {
        "dataset": dataset_name,
        "data": str(Path(data_path).resolve()),
        "split": split,
        "model_role": model_role,
        "model": str(model_path.resolve()) if model_path else None,
        "pred_label_dirs": [str(p) for p in pred_label_dirs],
        "names": data.get("names"),
        "match": {
            "type": "keypoint_or_center_with_padded_iou",
            "kp_px": cfg.kp_px,
            "kp_box_ratio": cfg.kp_box_ratio,
            "center_px": cfg.center_px,
            "center_box_ratio": cfg.center_box_ratio,
            "pad_px": cfg.pad_px,
            "min_padded_iou": cfg.min_padded_iou,
        },
        "totals": totals,
        "metrics": metrics,
        "adjusted_metrics_ignore_duplicate_fp": adjusted_metrics,
        "baseline_iou": {
            "threshold": cfg.baseline_iou,
            "tp": totals["baseline_tp"],
            "fp": baseline_fp,
            "fn": baseline_fn,
            **baseline_metrics,
        },
    }
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{model_role}_{dataset_name}_{split}"
    (metrics_dir / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (metrics_dir / f"{stem}_per_image.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    with (metrics_dir / f"{stem}_per_image.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["dataset", "split", "model_role", "image", "gt", "pred", "tp", "fp", "fn", "duplicate_fp", "real_fp", "baseline_iou_tp"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})
    return {"summary": summary, "rows": rows}


def tiny_config_from_dict(data: dict[str, Any]) -> TinyMatchConfig:
    """从 dict 创建 TinyMatchConfig，自动忽略不在 dataclass 字段中的 key。"""
    return TinyMatchConfig(**{k: v for k, v in data.items() if k in TinyMatchConfig.__dataclass_fields__})
