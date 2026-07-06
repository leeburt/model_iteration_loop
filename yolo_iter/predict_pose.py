"""单模型图片推理工具。

支持指定模型和单张图片/图片目录，输出 YOLO pose 预测标签、可视化图和逐图汇总。
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .paths import IMG_EXTS, collect_images
from .pose_io import PoseItem, image_size, result_to_items, write_pose_txt
from .pose_tiny_match import TinyMatchConfig, draw_prediction_item, draw_title, maybe_progress


def collect_source_images(source: str | Path) -> list[Path]:
    """收集单张图片或目录下的图片，返回绝对路径列表。"""
    src = Path(source).expanduser()
    if src.is_file():
        if src.suffix.lower() not in IMG_EXTS:
            raise ValueError(f"source is not a supported image: {src}")
        return [src.resolve()]
    if src.is_dir():
        images = [p.resolve() for p in collect_images(src)]
        if not images:
            raise ValueError(f"No images found in source directory: {src}")
        return images
    raise FileNotFoundError(f"source does not exist: {src}")


def render_prediction_visual(img_path: Path, items: list[PoseItem], out_path: Path) -> None:
    """保存单图预测覆盖图。"""
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for item in items:
        draw_prediction_item(draw, item, "lime")
    out = draw_title(img, "prediction")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)


def write_prediction_outputs(
    items_by_image: dict[Path, list[PoseItem]],
    output_dir: str | Path,
    *,
    save_visualizations: bool = True,
) -> list[dict[str, Any]]:
    """写出预测标签、可视化图和 summary 文件。"""
    out_dir = Path(output_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    labels_dir = out_dir / "labels"
    visuals_dir = out_dir / "visualizations"
    labels_dir.mkdir(parents=True, exist_ok=True)
    if save_visualizations:
        visuals_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for img_path, items in sorted(items_by_image.items(), key=lambda kv: str(kv[0])):
        img_w, img_h = image_size(img_path)
        label_path = labels_dir / f"{img_path.stem}.txt"
        write_pose_txt(label_path, items, img_w, img_h)
        if save_visualizations:
            render_prediction_visual(img_path, items, visuals_dir / img_path.name)
        rows.append(
            {
                "image": str(img_path),
                "label": str(label_path),
                "visualization": str(visuals_dir / img_path.name) if save_visualizations else "",
                "predictions": len(items),
            }
        )

    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "label", "visualization", "predictions"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return rows


def predict_pose_images(
    *,
    model_path: str | Path,
    source: str | Path,
    output_dir: str | Path,
    cfg: TinyMatchConfig,
    save_visualizations: bool = True,
) -> Path:
    """执行 YOLO pose 推理并写出结果，返回输出目录。"""
    from ultralytics import YOLO

    model = Path(model_path).expanduser()
    if not model.is_file():
        raise FileNotFoundError(f"model does not exist: {model}")
    images = collect_source_images(source)
    yolo = YOLO(str(model))
    results = yolo.predict(
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

    items_by_image: dict[Path, list[PoseItem]] = {}
    for result in maybe_progress(
        results,
        enabled=cfg.show_progress,
        desc=f"{model.stem} predict",
        total=len(images),
    ):
        img_path = Path(result.path).resolve()
        items_by_image[img_path] = result_to_items(result)

    write_prediction_outputs(items_by_image, output_dir, save_visualizations=save_visualizations)
    return Path(output_dir)
