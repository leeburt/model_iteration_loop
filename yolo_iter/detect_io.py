"""YOLO detect label IO and Ultralytics result conversion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass
class DetectItem:
    """Single detection item in pixel coordinates."""

    cls_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float = 1.0


def image_size(path: Path) -> tuple[int, int]:
    """Return image size as (width, height)."""
    with Image.open(path) as im:
        return im.size


def xywhn_to_item(parts: list[str], img_w: int, img_h: int, has_conf: bool) -> DetectItem | None:
    """Convert YOLO detect row tokens to a DetectItem."""
    if len(parts) < 5:
        return None
    cls_id = int(float(parts[0]))
    xc, yc, bw, bh = map(float, parts[1:5])
    conf = float(parts[5]) if has_conf and len(parts) >= 6 else 1.0
    cx = xc * img_w
    cy = yc * img_h
    w = bw * img_w
    h = bh * img_h
    return DetectItem(
        cls_id=cls_id,
        x1=cx - w / 2.0,
        y1=cy - h / 2.0,
        x2=cx + w / 2.0,
        y2=cy + h / 2.0,
        conf=conf,
    )


def read_detect_txt(path: Path, img_w: int, img_h: int, has_conf: bool = False) -> list[DetectItem]:
    """Read YOLO detect labels. Missing files return an empty list."""
    if not path.is_file():
        return []
    items: list[DetectItem] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            parts = raw.strip().split()
            if not parts:
                continue
            item = xywhn_to_item(parts, img_w, img_h, has_conf)
            if item is not None:
                items.append(item)
    return items


def write_detect_txt(path: Path, items: list[DetectItem], img_w: int, img_h: int) -> None:
    """Write YOLO detect prediction labels with confidence as the final field."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            xc = (item.x1 + item.x2) / 2.0 / img_w
            yc = (item.y1 + item.y2) / 2.0 / img_h
            bw = (item.x2 - item.x1) / img_w
            bh = (item.y2 - item.y1) / img_h
            fields = [
                str(item.cls_id),
                f"{xc:.6f}",
                f"{yc:.6f}",
                f"{bw:.6f}",
                f"{bh:.6f}",
                f"{item.conf:.6f}",
            ]
            f.write(" ".join(fields) + "\n")


def result_to_items(result) -> list[DetectItem]:
    """Convert an Ultralytics detect Results object to DetectItems."""
    items: list[DetectItem] = []
    if result.boxes is None:
        return items
    for box in result.boxes:
        cls_id = int(box.cls.item() if hasattr(box.cls, "item") else box.cls)
        conf = float(box.conf.item() if hasattr(box.conf, "item") else box.conf)
        x1, y1, x2, y2 = map(float, box.xyxy[0].detach().cpu().numpy().flatten())
        items.append(DetectItem(cls_id, x1, y1, x2, y2, conf))
    return items
