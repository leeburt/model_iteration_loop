"""YOLO pose 标签读写与格式转换。

处理 YOLO 关键点检测标签的归一化坐标 ↔ 像素坐标转换，
以及 ultralytics Results → PoseItem 的解析。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass
class PoseItem:
    """单个检测结果：类别、边界框（像素坐标）、关键点列表、置信度。"""

    cls_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    kpts: list[tuple[float, float, float]]
    conf: float = 1.0


def image_size(path: Path) -> tuple[int, int]:
    """读取图片尺寸，返回 (width, height)。"""
    with Image.open(path) as im:
        return im.size


def xywhn_to_item(parts: list[str], img_w: int, img_h: int, has_conf: bool) -> PoseItem | None:
    """将 YOLO 归一化标签 token 列表转为 PoseItem。

    Args:
        parts: 空格分隔的 token 列表，格式为 cls xc yc w h [kx ky v ...] [conf]。
        img_w, img_h: 图片宽高。
        has_conf: True 表示预测标签（末尾带 conf），False 表示 GT 标签。
    """
    if len(parts) < 5:
        return None
    cls_id = int(float(parts[0]))
    xc, yc, bw, bh = map(float, parts[1:5])
    cx = xc * img_w
    cy = yc * img_h
    w = bw * img_w
    h = bh * img_h
    rest = parts[5:]
    conf = 1.0
    if has_conf and len(rest) % 3 == 1:
        conf = float(rest[-1])
        rest = rest[:-1]

    kpts: list[tuple[float, float, float]] = []
    for i in range(0, len(rest), 3):
        if i + 2 >= len(rest):
            break
        kpts.append((float(rest[i]) * img_w, float(rest[i + 1]) * img_h, float(rest[i + 2])))

    return PoseItem(
        cls_id=cls_id,
        x1=cx - w / 2.0,
        y1=cy - h / 2.0,
        x2=cx + w / 2.0,
        y2=cy + h / 2.0,
        kpts=kpts,
        conf=conf,
    )


def read_pose_txt(path: Path, img_w: int, img_h: int, has_conf: bool = False) -> list[PoseItem]:
    """读取 YOLO pose 标签文件，返回 PoseItem 列表。文件不存在返回空列表。"""
    if not path.is_file():
        return []
    items: list[PoseItem] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            parts = raw.strip().split()
            if not parts:
                continue
            item = xywhn_to_item(parts, img_w, img_h, has_conf)
            if item is not None:
                items.append(item)
    return items


def write_pose_txt(path: Path, items: list[PoseItem], img_w: int, img_h: int) -> None:
    """将 PoseItem 列表写回 YOLO 归一化标签文件，保留 6 位小数。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            xc = (item.x1 + item.x2) / 2.0 / img_w
            yc = (item.y1 + item.y2) / 2.0 / img_h
            bw = (item.x2 - item.x1) / img_w
            bh = (item.y2 - item.y1) / img_h
            fields = [str(item.cls_id), f"{xc:.6f}", f"{yc:.6f}", f"{bw:.6f}", f"{bh:.6f}"]
            for x, y, v in item.kpts:
                fields.extend([f"{x / img_w:.6f}", f"{y / img_h:.6f}", f"{v:.6f}"])
            fields.append(f"{item.conf:.6f}")
            f.write(" ".join(fields) + "\n")


def result_to_items(result) -> list[PoseItem]:
    """将 ultralytics Results 对象转为 PoseItem 列表。

    从 result.boxes 和 result.keypoints 中提取检测框和关键点，
    转为像素坐标的 PoseItem。无检测时返回空列表。
    """
    items: list[PoseItem] = []
    if result.boxes is None:
        return items
    keypoints = result.keypoints if hasattr(result, "keypoints") and result.keypoints is not None else None
    for i, box in enumerate(result.boxes):
        cls_id = int(box.cls.item() if hasattr(box.cls, "item") else box.cls)
        conf = float(box.conf.item() if hasattr(box.conf, "item") else box.conf)
        x1, y1, x2, y2 = map(float, box.xyxy[0].detach().cpu().numpy().flatten())
        kpts: list[tuple[float, float, float]] = []
        if keypoints is not None and i < len(keypoints):
            kp_data = keypoints[i].data
            if hasattr(kp_data, "detach"):
                kp_data = kp_data.detach().cpu().numpy().flatten()
            else:
                kp_data = kp_data.flatten()
            for j in range(0, len(kp_data), 3):
                if j + 2 >= len(kp_data):
                    break
                kpts.append((float(kp_data[j]), float(kp_data[j + 1]), float(kp_data[j + 2])))
        items.append(PoseItem(cls_id, x1, y1, x2, y2, kpts, conf))
    return items
