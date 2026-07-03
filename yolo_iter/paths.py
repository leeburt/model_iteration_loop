"""数据集路径解析工具。

处理 YOLO 格式数据集中 images/ 到 labels/ 的路径映射，
以及 data.yaml 中 split 路径的解析。
"""

from __future__ import annotations

from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def to_label_dir(img_dir: Path) -> Path:
    """从 images 目录路径推导对应的 labels 目录路径。

    支持三种模式：
    1. 路径以 /images 结尾 → 替换为 /labels
    2. 路径中包含 /images/ → 替换为 /labels/
    3. 路径中 parts 包含 "images" → 替换为 "labels"
    """
    s = str(img_dir)
    if s.endswith("/images"):
        return Path(s[: -len("/images")] + "/labels")
    replaced = s.replace("/images/", "/labels/")
    if replaced != s:
        return Path(replaced)
    parts = img_dir.parts
    try:
        idx = parts.index("images")
    except ValueError as exc:
        raise ValueError(f"Cannot infer labels dir from image dir: {img_dir}") from exc
    new_parts = list(parts)
    new_parts[idx] = "labels"
    return Path(*new_parts)


def resolve_dataset_paths(data_yaml: dict, split: str) -> list[tuple[Path, Path]]:
    """解析 data.yaml 中指定 split 的 (image_dir, label_dir) 路径对列表。

    Args:
        data_yaml: YOLO 格式 data.yaml 的解析结果。
        split: 目标 split 名，如 "train"、"val"、"test"。

    Returns:
        (image_dir, label_dir) 元组列表，均为 resolve 后的绝对路径。
    """
    if split not in data_yaml:
        return []
    root = Path(data_yaml.get("path", ""))
    values = data_yaml[split]
    if isinstance(values, (str, Path)):
        values = [values]
    if not isinstance(values, list):
        raise ValueError(f"Split {split} must be a path or list of paths")

    pairs: list[tuple[Path, Path]] = []
    for raw in values:
        img_dir = Path(raw)
        if not img_dir.is_absolute():
            img_dir = root / img_dir
        img_dir = img_dir.resolve()
        pairs.append((img_dir, to_label_dir(img_dir).resolve()))
    return pairs


def collect_images(image_dir: Path) -> list[Path]:
    """收集目录下所有图片文件路径，按文件名排序。仅匹配 IMG_EXTS 中的扩展名。"""
    if not image_dir.is_dir():
        return []
    return sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)
