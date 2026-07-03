"""数据集清单生成与版本管理。

为每个数据集生成包含文件哈希、split 分布、数据泄漏检测的完整清单，
用于实验可追溯和版本对比。
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import load_yaml
from .paths import collect_images, resolve_dataset_paths


def utc_now() -> str:
    """返回当前 UTC 时间的 ISO 格式字符串。"""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def file_sha256(path: Path) -> str | None:
    """计算文件的 SHA256 哈希，文件不存在返回 None。"""
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def count_label_objects(label_path: Path) -> int:
    """统计 YOLO 标签文件中的目标数量（非空行数）。"""
    if not label_path.is_file():
        return 0
    with label_path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def normalize_names(names: Any) -> dict[str, str]:
    """将类别名统一为 {index_str: name_str} 格式。支持 dict 和 list 输入。"""
    if isinstance(names, dict):
        return {str(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {str(i): str(v) for i, v in enumerate(names)}
    return {}


def build_dataset_manifest(data_path: str | Path, dataset_name: str | None = None) -> dict[str, Any]:
    """构建数据集完整清单。

    遍历 data.yaml 中所有 split，记录每张图片和标签的文件哈希、
    目标数量，检测跨 split 数据泄漏，生成数据集版本指纹。

    Args:
        data_path: data.yaml 文件路径。
        dataset_name: 数据集名称，默认取 data.yaml 的 stem。

    Returns:
        包含 records、split_summary、leakage、dataset_version 等字段的 dict。
    """
    data_yaml_path = Path(data_path).expanduser().resolve()
    data = load_yaml(data_yaml_path)
    class_names = normalize_names(data.get("names", {}))
    records: list[dict[str, Any]] = []
    split_summary: dict[str, dict[str, Any]] = {}
    image_hash_to_splits: dict[str, set[str]] = {}
    path_to_splits: dict[str, set[str]] = {}

    for split in ("train", "val", "test"):
        pairs = resolve_dataset_paths(data, split)
        if not pairs:
            continue
        split_images = 0
        split_labels = 0
        split_objects = 0
        split_sources: list[str] = []
        for image_dir, label_dir in pairs:
            split_sources.append(str(image_dir))
            for image_path in collect_images(image_dir):
                label_path = label_dir / f"{image_path.stem}.txt"
                image_hash = file_sha256(image_path)
                label_hash = file_sha256(label_path)
                object_count = count_label_objects(label_path)
                rec = {
                    "image_path": str(image_path),
                    "label_path": str(label_path),
                    "image_hash": image_hash,
                    "label_hash": label_hash,
                    "split": split,
                    "source": str(image_dir),
                    "label_exists": label_path.is_file(),
                    "object_count": object_count,
                }
                records.append(rec)
                split_images += 1
                split_labels += int(label_path.is_file())
                split_objects += object_count
                if image_hash:
                    image_hash_to_splits.setdefault(image_hash, set()).add(split)
                path_to_splits.setdefault(str(image_path), set()).add(split)
        split_summary[split] = {
            "image_count": split_images,
            "label_count": split_labels,
            "object_count": split_objects,
            "sources": split_sources,
        }

    leakage = []
    for image_hash, splits in image_hash_to_splits.items():
        if len(splits) > 1:
            leakage.append({"type": "image_hash_in_multiple_splits", "image_hash": image_hash, "splits": sorted(splits)})
    for image_path, splits in path_to_splits.items():
        if len(splits) > 1:
            leakage.append({"type": "image_path_in_multiple_splits", "image_path": image_path, "splits": sorted(splits)})

    dataset_version = hashlib.sha256(
        json.dumps(
            {
                "data_yaml_hash": file_sha256(data_yaml_path),
                "records": [
                    {
                        "image_path": r["image_path"],
                        "image_hash": r["image_hash"],
                        "label_hash": r["label_hash"],
                        "split": r["split"],
                    }
                    for r in records
                ],
            },
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()[:16]

    return {
        "manifest_version": 1,
        "dataset_name": dataset_name or data_yaml_path.stem,
        "dataset_version": dataset_version,
        "created_at": utc_now(),
        "data_yaml_path": str(data_yaml_path),
        "data_yaml_hash": file_sha256(data_yaml_path),
        "class_names": class_names,
        "class_count": int(data.get("nc", len(class_names) or 0)),
        "kpt_shape": data.get("kpt_shape"),
        "image_count": len(records),
        "object_count": sum(int(r["object_count"]) for r in records),
        "split_summary": split_summary,
        "leakage": leakage,
        "records": records,
    }


def write_json(path: str | Path, data: Any) -> None:
    """将数据写入 JSON 文件，自动创建父目录。"""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def load_manifest(path: str | Path) -> dict[str, Any]:
    """从 JSON 文件加载 manifest。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))
