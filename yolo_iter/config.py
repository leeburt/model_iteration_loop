"""配置加载与解析。

从统一的 project.yaml 中提取训练配置和验收 profile 配置，
支持向后兼容独立的 train/acceptance 配置文件。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """读取 YAML 文件，返回 dict。根节点必须为 mapping 类型。"""
    cfg_path = Path(path).expanduser().resolve()
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {cfg_path}")
    return data


def write_yaml(path: str | Path, data: dict[str, Any]) -> None:
    """将 dict 写入 YAML 文件，自动创建父目录。"""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def require_path(value: str | Path, field: str) -> Path:
    """校验路径字段非空且存在，返回 resolve 后的绝对路径。"""
    if value is None or str(value).strip() == "":
        raise ValueError(f"Missing required path field: {field}")
    path = Path(value).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{field} does not exist: {path}")
    return path.resolve()


def train_config_from_project(project_config: dict[str, Any]) -> dict[str, Any]:
    """从 project.yaml 中提取 train 段；若无则向后兼容直接返回原配置。"""
    if "train" not in project_config:
        # Backward-compatible support for old dedicated train config files.
        return project_config
    return dict(project_config["train"])


def acceptance_config_from_project(project_config: dict[str, Any], profile: str = "full") -> dict[str, Any]:
    """从 project.yaml 中提取指定 acceptance profile，合并 eval_protocol。

    Args:
        project_config: 完整的 project.yaml 解析结果。
        profile: acceptance_profiles 中的 profile 名，默认 "full"。

    Returns:
        合并了 eval_protocol 的单个 profile 配置 dict。
    """
    if "acceptance_profiles" not in project_config:
        # Backward-compatible support for old dedicated acceptance config files.
        return project_config
    profiles = project_config.get("acceptance_profiles") or {}
    if profile not in profiles:
        available = ", ".join(sorted(profiles))
        raise KeyError(f"Unknown acceptance profile '{profile}'. Available profiles: {available}")
    cfg = dict(profiles[profile])
    cfg["eval"] = dict(project_config.get("eval_protocol") or {})
    if "train" in project_config:
        cfg["train"] = dict(project_config["train"])
    return cfg
