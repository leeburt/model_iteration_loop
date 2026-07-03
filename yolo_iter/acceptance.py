"""训练后自动验收流程编排。

完整验收流程：
1. 解析 project.yaml → 提取指定 acceptance profile
2. 创建时间戳 run 目录
3. 为每个数据集生成 manifest（版本指纹 + 数据泄漏检测）
4. 对每个 (model_role, dataset, split) 组合调用 evaluate_split
5. 汇总输出 overall_summary.json/csv
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import acceptance_config_from_project, load_yaml
from .logging_utils import add_file_handler
from .manifest import build_dataset_manifest, file_sha256, write_json
from .pose_tiny_match import evaluate_split, tiny_config_from_dict


def timestamp_run_id(prefix: str) -> str:
    """生成带时间戳的运行 ID：{safe_prefix}_{YYYYMMDD_HHMMSS}。"""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in prefix)
    return f"{safe}_{stamp}"


def make_run_dir(config: dict[str, Any]) -> Path:
    """根据配置创建运行输出目录。目录已存在时抛异常，防止覆盖历史结果。"""
    output_root = Path(config.get("output_root", "runs"))
    run_id = timestamp_run_id(str(config.get("run_id", "acceptance")))
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def default_candidate_from_train(config: dict[str, Any]) -> Path | None:
    """candidate_model 为空时，从 train.project/train.name 推导 best.pt。"""
    train_cfg = config.get("train") or {}
    project = train_cfg.get("project")
    name = train_cfg.get("name")
    if not project or not name:
        return None
    return (Path(project) / str(name) / "weights" / "best.pt").expanduser().resolve()


def resolve_model_paths(config: dict[str, Any]) -> dict[str, Path]:
    """解析 candidate/champion 模型路径；candidate 为空时自动使用当前训练 best.pt。"""
    resolved: dict[str, Path] = {}
    candidate = str(config.get("candidate_model") or "").strip()
    champion = str(config.get("champion_model") or "").strip()
    if candidate:
        resolved["candidate"] = Path(candidate).expanduser().resolve()
    else:
        default_candidate = default_candidate_from_train(config)
        if default_candidate is not None:
            resolved["candidate"] = default_candidate
            config["candidate_model"] = str(default_candidate)
            config["candidate_model_source"] = "train_best"
    if champion:
        resolved["champion"] = Path(champion).expanduser().resolve()
    return resolved


def model_roles(config: dict[str, Any]) -> list[tuple[str, Path]]:
    """解析配置中的 candidate/champion 模型，校验文件存在。

    Returns:
        [("candidate", path), ("champion", path)] 列表，至少包含一个。
    """
    resolved = resolve_model_paths(config)
    roles = list(resolved.items())
    if not roles:
        raise ValueError(
            "No model configured. Set candidate_model/champion_model, or provide train.project and train.name "
            "so candidate can default to train.project/train.name/weights/best.pt."
        )
    for role, path in roles:
        if not path.exists():
            raise FileNotFoundError(f"{role}_model does not exist: {path}")
    return roles


def write_run_manifest(run_dir: Path, config_path: Path, config: dict[str, Any], manifests: list[dict[str, Any]]) -> None:
    """生成运行级 manifest，记录配置哈希、模型哈希、各数据集版本信息。"""
    roles = {}
    for role_key in ("candidate_model", "champion_model"):
        value = str(config.get(role_key) or "").strip()
        if value:
            path = Path(value).expanduser().resolve()
            roles[role_key] = {"path": str(path), "sha256": file_sha256(path)}
    run_manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(config_path.resolve()),
        "config_hash": file_sha256(config_path),
        "models": roles,
        "eval": config.get("eval", {}),
        "datasets": [
            {
                "dataset_name": m["dataset_name"],
                "dataset_version": m["dataset_version"],
                "data_yaml_path": m["data_yaml_path"],
                "image_count": m["image_count"],
                "object_count": m["object_count"],
                "leakage_count": len(m.get("leakage", [])),
            }
            for m in manifests
        ],
    }
    write_json(run_dir / "manifests" / "run_manifest.json", run_manifest)


def write_overall_summary(run_dir: Path, results: list[dict[str, Any]]) -> None:
    """将所有 evaluate_split 结果汇总为 overall_summary.json 和 .csv。"""
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
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    write_json(metrics_dir / "overall_summary.json", rows)
    with (metrics_dir / "overall_summary.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "model_role",
            "dataset",
            "split",
            "images",
            "gt",
            "pred",
            "tp",
            "fp",
            "fn",
            "duplicate_fp",
            "real_fp",
            "precision",
            "recall",
            "f1",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_acceptance(config_path: str | Path, profile: str = "full", logger=None) -> Path:
    """执行完整验收流程，返回 run_dir。

    Args:
        config_path: project.yaml 路径。
        profile: acceptance_profiles 中的 profile 名。
        logger: 可选 logger，传入后会将日志写入 run_dir/logs/run.log。
    """
    cfg_path = Path(config_path).expanduser().resolve()
    config = acceptance_config_from_project(load_yaml(cfg_path), profile=profile)
    run_dir = make_run_dir(config)
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    if logger:
        add_file_handler(logger, log_dir / "run.log")
        logger.info("Acceptance run dir: %s", run_dir)

    manifests = []
    for ds in config.get("eval_datasets", []):
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

    write_run_manifest(run_dir, cfg_path, config, manifests)

    tiny_cfg = tiny_config_from_dict(config.get("eval", {}))
    roles = model_roles(config)
    results = []
    for role, model_path in roles:
        for ds in config.get("eval_datasets", []):
            dataset_name = ds["name"]
            data_path = Path(ds["data"]).expanduser().resolve()
            for split in ds.get("splits", ["val"]):
                if logger:
                    logger.info("Evaluating role=%s dataset=%s split=%s", role, dataset_name, split)
                result = evaluate_split(
                    data_path=data_path,
                    dataset_name=dataset_name,
                    split=split,
                    output_dir=run_dir,
                    cfg=tiny_cfg,
                    model_path=model_path,
                    model_role=role,
                )
                results.append(result)
                if logger:
                    logger.info("Result %s/%s/%s: %s", role, dataset_name, split, result["summary"]["metrics"])

    write_overall_summary(run_dir, results)
    return run_dir
