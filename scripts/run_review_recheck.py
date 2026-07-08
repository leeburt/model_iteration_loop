#!/usr/bin/env python3
"""人工修正标签后复核评估入口脚本。

复用原始验收的缓存预测结果，对修正后的标签重新计算指标。

用法: python scripts/run_review_recheck.py --run <acceptance_run_dir> --config configs/project.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yolo_iter.logging_utils import setup_logger
from yolo_iter.recheck import run_review_recheck


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Recompute metrics from cached YOLO predictions after label edits.")
    ap.add_argument("--run", required=True, help="Original acceptance run directory")
    ap.add_argument("--config", default="configs/project.yaml")
    ap.add_argument("--profile", default="full", help="acceptance profile in the unified project config")
    ap.add_argument("--review-name", default="review_after_label_fix")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run).expanduser().resolve()
    logger = setup_logger("review_recheck", run_dir / "logs" / "review_recheck.log")
    logger.info("Using run dir: %s", run_dir)
    logger.info("Using config: %s", Path(args.config).resolve())
    out = run_review_recheck(run_dir, args.config, args.review_name, profile=args.profile, logger=logger)
    logger.info("Review recheck complete: %s", out)


if __name__ == "__main__":
    main()
