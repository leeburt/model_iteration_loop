#!/usr/bin/env python3
"""指定模型和图片路径执行 YOLO pose 推理。

用法:
python scripts/predict_pose.py --model /path/to/best.pt --source /path/to/image_or_dir
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yolo_iter.logging_utils import setup_logger
from yolo_iter.pose_tiny_match import TinyMatchConfig
from yolo_iter.predict_pose import predict_pose_images


def default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("runs") / f"predict_pose_{stamp}"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run YOLO pose prediction on one image or an image directory.")
    ap.add_argument("--model", required=True, help="YOLO pose .pt model path")
    ap.add_argument("--source", required=True, help="image file or image directory")
    ap.add_argument("--output", default=None, help="output directory, default: runs/predict_pose_YYYYMMDD_HHMMSS")
    ap.add_argument("--device", default="0", help="CUDA device string, e.g. 0 or 0,1")
    ap.add_argument("--imgsz", type=int, default=1536, help="inference image size")
    ap.add_argument("--batch", type=int, default=1, help="inference batch size")
    ap.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    ap.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    ap.add_argument("--half", dest="half", action="store_true", default=True, help="use FP16 inference")
    ap.add_argument("--no-half", dest="half", action="store_false", help="disable FP16 inference")
    ap.add_argument("--no-visuals", action="store_true", help="do not save visualization images")
    ap.add_argument("--no-progress", action="store_true", help="disable progress bar")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output) if args.output else default_output_dir()
    logger = setup_logger("predict_pose", output_dir / "logs" / "run.log")
    logger.info("Model: %s", Path(args.model).expanduser())
    logger.info("Source: %s", Path(args.source).expanduser())
    logger.info("Output: %s", output_dir)

    cfg = TinyMatchConfig(
        device=args.device,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        nms_iou=args.iou,
        half=args.half,
        show_progress=not args.no_progress,
    )
    predict_pose_images(
        model_path=args.model,
        source=args.source,
        output_dir=output_dir,
        cfg=cfg,
        save_visualizations=not args.no_visuals,
    )
    logger.info("Prediction complete: %s", output_dir)


if __name__ == "__main__":
    main()
