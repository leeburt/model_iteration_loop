from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from yolo_iter.detect_io import DetectItem, read_detect_txt, write_detect_txt
from yolo_iter.detect_match import (
    DetectMatchConfig,
    evaluate_split,
    format_item_label,
    generate_comparison_visualizations,
    render_prediction_panel,
)


def write_label(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


class DetectMatchTest(unittest.TestCase):
    def test_detect_labels_round_trip_with_prediction_confidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            label_path = Path(tmp) / "pred.txt"
            items = [DetectItem(cls_id=1, x1=20.0, y1=10.0, x2=60.0, y2=50.0, conf=0.72)]

            write_detect_txt(label_path, items, img_w=100, img_h=80)
            parsed = read_detect_txt(label_path, img_w=100, img_h=80, has_conf=True)

            self.assertEqual(len(parsed), 1)
            self.assertEqual(parsed[0].cls_id, 1)
            self.assertAlmostEqual(parsed[0].x1, 20.0)
            self.assertAlmostEqual(parsed[0].y1, 10.0)
            self.assertAlmostEqual(parsed[0].x2, 60.0)
            self.assertAlmostEqual(parsed[0].y2, 50.0)
            self.assertAlmostEqual(parsed[0].conf, 0.72)

    def test_evaluate_split_uses_bbox_iou_for_detect_tp_fp_fn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            img_dir = root / "images" / "val"
            lbl_dir = root / "labels" / "val"
            pred_dir = root / "pred"
            out_dir = root / "runs" / "case"
            img_dir.mkdir(parents=True)
            lbl_dir.mkdir(parents=True)
            pred_dir.mkdir(parents=True)

            for name in ("tp", "fp", "fn", "class_mismatch"):
                Image.new("RGB", (100, 100), "white").save(img_dir / f"{name}.png")

            write_label(lbl_dir / "tp.txt", ["0 0.500000 0.500000 0.200000 0.200000"])
            write_label(lbl_dir / "fp.txt", [])
            write_label(lbl_dir / "fn.txt", ["0 0.500000 0.500000 0.200000 0.200000"])
            write_label(lbl_dir / "class_mismatch.txt", ["0 0.500000 0.500000 0.200000 0.200000"])

            write_label(pred_dir / "tp.txt", ["0 0.500000 0.500000 0.200000 0.200000 0.900000"])
            write_label(pred_dir / "fp.txt", ["0 0.200000 0.200000 0.100000 0.100000 0.800000"])
            write_label(pred_dir / "fn.txt", [])
            write_label(pred_dir / "class_mismatch.txt", ["1 0.500000 0.500000 0.200000 0.200000 0.700000"])

            data_yaml = root / "data.yaml"
            data_yaml.write_text(
                "\n".join(
                    [
                        f"path: {root}",
                        "val: images/val",
                        "nc: 2",
                        "names:",
                        "  0: target",
                        "  1: other",
                    ]
                ),
                encoding="utf-8",
            )

            result = evaluate_split(
                data_path=data_yaml,
                dataset_name="detect",
                split="val",
                output_dir=out_dir,
                cfg=DetectMatchConfig(match_iou=0.5, save_diff=True, show_progress=False),
                pred_label_dirs=[pred_dir],
                model_role="candidate",
            )

            summary = result["summary"]
            self.assertEqual(summary["match"]["type"], "bbox_iou")
            self.assertEqual(summary["totals"]["tp"], 1)
            self.assertEqual(summary["totals"]["fp"], 2)
            self.assertEqual(summary["totals"]["fn"], 2)
            self.assertAlmostEqual(summary["metrics"]["f1"], 1 / 3)
            self.assertTrue((out_dir / "metrics" / "candidate_detect_val_summary.json").exists())

            generate_comparison_visualizations(
                output_dir=out_dir,
                dataset_name="detect",
                split="val",
                candidate_rows=result["rows"],
                cfg=DetectMatchConfig(match_iou=0.5, show_progress=False),
            )
            self.assertTrue((out_dir / "visualizations" / "detect" / "val" / "fp" / "images" / "fp.png").exists())
            self.assertTrue((out_dir / "visualizations" / "detect" / "val" / "fn" / "images" / "fn.png").exists())
            self.assertTrue((out_dir / "visualizations" / "detect" / "val" / "fp" / "compare" / "fp.png").exists())

    def test_fp_visualization_uses_red(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            img_path = root / "source.png"
            Image.new("RGB", (100, 100), "white").save(img_path)

            panel = render_prediction_panel(
                img_path,
                "candidate",
                gt_items=[],
                pred_items=[DetectItem(cls_id=0, x1=20.0, y1=20.0, x2=40.0, y2=40.0, conf=0.8)],
                cfg=DetectMatchConfig(show_progress=False),
            )

            self.assertEqual(panel.getpixel((18, 54)), (255, 0, 0))
            self.assertNotEqual(panel.getpixel((18, 54)), (255, 165, 0))
            self.assertNotEqual(panel.getpixel((18, 54)), (255, 255, 0))

    def test_fn_visualization_uses_orange(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            img_path = root / "source.png"
            Image.new("RGB", (100, 100), "white").save(img_path)

            panel = render_prediction_panel(
                img_path,
                "candidate",
                gt_items=[DetectItem(cls_id=0, x1=20.0, y1=20.0, x2=40.0, y2=40.0)],
                pred_items=[],
                cfg=DetectMatchConfig(show_progress=False),
            )

            self.assertEqual(panel.getpixel((16, 50)), (255, 165, 0))
            self.assertNotEqual(panel.getpixel((16, 50)), (255, 0, 0))
            self.assertNotEqual(panel.getpixel((16, 50)), (255, 255, 0))

    def test_detection_compare_labels_include_class_and_prediction_score(self) -> None:
        class_names = {0: "target", 1: "other"}

        gt_label = format_item_label(
            DetectItem(cls_id=0, x1=0.0, y1=0.0, x2=10.0, y2=10.0),
            class_names=class_names,
            include_conf=False,
        )
        pred_label = format_item_label(
            DetectItem(cls_id=1, x1=0.0, y1=0.0, x2=10.0, y2=10.0, conf=0.734),
            class_names=class_names,
            include_conf=True,
        )
        fallback_label = format_item_label(
            DetectItem(cls_id=7, x1=0.0, y1=0.0, x2=10.0, y2=10.0, conf=0.8),
            class_names=class_names,
            include_conf=True,
        )

        self.assertEqual(gt_label, "target")
        self.assertEqual(pred_label, "other 0.73")
        self.assertEqual(fallback_label, "cls_7 0.80")


if __name__ == "__main__":
    unittest.main()
