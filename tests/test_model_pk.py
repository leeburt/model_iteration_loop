from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from yolo_iter.detect_match import DetectMatchConfig
from yolo_iter.model_pk import run_model_pk_from_prediction_dirs


def write_label(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


class ModelPkTest(unittest.TestCase):
    def test_model_pk_uses_champion_predictions_as_pseudo_gt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            img_dir = root / "images" / "val"
            img_dir.mkdir(parents=True)
            candidate_dir = root / "candidate"
            champion_dir = root / "champion"
            out_dir = root / "runs" / "pk"

            for name in ("tp", "fp", "fn"):
                Image.new("RGB", (100, 100), "white").save(img_dir / f"{name}.png")

            write_label(
                champion_dir / "tp.txt",
                ["0 0.500000 0.500000 0.100000 0.100000 0.500000 0.500000 2 0.900000"],
            )
            write_label(champion_dir / "fp.txt", [])
            write_label(
                champion_dir / "fn.txt",
                ["0 0.500000 0.500000 0.100000 0.100000 0.500000 0.500000 2 0.850000"],
            )

            write_label(
                candidate_dir / "tp.txt",
                ["0 0.500000 0.500000 0.100000 0.100000 0.500000 0.500000 2 0.800000"],
            )
            write_label(
                candidate_dir / "fp.txt",
                ["0 0.200000 0.200000 0.100000 0.100000 0.200000 0.200000 2 0.700000"],
            )
            write_label(candidate_dir / "fn.txt", [])

            data_yaml = root / "data.yaml"
            data_yaml.write_text(
                "\n".join(
                    [
                        f"path: {root}",
                        "val: images/val",
                        "nc: 1",
                        "kpt_shape: [1, 3]",
                        "names:",
                        "  0: in_line",
                    ]
                ),
                encoding="utf-8",
            )

            result = run_model_pk_from_prediction_dirs(
                data_path=data_yaml,
                dataset_name="tiny",
                split="val",
                output_dir=out_dir,
                candidate_pred_dir=candidate_dir,
                champion_pred_dir=champion_dir,
                save_visualizations=True,
            )

            summary = result["summary"]
            self.assertEqual(summary["totals"]["tp"], 1)
            self.assertEqual(summary["totals"]["fp"], 1)
            self.assertEqual(summary["totals"]["fn"], 1)
            self.assertAlmostEqual(summary["metrics"]["f1"], 0.5)
            self.assertTrue((out_dir / "metrics" / "model_pk_tiny_val_summary.json").exists())
            self.assertTrue((out_dir / "metrics" / "model_pk_tiny_val_per_image.csv").exists())
            self.assertTrue((out_dir / "visualizations" / "tiny" / "val" / "fp" / "images" / "fp.png").exists())
            self.assertTrue((out_dir / "visualizations" / "tiny" / "val" / "fn" / "images" / "fn.png").exists())

    def test_model_pk_can_use_detect_bbox_iou_matching(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            img_dir = root / "images" / "val"
            img_dir.mkdir(parents=True)
            candidate_dir = root / "candidate"
            champion_dir = root / "champion"
            out_dir = root / "runs" / "pk"

            Image.new("RGB", (100, 100), "white").save(img_dir / "low_iou.png")
            write_label(champion_dir / "low_iou.txt", ["0 0.500000 0.500000 0.400000 0.400000 0.900000"])
            write_label(candidate_dir / "low_iou.txt", ["0 0.500000 0.500000 0.020000 0.020000 0.800000"])

            data_yaml = root / "data.yaml"
            data_yaml.write_text(
                "\n".join(
                    [
                        f"path: {root}",
                        "val: images/val",
                        "nc: 1",
                        "names:",
                        "  0: target",
                    ]
                ),
                encoding="utf-8",
            )

            result = run_model_pk_from_prediction_dirs(
                data_path=data_yaml,
                dataset_name="detect",
                split="val",
                output_dir=out_dir,
                candidate_pred_dir=candidate_dir,
                champion_pred_dir=champion_dir,
                cfg=DetectMatchConfig(match_iou=0.5, show_progress=False),
                save_visualizations=True,
            )

            summary = result["summary"]
            self.assertEqual(summary["match"]["type"], "bbox_iou")
            self.assertEqual(summary["totals"]["tp"], 0)
            self.assertEqual(summary["totals"]["fp"], 1)
            self.assertEqual(summary["totals"]["fn"], 1)
            self.assertTrue((out_dir / "visualizations" / "detect" / "val" / "fp" / "images" / "low_iou.png").exists())


if __name__ == "__main__":
    unittest.main()
