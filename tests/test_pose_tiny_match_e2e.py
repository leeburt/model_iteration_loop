from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from yolo_iter.manifest import build_dataset_manifest
from yolo_iter.pose_tiny_match import TinyMatchConfig, evaluate_split


def write_label(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


class PoseTinyMatchE2ETest(unittest.TestCase):
    def test_manifest_eval_and_recheck(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            img_dir = root / "images" / "val"
            lbl_dir = root / "labels" / "val"
            pred_dir = root / "pred"
            out_dir = root / "runs" / "case"
            img_dir.mkdir(parents=True)
            lbl_dir.mkdir(parents=True)
            pred_dir.mkdir(parents=True)

            Image.new("RGB", (100, 100), "white").save(img_dir / "tp.png")
            Image.new("RGB", (100, 100), "white").save(img_dir / "fn.png")
            Image.new("RGB", (100, 100), "white").save(img_dir / "fp.png")

            # YOLO pose: cls xc yc w h kx ky v [conf for predictions]
            write_label(lbl_dir / "tp.txt", ["0 0.500000 0.500000 0.100000 0.100000 0.500000 0.500000 2"])
            write_label(lbl_dir / "fn.txt", ["0 0.500000 0.500000 0.100000 0.100000 0.500000 0.500000 2"])
            write_label(lbl_dir / "fp.txt", [])

            write_label(pred_dir / "tp.txt", ["0 0.500000 0.500000 0.100000 0.100000 0.500000 0.500000 2 0.900000"])
            write_label(pred_dir / "fn.txt", [])
            write_label(pred_dir / "fp.txt", ["0 0.200000 0.200000 0.100000 0.100000 0.200000 0.200000 2 0.800000"])

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

            manifest = build_dataset_manifest(data_yaml, "tiny")
            self.assertEqual(manifest["image_count"], 3)
            self.assertEqual(manifest["object_count"], 2)

            result = evaluate_split(
                data_path=data_yaml,
                dataset_name="tiny",
                split="val",
                output_dir=out_dir,
                cfg=TinyMatchConfig(save_diff=True),
                pred_label_dirs=[pred_dir],
                model_role="candidate",
            )
            summary = result["summary"]
            self.assertEqual(summary["totals"]["tp"], 1)
            self.assertEqual(summary["totals"]["fp"], 1)
            self.assertEqual(summary["totals"]["fn"], 1)
            self.assertAlmostEqual(summary["metrics"]["f1"], 0.5)
            self.assertTrue((out_dir / "metrics" / "candidate_tiny_val_summary.json").exists())
            self.assertTrue((out_dir / "diff" / "candidate" / "tiny" / "val" / "images" / "fn.png").exists())

            # Simulate label fix: remove the FN object and add an FP target at the predicted point.
            write_label(lbl_dir / "fn.txt", [])
            write_label(lbl_dir / "fp.txt", ["0 0.200000 0.200000 0.100000 0.100000 0.200000 0.200000 2"])
            review_dir = root / "runs" / "review"
            review = evaluate_split(
                data_path=data_yaml,
                dataset_name="tiny",
                split="val",
                output_dir=review_dir,
                cfg=TinyMatchConfig(save_diff=True),
                pred_label_dirs=[pred_dir],
                model_role="candidate",
            )
            self.assertEqual(review["summary"]["totals"]["tp"], 2)
            self.assertEqual(review["summary"]["totals"]["fp"], 0)
            self.assertEqual(review["summary"]["totals"]["fn"], 0)


if __name__ == "__main__":
    unittest.main()
