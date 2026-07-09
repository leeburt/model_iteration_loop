from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from yolo_iter.manifest import build_dataset_manifest
from yolo_iter.pose_io import PoseItem
from yolo_iter.pose_tiny_match import (
    TinyMatchConfig,
    build_visualization_plan,
    classify_items_for_visualization,
    evaluate_split,
    generate_comparison_visualizations,
    maybe_progress,
    render_prediction_panel,
    tiny_config_from_dict,
)


def write_label(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


class PoseTinyMatchE2ETest(unittest.TestCase):
    def test_tiny_config_enables_progress_by_default_and_honors_override(self) -> None:
        self.assertTrue(tiny_config_from_dict({}).show_progress)
        self.assertFalse(tiny_config_from_dict({"show_progress": False}).show_progress)

    def test_maybe_progress_wraps_iterable_when_enabled(self) -> None:
        calls = []

        def fake_tqdm(iterable, **kwargs):
            calls.append(kwargs)
            return iterable

        values = list(
            maybe_progress(
                [1, 2, 3],
                enabled=True,
                desc="candidate tiny val match",
                total=3,
                tqdm_factory=fake_tqdm,
            )
        )

        self.assertEqual(values, [1, 2, 3])
        self.assertEqual(calls[0]["desc"], "candidate tiny val match")
        self.assertEqual(calls[0]["total"], 3)

    def test_build_visualization_plan_compares_candidate_and_champion(self) -> None:
        candidate_rows = [
            {"image": "/tmp/a.png", "fp": 1, "fn": 0},
            {"image": "/tmp/b.png", "fp": 0, "fn": 1},
            {"image": "/tmp/c.png", "fp": 0, "fn": 0},
        ]
        champion_rows = [
            {"image": "/tmp/a.png", "fp": 0, "fn": 0},
            {"image": "/tmp/b.png", "fp": 0, "fn": 0},
            {"image": "/tmp/c.png", "fp": 1, "fn": 1},
        ]

        plan = build_visualization_plan(candidate_rows, champion_rows)

        self.assertEqual(plan["/tmp/a.png"], {"fp", "candidate_new_fp"})
        self.assertEqual(plan["/tmp/b.png"], {"fn", "candidate_new_fn"})
        self.assertEqual(plan["/tmp/c.png"], {"candidate_improved"})

    def test_classify_items_for_visualization_marks_tp_fp_and_fn(self) -> None:
        gt_items = [
            PoseItem(0, 10, 10, 20, 20, [(15, 15, 2)]),
            PoseItem(0, 50, 50, 60, 60, [(55, 55, 2)]),
        ]
        pred_items = [
            PoseItem(0, 10, 10, 20, 20, [(15, 15, 2)], conf=0.91),
            PoseItem(0, 80, 80, 90, 90, [(85, 85, 2)], conf=0.72),
        ]

        classified = classify_items_for_visualization(gt_items, pred_items, TinyMatchConfig())

        self.assertEqual(classified["tp_pred_indices"], {0})
        self.assertEqual(classified["fp_pred_indices"], {1})
        self.assertEqual(classified["fn_gt_indices"], {1})

    def test_pose_fn_visualization_uses_orange(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            img_path = root / "source.png"
            Image.new("RGB", (100, 100), "white").save(img_path)

            panel = render_prediction_panel(
                img_path,
                "candidate",
                gt_items=[PoseItem(0, 20, 20, 40, 40, [(30, 30, 2)])],
                pred_items=[],
                cfg=TinyMatchConfig(show_progress=False),
            )

            self.assertEqual(panel.getpixel((16, 50)), (255, 165, 0))
            self.assertNotEqual(panel.getpixel((16, 50)), (255, 0, 0))
            self.assertNotEqual(panel.getpixel((16, 50)), (255, 255, 0))

    def test_pose_fp_visualization_uses_red(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            img_path = root / "source.png"
            Image.new("RGB", (100, 100), "white").save(img_path)

            panel = render_prediction_panel(
                img_path,
                "candidate",
                gt_items=[],
                pred_items=[PoseItem(0, 20, 20, 40, 40, [(30, 30, 2)], conf=0.8)],
                cfg=TinyMatchConfig(show_progress=False),
            )

            self.assertEqual(panel.getpixel((18, 54)), (255, 0, 0))
            self.assertNotEqual(panel.getpixel((18, 54)), (255, 165, 0))
            self.assertNotEqual(panel.getpixel((18, 54)), (255, 255, 0))

    def test_evaluate_split_reports_empty_split_before_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images" / "train").mkdir(parents=True)
            data_yaml = root / "data.yaml"
            data_yaml.write_text(
                "\n".join(
                    [
                        f"path: {root}",
                        "train: images/train",
                        "nc: 1",
                        "kpt_shape: [1, 3]",
                        "names:",
                        "  0: in_line",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "No images found for dataset=tiny split=train"):
                evaluate_split(
                    data_path=data_yaml,
                    dataset_name="tiny",
                    split="train",
                    output_dir=root / "runs",
                    cfg=TinyMatchConfig(show_progress=False),
                    pred_label_dirs=[root / "pred"],
                    model_role="candidate",
                )

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
                cfg=TinyMatchConfig(save_diff=True, show_progress=False),
                pred_label_dirs=[pred_dir],
                model_role="candidate",
            )
            summary = result["summary"]
            self.assertEqual(summary["totals"]["tp"], 1)
            self.assertEqual(summary["totals"]["fp"], 1)
            self.assertEqual(summary["totals"]["fn"], 1)
            self.assertAlmostEqual(summary["metrics"]["f1"], 0.5)
            self.assertTrue((out_dir / "metrics" / "candidate_tiny_val_summary.json").exists())
            generate_comparison_visualizations(
                output_dir=out_dir,
                dataset_name="tiny",
                split="val",
                candidate_rows=result["rows"],
            )
            self.assertFalse((out_dir / "diff").exists())
            self.assertTrue((out_dir / "visualizations" / "tiny" / "val" / "fp" / "images" / "fp.png").exists())
            self.assertTrue((out_dir / "visualizations" / "tiny" / "val" / "fp" / "labels_gt" / "fp.txt").exists())
            self.assertTrue((out_dir / "visualizations" / "tiny" / "val" / "fp" / "labels_candidate" / "fp.txt").exists())
            self.assertTrue((out_dir / "visualizations" / "tiny" / "val" / "fp" / "compare" / "fp.png").exists())
            self.assertTrue((out_dir / "visualizations" / "tiny" / "val" / "fn" / "images" / "fn.png").exists())
            self.assertTrue((out_dir / "visualizations" / "tiny" / "val" / "fn" / "labels_gt" / "fn.txt").exists())
            self.assertTrue((out_dir / "visualizations" / "tiny" / "val" / "fn" / "labels_candidate" / "fn.txt").exists())
            self.assertTrue((out_dir / "visualizations" / "tiny" / "val" / "fn" / "compare" / "fn.png").exists())

            # Simulate label fix: remove the FN object and add an FP target at the predicted point.
            write_label(lbl_dir / "fn.txt", [])
            write_label(lbl_dir / "fp.txt", ["0 0.200000 0.200000 0.100000 0.100000 0.200000 0.200000 2"])
            review_dir = root / "runs" / "review"
            review = evaluate_split(
                data_path=data_yaml,
                dataset_name="tiny",
                split="val",
                output_dir=review_dir,
                cfg=TinyMatchConfig(save_diff=True, show_progress=False),
                pred_label_dirs=[pred_dir],
                model_role="candidate",
            )
            self.assertEqual(review["summary"]["totals"]["tp"], 2)
            self.assertEqual(review["summary"]["totals"]["fp"], 0)
            self.assertEqual(review["summary"]["totals"]["fn"], 0)


if __name__ == "__main__":
    unittest.main()
