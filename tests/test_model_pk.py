from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from yolo_iter.detect_io import DetectItem
from yolo_iter.detect_match import DetectMatchConfig
from yolo_iter.model_pk import (
    build_model_class_context,
    collect_model_pk_sources,
    remap_items_to_class_context,
    run_model_pk_from_prediction_dirs,
)


def write_label(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


class ModelPkTest(unittest.TestCase):
    def test_model_pk_collects_single_image_source_without_dataset_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "one.png"
            Image.new("RGB", (100, 100), "white").save(image)

            sources = collect_model_pk_sources({"name": "single", "images": str(image), "names": {0: "target"}})

            self.assertEqual(len(sources), 1)
            self.assertEqual(sources[0]["dataset_name"], "single")
            self.assertEqual(sources[0]["split"], "images")
            self.assertEqual(sources[0]["images"], [image.resolve()])
            self.assertIsNone(sources[0]["data_path"])
            self.assertEqual(sources[0]["class_names"], {0: "target"})

    def test_model_pk_collects_image_directory_source_without_dataset_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new("RGB", (100, 100), "white").save(root / "b.jpg")
            Image.new("RGB", (100, 100), "white").save(root / "a.png")
            (root / "note.txt").write_text("ignored", encoding="utf-8")

            sources = collect_model_pk_sources({"name": "dir_case", "images": str(root), "split": "manual"})

            self.assertEqual([p.name for p in sources[0]["images"]], ["a.png", "b.jpg"])
            self.assertEqual(sources[0]["split"], "manual")

    def test_model_pk_remaps_model_class_ids_by_name_before_matching(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            img_dir = root / "images"
            candidate_dir = root / "candidate"
            champion_dir = root / "champion"
            out_dir = root / "runs" / "pk"
            img_dir.mkdir(parents=True)
            Image.new("RGB", (100, 100), "white").save(img_dir / "same_name.png")

            write_label(champion_dir / "same_name.txt", ["0 0.500000 0.500000 0.400000 0.400000 0.900000"])
            write_label(candidate_dir / "same_name.txt", ["1 0.500000 0.500000 0.400000 0.400000 0.800000"])

            result = run_model_pk_from_prediction_dirs(
                data_path=None,
                dataset_name="detect",
                split="images",
                images=[img_dir / "same_name.png"],
                output_dir=out_dir,
                candidate_pred_dir=candidate_dir,
                champion_pred_dir=champion_dir,
                cfg=DetectMatchConfig(match_iou=0.5, show_progress=False),
                save_visualizations=False,
                candidate_model_names={1: "junction"},
                champion_model_names={0: "junction"},
            )

            self.assertEqual(result["summary"]["totals"]["tp"], 1)
            self.assertEqual(result["summary"]["totals"]["fp"], 0)
            self.assertEqual(result["summary"]["totals"]["fn"], 0)

    def test_model_pk_keeps_different_model_class_names_unmatched(self) -> None:
        context = build_model_class_context(
            configured_names={},
            candidate_model_names={1: "shadow"},
            champion_model_names={0: "junction"},
        )
        champion_items = remap_items_to_class_context([DetectItem(0, 20, 20, 60, 60, 0.9)], context.champion_id_to_unified_id)
        candidate_items = remap_items_to_class_context([DetectItem(1, 20, 20, 60, 60, 0.8)], context.candidate_id_to_unified_id)

        self.assertEqual(context.class_names, {0: "junction", 1: "shadow"})
        self.assertEqual(champion_items[0].cls_id, 0)
        self.assertEqual(candidate_items[0].cls_id, 1)

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
