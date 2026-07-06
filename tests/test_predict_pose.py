from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from yolo_iter.pose_io import PoseItem
from yolo_iter.predict_pose import collect_source_images, write_prediction_outputs


class PredictPoseTest(unittest.TestCase):
    def test_collect_source_images_accepts_single_image_and_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            img_a = root / "a.png"
            img_b = root / "b.jpg"
            ignored = root / "notes.txt"
            Image.new("RGB", (64, 64), "white").save(img_a)
            Image.new("RGB", (64, 64), "white").save(img_b)
            ignored.write_text("not an image", encoding="utf-8")

            self.assertEqual(collect_source_images(img_a), [img_a.resolve()])
            self.assertEqual(collect_source_images(root), [img_a.resolve(), img_b.resolve()])

    def test_write_prediction_outputs_saves_labels_visuals_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            img_path = root / "source.png"
            out_dir = root / "predict"
            Image.new("RGB", (100, 100), "white").save(img_path)
            items_by_image = {
                img_path.resolve(): [
                    PoseItem(
                        cls_id=0,
                        x1=40.0,
                        y1=40.0,
                        x2=60.0,
                        y2=60.0,
                        kpts=[(50.0, 50.0, 2.0)],
                        conf=0.91,
                    )
                ]
            }

            rows = write_prediction_outputs(items_by_image, out_dir, save_visualizations=True)

            self.assertEqual(rows[0]["image"], str(img_path.resolve()))
            self.assertEqual(rows[0]["predictions"], 1)
            self.assertTrue((out_dir / "labels" / "source.txt").exists())
            self.assertTrue((out_dir / "visualizations" / "source.png").exists())
            self.assertTrue((out_dir / "summary.csv").exists())
            self.assertTrue((out_dir / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
