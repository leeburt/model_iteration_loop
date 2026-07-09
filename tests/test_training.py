from __future__ import annotations

import os
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from scripts.train import (
    DEFAULT_DATA,
    DEFAULT_MODEL,
    DEFAULT_NAME,
    DEFAULT_PROJECT,
    build_config,
)
from yolo_iter.training import get_resume_checkpoint


class TrainingTest(unittest.TestCase):
    def train_args(self, **overrides) -> Namespace:
        values = {
            "config": None,
            "model": DEFAULT_MODEL,
            "data": DEFAULT_DATA,
            "project": DEFAULT_PROJECT,
            "name": DEFAULT_NAME,
            "resume": "auto",
            "epochs": 20,
            "imgsz": 1280,
            "batch": 4,
            "device": "5",
            "workers": 8,
            "patience": 15,
            "optimizer": "AdamW",
            "lr0": 1e-4,
            "lrf": 0.01,
            "close_mosaic": 10,
            "log_dir": "runs/train_logs",
            "dry_run": False,
            "skip_manifest": False,
            "no_cache": False,
            "no_amp": False,
        }
        values.update(overrides)
        return Namespace(**values)

    def test_build_config_reads_train_section_from_project_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "detection.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "project:",
                        "  name: detection",
                        "train:",
                        "  data: /data/train.yaml",
                        "  initial_weights: /weights/best.pt",
                        "  project: /runs/train",
                        "  name: exp",
                        "  resume: auto",
                        "  args:",
                        "    task: detect",
                        "    epochs: 200",
                    ]
                ),
                encoding="utf-8",
            )

            config = build_config(self.train_args(config=str(config_path)))

            self.assertEqual(config["data"], "/data/train.yaml")
            self.assertEqual(config["initial_weights"], "/weights/best.pt")
            self.assertEqual(config["project"], "/runs/train")
            self.assertEqual(config["name"], "exp")
            self.assertEqual(config["args"]["task"], "detect")
            self.assertEqual(config["args"]["epochs"], 200)

    def test_build_config_command_line_overrides_yaml_train_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "detection.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "train:",
                        "  data: /data/train.yaml",
                        "  initial_weights: /weights/best.pt",
                        "  project: /runs/train",
                        "  name: exp",
                        "  args:",
                        "    task: detect",
                        "    epochs: 200",
                    ]
                ),
                encoding="utf-8",
            )

            config = build_config(self.train_args(config=str(config_path), data="/override.yaml", epochs=50, no_cache=True))

            self.assertEqual(config["data"], "/override.yaml")
            self.assertEqual(config["args"]["epochs"], 50)
            self.assertFalse(config["args"]["cache"])

    def test_resume_checkpoint_uses_task_specific_default_run_dir(self) -> None:
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "runs" / "detect" / "project" / "exp" / "weights" / "last.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint")
            os.chdir(root)
            try:
                self.assertEqual(get_resume_checkpoint("project", "exp", task="detect"), checkpoint)
            finally:
                os.chdir(original_cwd)


if __name__ == "__main__":
    unittest.main()
