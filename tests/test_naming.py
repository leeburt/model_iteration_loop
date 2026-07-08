from __future__ import annotations

import unittest
from pathlib import Path


class NamingTest(unittest.TestCase):
    def test_detection_training_entrypoint_uses_generic_train_name(self) -> None:
        self.assertTrue(Path("scripts/train.py").is_file())
        self.assertFalse(Path("train.py").exists())
        self.assertFalse(Path("train_junction.py").exists())
        content = Path("scripts/train.py").read_text(encoding="utf-8")
        self.assertNotIn("train_junction", content)
        self.assertNotIn("junction_train_data", content)

    def test_detection_config_uses_generic_filename(self) -> None:
        self.assertTrue(Path("configs/detection.yaml").is_file())
        self.assertFalse(Path("configs/junction_detection.yaml").exists())


if __name__ == "__main__":
    unittest.main()
