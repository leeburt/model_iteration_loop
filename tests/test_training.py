from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from yolo_iter.training import get_resume_checkpoint


class TrainingTest(unittest.TestCase):
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
