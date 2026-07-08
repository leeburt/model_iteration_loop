from __future__ import annotations

import unittest

from yolo_iter.detect_match import DetectMatchConfig
from yolo_iter.evaluation import backend_from_eval_protocol
from yolo_iter.pose_tiny_match import TinyMatchConfig


class EvaluationBackendTest(unittest.TestCase):
    def test_detect_protocol_selects_detect_backend(self) -> None:
        backend = backend_from_eval_protocol({"task": "detect", "metric": "bbox_iou_f1", "match_iou": 0.45})

        self.assertEqual(backend.task, "detect")
        self.assertIsInstance(backend.cfg, DetectMatchConfig)
        self.assertAlmostEqual(backend.cfg.match_iou, 0.45)

    def test_pose_protocol_keeps_tiny_match_backend(self) -> None:
        backend = backend_from_eval_protocol({"task": "pose", "metric": "tiny_match_f1", "kp_px": 4.0})

        self.assertEqual(backend.task, "pose")
        self.assertIsInstance(backend.cfg, TinyMatchConfig)
        self.assertAlmostEqual(backend.cfg.kp_px, 4.0)


if __name__ == "__main__":
    unittest.main()
