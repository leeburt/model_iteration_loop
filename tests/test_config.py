from __future__ import annotations

import unittest

from yolo_iter.config import acceptance_config_from_project


class AcceptanceConfigTest(unittest.TestCase):
    def test_acceptance_profile_inherits_top_level_models_when_empty(self) -> None:
        cfg = acceptance_config_from_project(
            {
                "models": {
                    "candidate_model": "/models/candidate.pt",
                    "champion_model": "/models/champion.pt",
                },
                "acceptance_profiles": {
                    "full": {
                        "candidate_model": "",
                        "champion_model": "",
                        "eval_datasets": [],
                    }
                },
            },
            profile="full",
        )

        self.assertEqual(cfg["candidate_model"], "/models/candidate.pt")
        self.assertEqual(cfg["champion_model"], "/models/champion.pt")

    def test_acceptance_profile_model_values_override_top_level_models(self) -> None:
        cfg = acceptance_config_from_project(
            {
                "models": {
                    "candidate_model": "/models/default_candidate.pt",
                    "champion_model": "/models/default_champion.pt",
                },
                "acceptance_profiles": {
                    "full": {
                        "candidate_model": "/models/profile_candidate.pt",
                        "champion_model": "/models/profile_champion.pt",
                        "eval_datasets": [],
                    }
                },
            },
            profile="full",
        )

        self.assertEqual(cfg["candidate_model"], "/models/profile_candidate.pt")
        self.assertEqual(cfg["champion_model"], "/models/profile_champion.pt")


if __name__ == "__main__":
    unittest.main()
