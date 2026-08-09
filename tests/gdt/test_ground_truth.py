from __future__ import annotations

import unittest

from src.gdt.ground_truth import build_ground_truth_payload


class GroundTruthBuilderTests(unittest.TestCase):
    def setUp(self):
        self.candidates = {
            "case_id": "case_x",
            "pdf": "drawing.pdf",
            "page": 1,
            "candidates": [
                {
                    "candidate_id": "C-001",
                    "page": 1,
                    "frame_bbox": [10, 20, 40, 30],
                },
                {
                    "candidate_id": "C-002",
                    "page": 1,
                    "frame_bbox": [50, 60, 90, 70],
                },
            ],
        }

    def test_builds_gt_from_reviewed_candidate(self):
        review = {
            "case_id": "case_x",
            "accepted_candidates": [
                {"candidate_id": "C-001", "characteristic": "position"}
            ],
            "manual_frames": [],
        }

        payload = build_ground_truth_payload(self.candidates, review)

        self.assertEqual(payload["expected_frame_count"], 1)
        self.assertEqual(payload["frames"][0]["bbox"], [10.0, 20.0, 40.0, 30.0])
        self.assertEqual(payload["frames"][0]["source_candidate_id"], "C-001")

    def test_manual_frame_keeps_detector_miss_in_ground_truth(self):
        review = {
            "accepted_candidates": [],
            "manual_frames": [
                {
                    "page": 1,
                    "characteristic": "profile",
                    "bbox": [100, 200, 140, 210],
                    "notes": "não proposto pelo detector",
                }
            ],
        }

        payload = build_ground_truth_payload(self.candidates, review)

        self.assertEqual(payload["expected_frame_count"], 1)
        self.assertEqual(payload["frames"][0]["source"], "manual_annotation")

    def test_unknown_candidate_is_rejected(self):
        review = {
            "accepted_candidates": [
                {"candidate_id": "C-999", "characteristic": "position"}
            ]
        }

        with self.assertRaises(ValueError):
            build_ground_truth_payload(self.candidates, review)

    def test_bootstrap_rejects_unplanned_characteristic(self):
        review = {
            "accepted_candidates": [
                {"candidate_id": "C-001", "characteristic": "flatness"}
            ]
        }

        with self.assertRaises(ValueError):
            build_ground_truth_payload(self.candidates, review)


if __name__ == "__main__":
    unittest.main()
