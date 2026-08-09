from __future__ import annotations

import unittest
from dataclasses import dataclass

from src.gdt.geometry_validation import _bbox_iou, _bbox_overlap_metrics, match_ground_truth
from src.gdt.types import GroundTruthFrame


@dataclass
class _FakeBBox:
    values: tuple[float, float, float, float]

    def to_list(self):
        return list(self.values)


@dataclass
class _FakeCandidate:
    candidate_id: str
    page: int
    frame_bbox: _FakeBBox


class GeometryValidationTests(unittest.TestCase):
    def test_iou_identical_boxes_is_one(self):
        self.assertEqual(_bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)

    def test_iou_disjoint_boxes_is_zero(self):
        self.assertEqual(_bbox_iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)

    def test_overlap_metrics_identical_boxes(self):
        iou, overlap_smallest, area_ratio = _bbox_overlap_metrics(
            (0, 0, 10, 10), (0, 0, 10, 10)
        )
        self.assertEqual(iou, 1.0)
        self.assertEqual(overlap_smallest, 1.0)
        self.assertEqual(area_ratio, 1.0)

    def test_matching_is_one_to_one(self):
        gt = [
            GroundTruthFrame("GT-001", 1, "position", (0, 0, 10, 10)),
            GroundTruthFrame("GT-002", 1, "profile", (1, 1, 11, 11)),
        ]
        candidates = [
            _FakeCandidate("C-001", 1, _FakeBBox((0, 0, 10, 10))),
        ]

        metrics = match_ground_truth(gt, candidates, min_iou=0.35)

        self.assertEqual(metrics.true_positives, 1)
        self.assertEqual(metrics.false_negatives, 1)
        self.assertEqual(metrics.false_positives, 0)
        self.assertAlmostEqual(metrics.recall, 0.5)

    def test_manual_roi_overlap_can_match_when_iou_is_slightly_low(self):
        gt = [GroundTruthFrame("GT-001", 1, "profile", (0, 4, 12, 10))]
        candidates = [_FakeCandidate("C-001", 1, _FakeBBox((0, 0, 10, 8)))]

        metrics = match_ground_truth(
            gt,
            candidates,
            min_iou=0.35,
            min_overlap_smallest=0.50,
            max_area_ratio=2.50,
        )

        self.assertEqual(metrics.true_positives, 1)
        self.assertEqual(metrics.false_negatives, 0)
        self.assertEqual(metrics.matches[0].match_reason, "manual_roi_overlap")

    def test_small_nested_box_is_not_accepted_only_by_overlap(self):
        gt = [GroundTruthFrame("GT-001", 1, "profile", (0, 0, 100, 100))]
        candidates = [_FakeCandidate("C-001", 1, _FakeBBox((10, 10, 20, 20)))]

        metrics = match_ground_truth(
            gt,
            candidates,
            min_iou=0.35,
            min_overlap_smallest=0.50,
            max_area_ratio=2.50,
        )

        self.assertEqual(metrics.true_positives, 0)
        self.assertEqual(metrics.false_negatives, 1)

    def test_candidate_on_other_page_does_not_match(self):
        gt = [GroundTruthFrame("GT-001", 1, "position", (0, 0, 10, 10))]
        candidates = [_FakeCandidate("C-001", 2, _FakeBBox((0, 0, 10, 10)))]

        metrics = match_ground_truth(gt, candidates, min_iou=0.35)

        self.assertEqual(metrics.true_positives, 0)
        self.assertEqual(metrics.false_negatives, 1)
        self.assertEqual(metrics.false_positives, 1)

    def test_recall_gate(self):
        gt = [GroundTruthFrame("GT-001", 1, "position", (0, 0, 10, 10))]
        candidates = [_FakeCandidate("C-001", 1, _FakeBBox((0, 0, 10, 10)))]

        metrics = match_ground_truth(gt, candidates)

        self.assertTrue(metrics.passes_recall_gate(0.95))
        self.assertAlmostEqual(metrics.f1, 1.0)


if __name__ == "__main__":
    unittest.main()
