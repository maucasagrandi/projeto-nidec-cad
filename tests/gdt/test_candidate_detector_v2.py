from __future__ import annotations

import fitz

from src.gdt.candidate_detector_v2 import (
    RawProposal,
    audit_and_normalize_vector_primitives,
    validate_proposals,
)
from src.gdt.detector import BBox, GdtCell, GdtFrameCandidate


class _Score:
    def __init__(self, best_class: str, best_score: float, second_score: float):
        self.best_class = best_class
        self.best_score = best_score
        self.second_score = second_score

    def to_dict(self):
        return {
            "best_class": self.best_class,
            "best_score": self.best_score,
            "class_scores": {
                self.best_class: self.best_score,
                "other": self.second_score,
            },
        }


def _candidate(cid: str, texts: list[list[str]]) -> GdtFrameCandidate:
    cells = []
    for i, row in enumerate(texts):
        cells.append(GdtCell(BBox(i * 10.0, 0.0, (i + 1) * 10.0, 10.0), texts=row))
    return GdtFrameCandidate(
        candidate_id=cid,
        page=1,
        frame_bbox=BBox(0.0, 0.0, len(cells) * 10.0, 10.0),
        symbol_bbox=cells[0].bbox,
        cells=cells,
    )


def test_vector_normalizer_reads_line_and_rectangle_primitives():
    doc = fitz.open()
    page = doc.new_page(width=200, height=200)
    shape = page.new_shape()
    shape.draw_line((20, 20), (60, 20))
    shape.draw_rect(fitz.Rect(80, 40, 120, 60))
    shape.finish(color=(0, 0, 0))
    shape.commit()

    h, v, audit = audit_and_normalize_vector_primitives(page)

    assert len(h) >= 3
    assert len(v) >= 2
    assert audit["primitive_counts"].get("l", 0) >= 1
    assert audit["primitive_counts"].get("re", 0) >= 1


def test_validator_accepts_symbol_plus_tolerance_content():
    proposal = RawProposal(
        proposal_id="GDT-V2-P01-001",
        page=1,
        frame_bbox=BBox(0, 0, 30, 10),
        cell_bboxes=[BBox(0, 0, 10, 10), BBox(10, 0, 20, 10), BBox(20, 0, 30, 10)],
        symbol_bbox=BBox(0, 0, 10, 10),
    )
    cand = _candidate(proposal.proposal_id, [[], ["0,04"], ["A"]])
    accepted, rejected = validate_proposals([proposal], [cand], [_Score("perpendicularity", 0.80, 0.40)])

    assert len(accepted) == 1
    assert rejected == []
    assert proposal.validation_status == "accepted_for_downstream"


def test_validator_rejects_table_like_crop_without_symbol_support():
    proposal = RawProposal(
        proposal_id="GDT-V2-P01-001",
        page=1,
        frame_bbox=BBox(0, 0, 40, 10),
        cell_bboxes=[BBox(0, 0, 10, 10), BBox(10, 0, 20, 10), BBox(20, 0, 30, 10), BBox(30, 0, 40, 10)],
        symbol_bbox=BBox(0, 0, 10, 10),
    )
    cand = _candidate(proposal.proposal_id, [["BY"], ["AAP"], ["NXG"], ["HC"]])
    accepted, rejected = validate_proposals([proposal], [cand], [_Score("symmetry", 0.18, 0.17)])

    assert accepted == []
    assert len(rejected) == 1
    assert "FIRST_CELL_NOT_GDT_LIKE" in proposal.rejection_reasons
