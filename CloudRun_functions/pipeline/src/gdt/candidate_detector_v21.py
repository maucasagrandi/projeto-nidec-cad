"""GD&T Candidate Detector V2.1.

V2.1 keeps V2 proposal sources intact, then performs a geometry-only rectangle
refinement before symbol/content validation. The purpose is narrow: improve the
quality of the rectangular frame/cell chain that reaches the classifier without
reintroducing page-wide symbol-first search.
"""

from __future__ import annotations

from typing import Sequence

from src.gdt.candidate_detector_v2 import GdtCandidateDetectorV2, RawProposal
from src.gdt.detector import GdtFrameCandidate
from src.gdt.rectangle_refinement import refine_rectangle_proposals


class GdtCandidateDetectorV21:
    """V2 proposal generation + rectangle geometry refinement."""

    def __init__(self, *, raster_dpi: int = 220, rectangle_dpi: int = 220, min_geometry_score: float = 0.52):
        self.base = GdtCandidateDetectorV2(raster_dpi=raster_dpi)
        self.rectangle_dpi = int(rectangle_dpi)
        self.min_geometry_score = float(min_geometry_score)
        self.last_geometry_rejected: list[RawProposal] = []

    def propose(self, pdf_bytes: bytes, *, page_index: int = 0) -> tuple[list[RawProposal], dict]:
        raw, audit = self.base.propose(pdf_bytes, page_index=page_index)
        kept, rejected, geometry_audit = refine_rectangle_proposals(
            pdf_bytes,
            page_index,
            raw,
            dpi=self.rectangle_dpi,
            min_geometry_score=self.min_geometry_score,
        )
        self.last_geometry_rejected = list(rejected)
        audit = dict(audit)
        audit["v21_pre_rectangle_refinement"] = len(raw)
        audit["v21_geometry_rejected"] = len(rejected)
        audit["v21_post_rectangle_refinement"] = len(kept)
        audit.update(geometry_audit)
        return kept, audit

    def materialize(
        self,
        pdf_bytes: bytes,
        proposals: Sequence[RawProposal],
        *,
        page_index: int = 0,
        dpi: int = 300,
    ) -> list[GdtFrameCandidate]:
        return self.base.materialize(
            pdf_bytes,
            proposals,
            page_index=page_index,
            dpi=dpi,
        )


__all__ = ["GdtCandidateDetectorV21"]
