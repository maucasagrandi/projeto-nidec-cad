from __future__ import annotations

import unittest

import fitz

from src.gdt.candidate_detector_v2 import RawProposal
from src.gdt.detector import BBox
from src.gdt.rectangle_refinement import refine_rectangle_proposals


def _proposal(pid: str, x0: float, y0: float, xs: list[float], y1: float, source: str) -> RawProposal:
    cells = [BBox(a, y0, b, y1) for a, b in zip(xs, xs[1:])]
    return RawProposal(
        proposal_id=pid,
        page=1,
        frame_bbox=BBox(x0, y0, xs[-1], y1),
        cell_bboxes=cells,
        symbol_bbox=cells[0],
        sources=[source],
    )


def _synthetic_pdf() -> bytes:
    doc = fitz.open()
    page = doc.new_page(width=400, height=250)

    # Isolated three-cell row: FCF-like rectangle geometry.
    for y in (40, 55):
        page.draw_line((40, y), (100, y), width=0.8)
    for x in (40, 55, 80, 100):
        page.draw_line((x, 40), (x, 55), width=0.8)

    # Larger 4x4 grid. A row-shaped sub-window is a perfect rectangle but it is
    # embedded in a table because the column lines continue into neighboring rows.
    for y in (40, 60, 80, 100, 120):
        page.draw_line((180, y), (300, y), width=0.8)
    for x in (180, 210, 240, 270, 300):
        page.draw_line((x, 40), (x, 120), width=0.8)

    payload = doc.tobytes()
    doc.close()
    return payload


class RectangleRefinementTests(unittest.TestCase):
    def test_isolated_cell_row_survives_but_table_subwindow_is_rejected(self):
        pdf = _synthetic_pdf()
        isolated = _proposal("ISO", 40, 40, [40, 55, 80, 100], 55, "vector_normalized_v2")
        table = _proposal("TABLE", 180, 60, [180, 210, 240, 270], 80, "raster_morphology_v2")

        kept, rejected, audit = refine_rectangle_proposals(
            pdf,
            0,
            [isolated, table],
            dpi=220,
        )

        self.assertEqual([p.proposal_id for p in kept], ["ISO"])
        self.assertEqual([p.proposal_id for p in rejected], ["TABLE"])
        self.assertIn("EMBEDDED_IN_LARGE_GRID", table.rejection_reasons)
        self.assertGreater(
            isolated.primitive_evidence["rectangle_geometry"]["geometry_score"],
            table.primitive_evidence["rectangle_geometry"]["geometry_score"],
        )
        self.assertEqual(audit["rectangle_refinement_input"], 2)
        self.assertEqual(audit["rectangle_refinement_kept"], 1)

    def test_geometry_evidence_is_recorded_without_tp_fp_semantics(self):
        pdf = _synthetic_pdf()
        isolated = _proposal("ISO", 40, 40, [40, 55, 80, 100], 55, "vector_normalized_v2")

        kept, _rejected, _audit = refine_rectangle_proposals(pdf, 0, [isolated], dpi=220)

        evidence = kept[0].primitive_evidence["rectangle_geometry"]
        self.assertIn("top_coverage", evidence)
        self.assertIn("divider_mean", evidence)
        self.assertIn("grid_penalty", evidence)
        self.assertIn("geometry_score", evidence)
        self.assertNotIn("true_positive", evidence)
        self.assertNotIn("false_positive", evidence)


if __name__ == "__main__":
    unittest.main()
