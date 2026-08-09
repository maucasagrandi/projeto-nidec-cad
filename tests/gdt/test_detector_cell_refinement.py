from __future__ import annotations

from src.gdt.detector import BBox, GdtCell, GdtFrameDetector
from src.utils.gdt_detector import VSegment


def _cell(x0: float, x1: float) -> GdtCell:
    return GdtCell(BBox(x0, 0.0, x1, 10.0))


def test_symbol_stroke_is_not_used_as_cell_separator():
    detector = GdtFrameDetector(cell_endpoint_tolerance=1.0)
    frame = BBox(0.0, 0.0, 40.0, 10.0)

    # Segmentação permissiva antiga: a linha x=7 (traço do símbolo) virou
    # uma divisória falsa da primeira célula.
    original = [_cell(0, 7), _cell(7, 15), _cell(15, 30), _cell(30, 40)]

    verticals = [
        # Traço do símbolo: chega perto, mas NÃO atravessa o frame todo.
        VSegment(x=7.0, y0=-2.0, y1=8.4),
        # Divisórias reais: atravessam praticamente toda a altura.
        VSegment(x=15.0, y0=0.4, y1=9.8),
        VSegment(x=30.0, y0=0.2, y1=10.0),
    ]

    refined = detector._refine_cells_for_frame(frame, verticals, original)

    assert len(refined) == 3
    assert refined[0].bbox.to_list() == [0.0, 0.0, 15.0, 10.0]
    assert refined[1].bbox.to_list() == [15.0, 0.0, 30.0, 10.0]
    assert refined[2].bbox.to_list() == [30.0, 0.0, 40.0, 10.0]


def test_refinement_falls_back_when_no_plausible_cells_remain():
    detector = GdtFrameDetector(cell_endpoint_tolerance=1.0)
    frame = BBox(0.0, 0.0, 20.0, 10.0)
    original = [_cell(0, 10), _cell(10, 20)]

    # Nenhuma vertical interna atravessa o frame estritamente.
    verticals = [VSegment(x=10.0, y0=3.0, y1=7.0)]

    refined = detector._refine_cells_for_frame(frame, verticals, original)

    assert refined is original
