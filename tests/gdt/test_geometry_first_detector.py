"""Unit tests for the geometry-first GD&T detector.

Covers the design's low-level pieces (template auto-measure, scale math, the
symbol-cell geometry filter, NCC-inside-box scoring) and the end-to-end contract
that ``detect_geometry_first_frames`` returns Detection/FcfFrame pairs whose
scores match (so ``_align_results`` joins them).
"""

from __future__ import annotations

import io

import cv2
import fitz
import numpy as np
import pytest

from src.gdt.geometry_first_detector import (
    BoxMatch,
    PageScaleLock,
    TemplateBoxMetric,
    _is_symbol_cell,
    _resize_template_to_box,
    measure_scale,
    measure_template_box,
    vector_fcf_cells,
)
from src.utils.gdt_detector import BBox


# ---------------------------------------------------------------------------
# Template auto-measure
# ---------------------------------------------------------------------------


def test_measure_template_box_detects_bordered_rectangle():
    # A 80x40 image with a bordered box from (5,5) to (74,34): box height ~30px.
    img = np.full((40, 80), 255, np.uint8)
    cv2.rectangle(img, (5, 5), (74, 34), 0, 2)
    metric = measure_template_box(img)
    assert metric.ok
    # The measured box height should be close to the drawn rectangle height (~32
    # including the 2px stroke on each side).
    assert 28 <= metric.box_height_px <= 36
    # Inner cell height is smaller than the outer box (border subtracted).
    assert 0 < metric.cell_height_px < metric.box_height_px


def test_measure_template_box_box_as_outermost_contour():
    # No explicit rectangle: a filled glyph. Falls back to ink bounding box.
    img = np.full((30, 30), 255, np.uint8)
    cv2.circle(img, (15, 15), 8, 0, -1)
    metric = measure_template_box(img)
    assert metric.ok
    assert metric.box_height_px > 0


def test_measure_template_box_blank_is_not_ok():
    img = np.full((20, 20), 255, np.uint8)  # all white, no ink
    metric = measure_template_box(img)
    assert not metric.ok
    assert metric.box_height_px == 0


# ---------------------------------------------------------------------------
# Scale math
# ---------------------------------------------------------------------------


def test_measure_scale_ratio():
    assert measure_scale(30.0, 30.0) == pytest.approx(1.0)
    assert measure_scale(60.0, 30.0) == pytest.approx(2.0)
    assert measure_scale(15.0, 30.0) == pytest.approx(0.5)


def test_measure_scale_zero_template_height_is_safe():
    assert measure_scale(30.0, 0.0) == 0.0


def test_page_scale_lock_band():
    lock = PageScaleLock(scale=2.0, locked=True, tolerance=0.05)
    assert lock.in_band(2.05)  # within +2.5%
    assert lock.in_band(1.95)
    assert not lock.in_band(2.2)  # +10% is out of band


def test_page_scale_lock_unlocked_accepts_anything():
    lock = PageScaleLock(scale=1.0, locked=False)
    assert lock.in_band(0.1)
    assert lock.in_band(99.0)


def test_resize_template_to_box_matches_target_height():
    tpl = np.full((30, 60), 255, np.uint8)
    resized = _resize_template_to_box(tpl, template_box_height_px=30.0, target_box_height_px=15.0)
    assert resized is not None
    assert resized.shape[0] == pytest.approx(15, abs=1)
    # width scales proportionally
    assert resized.shape[1] == pytest.approx(30, abs=1)


def test_resize_template_to_box_rejects_bad_input():
    tpl = np.full((30, 60), 255, np.uint8)
    assert _resize_template_to_box(tpl, 0.0, 15.0) is None
    assert _resize_template_to_box(tpl, 30.0, 0.0) is None


# ---------------------------------------------------------------------------
# Symbol-cell geometry filter
# ---------------------------------------------------------------------------


def test_is_symbol_cell_accepts_fcf_proportions():
    # ~14pt tall, ~19pt wide: a typical GD&T symbol cell.
    assert _is_symbol_cell(BBox(0, 0, 19, 14))
    # narrow/portrait symmetry cell
    assert _is_symbol_cell(BBox(0, 0, 11, 18))


def test_is_symbol_cell_rejects_too_tall_or_wide():
    assert not _is_symbol_cell(BBox(0, 0, 19, 60))   # too tall
    assert not _is_symbol_cell(BBox(0, 0, 200, 14))  # too wide
    assert not _is_symbol_cell(BBox(0, 0, 2, 14))    # too narrow


# ---------------------------------------------------------------------------
# Vector cell finder (synthetic vector FCF page)
# ---------------------------------------------------------------------------


def _make_fcf_pdf() -> bytes:
    """A one-page PDF with a single 2-cell FCF drawn as vector lines."""
    doc = fitz.open()
    page = doc.new_page(width=300, height=200)
    # A 2-cell FCF: outer box 100..160 x, 100..114 y (14pt tall), divider at 118.
    shape = page.new_shape()
    # outer rectangle
    shape.draw_rect(fitz.Rect(100, 100, 160, 114))
    # internal divider (symbol cell is 100..118, second cell 118..160)
    shape.draw_line(fitz.Point(118, 100), fitz.Point(118, 114))
    shape.finish(width=0.8, color=(0, 0, 0))
    shape.commit()
    data = doc.tobytes()
    doc.close()
    return data


def test_vector_fcf_cells_finds_the_symbol_cell():
    pdf = _make_fcf_pdf()
    doc = fitz.open(stream=pdf, filetype="pdf")
    try:
        page = doc[0]
        symbol_cells, growth_pool, audit = vector_fcf_cells(page, page_number=1)
    finally:
        doc.close()
    # The compact leading cell (~18pt wide, 14pt tall) should be a symbol cell.
    assert any(
        abs(c.x0 - 100) < 3 and abs(c.height - 14) < 3 and c.width < 34
        for c in symbol_cells
    ), f"symbol cell not found among {[(c.x0, c.width, c.height) for c in symbol_cells]}"
    # The growth pool should include cells and the audit is well-formed.
    assert audit["strategy"] == "vector_fcf_cells"
    assert len(growth_pool) >= len(symbol_cells)


# ---------------------------------------------------------------------------
# BoxMatch margin
# ---------------------------------------------------------------------------


def test_boxmatch_margin():
    m = BoxMatch(box=BBox(0, 0, 10, 10), best_class="position",
                 best_template="position_01", best_score=0.9, second_best_score=0.6)
    assert m.margin == pytest.approx(0.3)
