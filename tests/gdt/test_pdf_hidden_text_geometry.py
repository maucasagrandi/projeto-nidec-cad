from __future__ import annotations

import fitz

from src.gdt.pdf_hidden_text_geometry import (
    PdfContentTextGeometryParser,
    ResolvedFontMetrics,
)


class _Resolver:
    def __init__(self, metrics=None):
        self.metrics = metrics

    def resolve(self, resource_name):
        if self.metrics is None:
            return None
        if resource_name == self.metrics.resource_name:
            return self.metrics
        return None


def _courier_metrics() -> ResolvedFontMetrics:
    return ResolvedFontMetrics(
        resource_name="F1",
        xref=38,
        basefont="CourierNewPSMT",
        encoding="WinAnsiEncoding",
        source="test_font",
        font=fitz.Font("cour"),
    )


def test_estimates_bbox_from_font_metrics_around_baseline() -> None:
    parser = PdfContentTextGeometryParser(
        font_resolver=_Resolver(_courier_metrics()),
        pdf_to_page=fitz.Matrix(1, 0, 0, 1, 0, 0),
    )
    stream = b"BT /F1 10 Tf 3 Tr 1 0 0 1 100 200 Tm (AB) Tj ET"

    events = parser.parse_stream(stream, xref=22)

    assert len(events) == 1
    event = events[0]
    assert event.text == "AB"
    assert event.invisible is True
    assert event.bbox_quality == "font_metrics"
    assert event.font_xref == 38
    assert event.pdf_origin == (100.0, 200.0)
    assert event.pdf_bbox[0] == 100.0
    assert event.pdf_bbox[2] > 100.0
    assert event.pdf_bbox[1] < 200.0 < event.pdf_bbox[3]
    assert event.page_bbox == event.pdf_bbox


def test_applies_page_transformation_matrix_to_quad() -> None:
    parser = PdfContentTextGeometryParser(
        font_resolver=_Resolver(_courier_metrics()),
        pdf_to_page=fitz.Matrix(1, 0, 0, -1, 0, 500),
    )
    stream = b"BT /F1 10 Tf 1 0 0 1 100 200 Tm (A) Tj ET"

    event = parser.parse_stream(stream, xref=1)[0]

    assert event.pdf_origin == (100.0, 200.0)
    assert event.page_origin == (100.0, 300.0)
    assert event.page_bbox[0] == event.pdf_bbox[0]
    assert event.page_bbox[2] == event.pdf_bbox[2]
    assert event.page_bbox[1] < 300.0 < event.page_bbox[3]


def test_advances_text_matrix_between_consecutive_tj_operators() -> None:
    parser = PdfContentTextGeometryParser(
        font_resolver=_Resolver(_courier_metrics()),
        pdf_to_page=fitz.Matrix(1, 0, 0, 1, 0, 0),
    )
    stream = b"BT /F1 10 Tf 1 0 0 1 10 20 Tm (A) Tj (B) Tj ET"

    events = parser.parse_stream(stream, xref=2)

    assert len(events) == 2
    assert events[0].pdf_origin == (10.0, 20.0)
    assert events[1].pdf_origin[0] > events[0].pdf_origin[0]
    expected_advance = events[0].width_text_space
    assert abs(events[1].pdf_origin[0] - (10.0 + expected_advance)) < 1e-6


def test_falls_back_to_explicit_heuristic_when_font_is_unresolved() -> None:
    parser = PdfContentTextGeometryParser(
        font_resolver=_Resolver(None),
        pdf_to_page=fitz.Matrix(1, 0, 0, 1, 0, 0),
    )
    stream = b"BT /Missing 10 Tf 1 0 0 1 0 0 Tm (ABC) Tj ET"

    event = parser.parse_stream(stream, xref=3)[0]

    assert event.bbox_quality == "fallback_heuristic"
    assert event.font_xref is None
    assert event.width_text_space == 18.0
    assert event.ascender_text_space == 8.0
    assert event.descender_text_space == -2.0
