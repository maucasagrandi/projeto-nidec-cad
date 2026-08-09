from __future__ import annotations

import fitz

from src.gdt.pdf_hidden_text import PdfContentTextParser


def test_reconstructs_tm_td_tj_positions_and_hidden_mode() -> None:
    parser = PdfContentTextParser(pdf_to_page=fitz.Matrix(1, 0, 0, 1, 0, 0))
    stream = b"BT /F1 10 Tf 3 Tr 1 0 0 1 100 200 Tm (A) Tj 20 0 Td (B) Tj ET"

    events = parser.parse_stream(stream, xref=22)

    assert [event.text for event in events] == ["A", "B"]
    assert events[0].font_resource == "F1"
    assert events[0].font_size == 10.0
    assert events[0].rendering_mode == 3
    assert events[0].invisible is True
    assert events[0].pdf_origin == (100.0, 200.0)
    assert events[1].pdf_origin == (120.0, 200.0)


def test_td_is_applied_in_text_matrix_coordinates() -> None:
    parser = PdfContentTextParser()
    stream = b"BT 2 0 0 2 10 20 Tm 5 3 Td (X) Tj ET"

    events = parser.parse_stream(stream, xref=1)

    assert len(events) == 1
    assert events[0].pdf_origin == (20.0, 26.0)


def test_decodes_literal_escapes_and_hex_strings() -> None:
    parser = PdfContentTextParser()
    stream = b"BT 1 0 0 1 0 0 Tm (A\\050B\\051\\101) Tj <432E3035> Tj ET"

    events = parser.parse_stream(stream, xref=2)

    assert [event.text for event in events] == ["A(B)A", "C.05"]


def test_text_state_persists_across_page_content_streams() -> None:
    parser = PdfContentTextParser()

    first = b"/FHidden 7 Tf 3 Tr"
    second = b"BT 1 0 0 1 50 60 Tm (0.05) Tj ET"

    assert parser.parse_stream(first, xref=22) == []
    events = parser.parse_stream(second, xref=29)

    assert len(events) == 1
    assert events[0].text == "0.05"
    assert events[0].font_resource == "FHidden"
    assert events[0].font_size == 7.0
    assert events[0].rendering_mode == 3
    assert events[0].xref == 29


def test_tj_array_concatenates_only_string_parts() -> None:
    parser = PdfContentTextParser()
    stream = b"BT 1 0 0 1 10 10 Tm [(0) -20 (.05)] TJ ET"

    events = parser.parse_stream(stream, xref=3)

    assert len(events) == 1
    assert events[0].operator == "TJ"
    assert events[0].text == "0.05"
