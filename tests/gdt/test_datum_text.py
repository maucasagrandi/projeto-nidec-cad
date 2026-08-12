from __future__ import annotations

import fitz
import numpy as np

from src.gdt.datum_extractor import extract_datum_cells
from src.gdt.datum_text import (
    DatumTextCandidate,
    extract_datum_text_candidates,
    match_text_candidate_to_bbox,
)
from src.gdt.fcf_expander import FcfCell, FcfFrame


def _text_pdf() -> bytes:
    doc = fitz.open()
    page = doc.new_page(width=220, height=180)
    page.insert_text((54.0, 51.0), "A", fontsize=10)
    page.insert_text((100.0, 100.0), "NOTE", fontsize=10)
    data = doc.tobytes()
    doc.close()
    return data


def test_extracts_only_single_uppercase_text_with_page_bbox() -> None:
    candidates = extract_datum_text_candidates(_text_pdf(), page_index=0)

    assert [candidate.label for candidate in candidates] == ["A"]
    assert candidates[0].page == 1
    assert candidates[0].source == "pymupdf_word"
    assert candidates[0].bbox[0] < candidates[0].bbox[2]


def test_matches_letter_by_center_with_small_coordinate_padding() -> None:
    candidate = DatumTextCandidate(
        label="B",
        page=1,
        bbox=(19.5, 10.0, 22.5, 18.0),
        source="pdf_content_stream",
        confidence=0.9,
    )

    found = match_text_candidate_to_bbox(
        [candidate],
        (10.0, 8.0, 21.0, 20.0),
        padding_pt=1.5,
    )

    assert found is candidate


def test_vector_letter_resolves_fcf_datum_even_when_raster_crop_is_blank() -> None:
    cells = [
        FcfCell(10.0, 10.0, 22.0, 24.0, index=0, role="symbol"),
        FcfCell(22.0, 10.0, 52.0, 24.0, index=1, role="tolerance"),
        FcfCell(52.0, 10.0, 64.0, 24.0, index=2, role="datum"),
    ]
    frame = FcfFrame(10.0, 10.0, 64.0, 24.0, cells=cells, class_name="perpendicularity")
    candidate = DatumTextCandidate(
        label="A",
        page=1,
        bbox=(55.0, 12.0, 61.0, 22.0),
        source="pymupdf_word",
        confidence=1.0,
    )
    page_gray = np.full((100, 100), 255, dtype=np.uint8)

    extraction = extract_datum_cells(
        page_gray,
        1.0,
        [frame],
        text_candidates=[candidate],
    )[0]

    assert extraction.datum_texts == ["A"]
    assert extraction.has_datums is True
    assert extraction.datum_refs[0].has_content is True
    assert extraction.datum_refs[0].text_source == "pymupdf_word"
