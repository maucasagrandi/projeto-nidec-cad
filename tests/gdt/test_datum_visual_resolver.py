from __future__ import annotations

import fitz
import numpy as np

from src.gdt.datum_extractor import DatumRef, FcfExtraction
from src.gdt.datum_text import extract_datum_text_candidates
from src.gdt.datum_visual_resolver import resolve_outlined_datum_references
from src.gdt.fcf_expander import FcfCell, FcfFrame


def _single_letter_pdf(letter: str) -> bytes:
    document = fitz.open()
    page = document.new_page(width=200, height=120)
    page.insert_text((50, 65), letter, fontsize=18, fontname="helv")
    pdf_bytes = document.tobytes()
    document.close()
    return pdf_bytes


def _render_candidate_crop(pdf_bytes: bytes) -> np.ndarray:
    candidate = extract_datum_text_candidates(pdf_bytes, page_index=0)[0]
    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = document[0]
    zoom = 300.0 / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csGRAY)
    gray = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    x0, y0, x1, y1 = candidate.bbox
    pad = 3.0
    crop = gray[
        int((y0 - pad) * zoom) : int((y1 + pad) * zoom),
        int((x0 - pad) * zoom) : int((x1 + pad) * zoom),
    ].copy()
    document.close()
    return crop


def test_resolves_outlined_cell_from_pdf_text_template() -> None:
    pdf_bytes = _single_letter_pdf("B")
    crop = _render_candidate_crop(pdf_bytes)
    cell = FcfCell(0, 0, 12, 14, index=2, role="datum")
    frame = FcfFrame(0, 0, 36, 14, cells=[cell], class_name="perpendicularity")
    datum_ref = DatumRef(cell=cell, crop=crop, glyph=crop, ink_ratio=0.1)
    extraction = FcfExtraction(frame=frame, datum_refs=[datum_ref])

    stats = resolve_outlined_datum_references(pdf_bytes, [extraction], [])

    assert datum_ref.text == "B"
    assert datum_ref.text_source == "visual_template_from_pdf_text"
    assert datum_ref.confidence >= 0.72
    assert stats.resolved_count == 1


def test_blank_cell_is_rejected_instead_of_forcing_a_label() -> None:
    pdf_bytes = _single_letter_pdf("B")
    cell = FcfCell(0, 0, 12, 14, index=2, role="datum")
    frame = FcfFrame(0, 0, 36, 14, cells=[cell], class_name="perpendicularity")
    blank = np.full((50, 45), 255, dtype=np.uint8)
    datum_ref = DatumRef(cell=cell, crop=blank, glyph=blank, ink_ratio=0.1)
    extraction = FcfExtraction(frame=frame, datum_refs=[datum_ref])

    stats = resolve_outlined_datum_references(pdf_bytes, [extraction], [])

    assert datum_ref.text == ""
    assert not datum_ref.has_content
    assert stats.resolved_count == 0
    assert stats.rejected_count == 0
    assert stats.empty_cell_count == 1
