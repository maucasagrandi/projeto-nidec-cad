"""Diagnóstico da Fase 5B: inspeciona como o conteúdo interno do GD&T está codificado no PDF.

Objetivo:
- distinguir texto realmente extraível de glifos/paths vetoriais;
- NÃO usar OCR;
- NÃO usar LLM;
- NÃO classificar tolerância/datums ainda.

Para cada quadro real do caso 41, o script registra por célula:
- words do PyMuPDF;
- spans textuais;
- chars individuais;
- quantidade de drawings vetoriais que intersectam a célula;
- crop raster em alta resolução para inspeção visual.

Uso:
    python validation/gdt/scripts/run_phase5_primitive_diagnostic.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import fitz

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gdt.detector import GdtFrameDetector

CASE_ID = "case_41_rev8"
CASE_PATH = PROJECT_ROOT / "validation" / "gdt" / "cases" / f"{CASE_ID}.json"
GEOMETRY_BASELINE = PROJECT_ROOT / "validation" / "gdt" / "baselines" / f"{CASE_ID}.geometry.json"
OUTPUT_DIR = PROJECT_ROOT / "validation" / "gdt" / "outputs" / "phase5" / CASE_ID / "primitive_diagnostic"
OUTPUT_PATH = OUTPUT_DIR / "pdf_primitives.json"
CROP_DPI = 600


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _center_in_bbox(item_bbox, bbox) -> bool:
    x0, y0, x1, y1 = [float(v) for v in item_bbox]
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    return bbox.contains_point(cx, cy)


def _span_rows(page: fitz.Page) -> list[dict]:
    rows: list[dict] = []
    data = page.get_text("dict")
    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = str(span.get("text", ""))
                bbox = span.get("bbox")
                if bbox is not None:
                    rows.append({"text": text, "bbox": bbox})
    return rows


def _char_rows(page: fitz.Page) -> list[dict]:
    rows: list[dict] = []
    data = page.get_text("rawdict")
    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                for char in span.get("chars", []):
                    bbox = char.get("bbox")
                    if bbox is not None:
                        rows.append({"text": str(char.get("c", "")), "bbox": bbox})
    return rows


def _drawing_intersections(page: fitz.Page, bbox) -> tuple[int, int]:
    cell_rect = fitz.Rect(bbox.to_list())
    drawing_count = 0
    item_count = 0
    for drawing in page.get_drawings():
        rect = drawing.get("rect")
        if rect is None:
            continue
        rect = fitz.Rect(rect)
        intersection = rect & cell_rect
        if intersection.is_empty or intersection.width <= 0 or intersection.height <= 0:
            continue
        drawing_count += 1
        item_count += len(drawing.get("items", []))
    return drawing_count, item_count


def _render_cell(page: fitz.Page, bbox, output_path: Path) -> None:
    zoom = CROP_DPI / 72.0
    clip = fitz.Rect(bbox.to_list())
    pix = page.get_pixmap(
        matrix=fitz.Matrix(zoom, zoom),
        clip=clip,
        colorspace=fitz.csGRAY,
        alpha=False,
    )
    pix.save(str(output_path))


def main() -> None:
    case = _load(CASE_PATH)
    baseline = _load(GEOMETRY_BASELINE)
    pdf_path = PROJECT_ROOT / case["pdf"]
    page_index = int(case.get("page_index", 0))
    pdf_bytes = pdf_path.read_bytes()

    detector = GdtFrameDetector()
    candidates = detector.detect_frames(pdf_bytes, page_index=page_index)
    benchmark_ids = {
        row["candidate_id"]
        for row in baseline.get("matches", [])
        if row.get("candidate_id")
    }

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        page = doc[page_index]
        words = page.get_text("words")
        spans = _span_rows(page)
        chars = _char_rows(page)
        drawings = page.get_drawings()

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        results = []
        for candidate in candidates:
            if candidate.candidate_id not in benchmark_ids:
                continue

            cell_rows = []
            for index, cell in enumerate(candidate.cells):
                cell_words = [
                    str(word[4])
                    for word in words
                    if len(word) >= 5 and _center_in_bbox(word[:4], cell.bbox)
                ]
                cell_spans = [
                    row["text"]
                    for row in spans
                    if _center_in_bbox(row["bbox"], cell.bbox)
                ]
                cell_chars = [
                    row["text"]
                    for row in chars
                    if _center_in_bbox(row["bbox"], cell.bbox)
                ]
                drawing_count, drawing_item_count = _drawing_intersections(page, cell.bbox)

                crop_name = f"{candidate.candidate_id}_cell_{index:02d}.png"
                _render_cell(page, cell.bbox, OUTPUT_DIR / crop_name)

                cell_rows.append(
                    {
                        "cell_index": index,
                        "bbox": [round(v, 3) for v in cell.bbox.to_list()],
                        "words": cell_words,
                        "spans": cell_spans,
                        "chars": cell_chars,
                        "drawing_count": drawing_count,
                        "drawing_item_count": drawing_item_count,
                        "crop": crop_name,
                    }
                )

            results.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "frame_bbox": [round(v, 3) for v in candidate.frame_bbox.to_list()],
                    "cell_count": len(candidate.cells),
                    "cells": cell_rows,
                }
            )

        payload = {
            "schema_version": 1,
            "phase": "phase5_pdf_primitive_diagnostic",
            "case_id": CASE_ID,
            "validation_status": "DIAGNOSTIC_ONLY",
            "ocr_used": False,
            "llm_used": False,
            "page_word_count": len(words),
            "page_span_count": len(spans),
            "page_char_count": len(chars),
            "page_drawing_count": len(drawings),
            "benchmark_real_frame_count": len(results),
            "crop_dpi": CROP_DPI,
            "results": results,
        }
        OUTPUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        print("phase=phase5_pdf_primitive_diagnostic")
        print("validation_status=DIAGNOSTIC_ONLY")
        print("ocr_used=False")
        print("llm_used=False")
        print(f"page_words={len(words)}")
        print(f"page_spans={len(spans)}")
        print(f"page_chars={len(chars)}")
        print(f"page_drawings={len(drawings)}")
        print(f"benchmark_real_frames={len(results)}")
        print("\nbenchmark_real_frame_primitives:")
        for row in results:
            print(f"  {row['candidate_id']} cells={row['cell_count']}")
            for cell in row["cells"]:
                print(
                    f"    cell[{cell['cell_index']}] "
                    f"words={cell['words']} spans={cell['spans']} chars={cell['chars']} "
                    f"drawings={cell['drawing_count']} items={cell['drawing_item_count']} "
                    f"crop={cell['crop']}"
                )
        print(f"\noutput={OUTPUT_PATH}")
        print(f"crops={OUTPUT_DIR}")
    finally:
        doc.close()


if __name__ == "__main__":
    main()
