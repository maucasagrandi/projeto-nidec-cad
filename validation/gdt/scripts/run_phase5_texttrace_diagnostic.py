"""Diagnóstico de baixo nível para texto/glyphs dentro de células GD&T.

Motivação:
- page.get_text('words'/'rawdict') não encontrou texto nas células do caso 41;
- o usuário consegue selecionar/copiar conteúdo no leitor de PDF;
- portanto testamos APIs mais próximas do content stream: get_texttrace() e get_bboxlog().

Este script NÃO usa OCR/LLM e NÃO classifica o conteúdo. Ele apenas registra
operações de texto e suas posições para verificar se há glyphs textuais que o
caminho de extração comum não expôs como words/chars.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import fitz

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gdt.detector import GdtFrameDetector

CASE_ID = "case_41_rev8"
CASE_PATH = PROJECT_ROOT / "validation" / "gdt" / "cases" / f"{CASE_ID}.json"
GEOMETRY_BASELINE = PROJECT_ROOT / "validation" / "gdt" / "baselines" / f"{CASE_ID}.geometry.json"
OUTPUT_DIR = PROJECT_ROOT / "validation" / "gdt" / "outputs" / "phase5" / CASE_ID / "texttrace_diagnostic"
OUTPUT_PATH = OUTPUT_DIR / "texttrace.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rect_intersects_bbox(rect_like: Any, bbox) -> bool:
    try:
        rect = fitz.Rect(rect_like)
    except Exception:
        return False
    cell = fitz.Rect(bbox.to_list())
    intersection = rect & cell
    return not intersection.is_empty and intersection.width > 0 and intersection.height > 0


def _safe_json(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _safe_json(v) for k, v in value.items()}
    try:
        return list(value)
    except Exception:
        return str(value)


def _decode_trace_chars(span: dict) -> list[dict]:
    decoded: list[dict] = []
    for item in span.get("chars", []) or []:
        # PyMuPDF texttrace chars commonly expose: (unicode, glyph_id, origin, bbox)
        row: dict[str, Any] = {"raw": _safe_json(item)}
        if isinstance(item, (list, tuple)):
            if len(item) >= 1:
                codepoint = item[0]
                row["codepoint"] = codepoint
                if isinstance(codepoint, int):
                    try:
                        row["char"] = chr(codepoint)
                    except ValueError:
                        row["char"] = None
            if len(item) >= 2:
                row["glyph_id"] = item[1]
            if len(item) >= 3:
                row["origin"] = _safe_json(item[2])
            if len(item) >= 4:
                row["bbox"] = _safe_json(item[3])
        decoded.append(row)
    return decoded


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
        trace = page.get_texttrace()
        bboxlog = page.get_bboxlog()

        results = []
        for candidate in candidates:
            if candidate.candidate_id not in benchmark_ids:
                continue

            cells = []
            for index, cell in enumerate(candidate.cells):
                trace_rows = []
                for span in trace:
                    span_bbox = span.get("bbox")
                    chars = _decode_trace_chars(span)
                    char_hits = [
                        row for row in chars
                        if row.get("bbox") is not None and _rect_intersects_bbox(row["bbox"], cell.bbox)
                    ]
                    span_hit = span_bbox is not None and _rect_intersects_bbox(span_bbox, cell.bbox)
                    if not span_hit and not char_hits:
                        continue
                    trace_rows.append(
                        {
                            "font": span.get("font"),
                            "size": span.get("size"),
                            "type": span.get("type"),
                            "bbox": _safe_json(span_bbox),
                            "chars": char_hits if char_hits else chars,
                        }
                    )

                bboxlog_rows = []
                for order, entry in enumerate(bboxlog):
                    if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                        continue
                    kind, rect = entry[0], entry[1]
                    if not _rect_intersects_bbox(rect, cell.bbox):
                        continue
                    bboxlog_rows.append(
                        {
                            "order": order,
                            "kind": str(kind),
                            "bbox": _safe_json(rect),
                        }
                    )

                cells.append(
                    {
                        "cell_index": index,
                        "bbox": [round(v, 3) for v in cell.bbox.to_list()],
                        "texttrace": trace_rows,
                        "bboxlog": bboxlog_rows,
                    }
                )

            results.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "cell_count": len(candidate.cells),
                    "cells": cells,
                }
            )

        payload = {
            "schema_version": 1,
            "phase": "phase5_texttrace_diagnostic",
            "case_id": CASE_ID,
            "validation_status": "DIAGNOSTIC_ONLY",
            "ocr_used": False,
            "llm_used": False,
            "page_texttrace_span_count": len(trace),
            "page_bboxlog_count": len(bboxlog),
            "results": results,
        }

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        print("phase=phase5_texttrace_diagnostic")
        print("validation_status=DIAGNOSTIC_ONLY")
        print("ocr_used=False")
        print("llm_used=False")
        print(f"page_texttrace_spans={len(trace)}")
        print(f"page_bboxlog_entries={len(bboxlog)}")
        print("\nbenchmark_real_frame_texttrace:")
        for row in results:
            print(f"  {row['candidate_id']} cells={row['cell_count']}")
            for cell in row["cells"]:
                chars = []
                fonts = []
                for span in cell["texttrace"]:
                    if span.get("font"):
                        fonts.append(str(span["font"]))
                    for char in span.get("chars", []):
                        value = char.get("char")
                        if value is not None:
                            chars.append(value)
                kinds = [entry["kind"] for entry in cell["bboxlog"]]
                print(
                    f"    cell[{cell['cell_index']}] "
                    f"trace_spans={len(cell['texttrace'])} chars={chars!r} "
                    f"fonts={sorted(set(fonts))!r} bboxlog={kinds!r}"
                )

        print(f"\noutput={OUTPUT_PATH}")
    finally:
        doc.close()


if __name__ == "__main__":
    main()
