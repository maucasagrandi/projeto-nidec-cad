"""Diagnóstico de baixo nível para texto/glyphs dentro de células GD&T.

Motivação:
- page.get_text('words'/'rawdict') não encontrou texto nas células do caso 41;
- o usuário consegue selecionar/copiar conteúdo no leitor de PDF;
- get_texttrace() confirmou uma camada textual de baixo nível;
- agora filtramos SOMENTE glyphs cuja posição individual pertence à célula.

Este script NÃO usa OCR/LLM e NÃO classifica o conteúdo. Ele registra glyphs,
posições e operações do content stream para descobrir se números/datums podem
ser recuperados diretamente do PDF.
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


def _rect_center_in_bbox(rect_like: Any, bbox) -> bool:
    """Critério estrito: o centro do bbox do glyph precisa estar na célula."""

    try:
        rect = fitz.Rect(rect_like)
    except Exception:
        return False
    center = rect.tl + (rect.br - rect.tl) * 0.5
    return bbox.contains_point(float(center.x), float(center.y))


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
        # PyMuPDF texttrace chars: (unicode, glyph_id, origin, bbox)
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


def _dedupe_chars(rows: list[dict]) -> list[dict]:
    """Remove duplicatas exatas de glyph sobreposto sem alterar a ordem."""

    seen: set[tuple] = set()
    result: list[dict] = []
    for row in rows:
        bbox = row.get("bbox") or []
        bbox_key = tuple(round(float(v), 3) for v in bbox) if len(bbox) == 4 else tuple()
        key = (row.get("codepoint"), bbox_key)
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


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
                span_bbox_hits_without_chars = 0

                for span_index, span in enumerate(trace):
                    span_bbox = span.get("bbox")
                    chars = _decode_trace_chars(span)
                    char_hits = [
                        row
                        for row in chars
                        if row.get("bbox") is not None
                        and _rect_center_in_bbox(row["bbox"], cell.bbox)
                    ]

                    # Importante: NÃO fazer fallback para todos os chars do span.
                    # O bug anterior vinha exatamente daí: spans enormes tocavam a
                    # célula e o script atribuía o texto da página inteira a ela.
                    if not char_hits:
                        if span_bbox is not None and _rect_intersects_bbox(span_bbox, cell.bbox):
                            span_bbox_hits_without_chars += 1
                        continue

                    trace_rows.append(
                        {
                            "span_index": span_index,
                            "seqno": span.get("seqno"),
                            "font": span.get("font"),
                            "size": span.get("size"),
                            "type": span.get("type"),
                            "bbox": _safe_json(span_bbox),
                            "chars": char_hits,
                            "text": "".join(
                                row.get("char") or ""
                                for row in char_hits
                            ),
                        }
                    )

                all_char_hits = _dedupe_chars(
                    [
                        char
                        for span in trace_rows
                        for char in span.get("chars", [])
                    ]
                )
                reconstructed_text = "".join(
                    row.get("char") or ""
                    for row in all_char_hits
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
                        "glyph_count": len(all_char_hits),
                        "reconstructed_text": reconstructed_text,
                        "span_bbox_hits_without_chars": span_bbox_hits_without_chars,
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
            "schema_version": 2,
            "phase": "phase5_texttrace_diagnostic",
            "case_id": CASE_ID,
            "validation_status": "DIAGNOSTIC_ONLY",
            "ocr_used": False,
            "llm_used": False,
            "glyph_assignment_rule": "glyph_bbox_center_inside_cell",
            "page_rotation": int(page.rotation),
            "page_rect": [float(v) for v in page.rect],
            "page_cropbox": [float(v) for v in page.cropbox],
            "page_texttrace_span_count": len(trace),
            "page_bboxlog_count": len(bboxlog),
            "results": results,
        }

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        print("phase=phase5_texttrace_diagnostic_v2")
        print("validation_status=DIAGNOSTIC_ONLY")
        print("ocr_used=False")
        print("llm_used=False")
        print("glyph_assignment_rule=glyph_bbox_center_inside_cell")
        print(f"page_rotation={page.rotation}")
        print(f"page_texttrace_spans={len(trace)}")
        print(f"page_bboxlog_entries={len(bboxlog)}")
        print("\nbenchmark_real_frame_texttrace:")

        for row in results:
            print(f"  {row['candidate_id']} cells={row['cell_count']}")
            for cell in row["cells"]:
                span_texts = [
                    span.get("text", "")
                    for span in cell["texttrace"]
                ]
                fonts = sorted(
                    {
                        str(span["font"])
                        for span in cell["texttrace"]
                        if span.get("font")
                    }
                )
                kinds = [entry["kind"] for entry in cell["bboxlog"]]
                print(
                    f"    cell[{cell['cell_index']}] "
                    f"glyphs={cell['glyph_count']} "
                    f"text={cell['reconstructed_text']!r} "
                    f"span_texts={span_texts!r} fonts={fonts!r} "
                    f"span_only_hits={cell['span_bbox_hits_without_chars']} "
                    f"bboxlog={kinds!r}"
                )

        print(f"\noutput={OUTPUT_PATH}")
    finally:
        doc.close()


if __name__ == "__main__":
    main()
