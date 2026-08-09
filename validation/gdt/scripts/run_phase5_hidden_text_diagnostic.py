"""Fase 5: associa a camada textual invisível do PDF às células GD&T.

Pré-condições já verificadas no caso 41:
- os quadros/células têm bbox geométrico conhecido;
- existe content stream com ``3 Tr`` + ``Tm/Td/Tj``;
- a extração comum do PyMuPDF não devolve words/chars úteis nessas células.

Este diagnóstico usa ``src.gdt.pdf_hidden_text`` para reconstruir a origem dos
``Tj/TJ`` diretamente do content stream e testa se essa origem cai dentro de
cada célula. Não usa OCR nem LLM e ainda NÃO é benchmark de acurácia de conteúdo.

Uso:
    python validation/gdt/scripts/run_phase5_hidden_text_diagnostic.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import fitz

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gdt.detector import GdtFrameDetector
from src.gdt.pdf_hidden_text import PdfTextEvent, extract_page_text_events

CASE_ID = "case_41_rev8"
CASE_PATH = PROJECT_ROOT / "validation" / "gdt" / "cases" / f"{CASE_ID}.json"
GEOMETRY_BASELINE = PROJECT_ROOT / "validation" / "gdt" / "baselines" / f"{CASE_ID}.geometry.json"
OUTPUT_DIR = PROJECT_ROOT / "validation" / "gdt" / "outputs" / "phase5" / CASE_ID / "hidden_text_diagnostic"
OUTPUT_PATH = OUTPUT_DIR / "hidden_text_cells.json"
PADDED_TOLERANCE_PT = 1.5
NEAREST_LIMIT = 5


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _contains(bbox, point, padding: float = 0.0) -> bool:
    x, y = point
    return (
        bbox.x0 - padding <= x <= bbox.x1 + padding
        and bbox.y0 - padding <= y <= bbox.y1 + padding
    )


def _distance_to_bbox(bbox, point) -> float:
    x, y = point
    dx = max(bbox.x0 - x, 0.0, x - bbox.x1)
    dy = max(bbox.y0 - y, 0.0, y - bbox.y1)
    return math.hypot(dx, dy)


def _event_row(event: PdfTextEvent) -> dict:
    return event.to_dict()


def _compact_event(event: PdfTextEvent) -> str:
    x, y = event.page_origin
    text = event.text.replace("\r", "\\r").replace("\n", "\\n")
    if len(text) > 40:
        text = text[:37] + "..."
    return f"{text!r}@({x:.1f},{y:.1f}) xref={event.xref} Tr={event.rendering_mode}"


def main() -> None:
    case = _load(CASE_PATH)
    baseline = _load(GEOMETRY_BASELINE)
    pdf_path = PROJECT_ROOT / case["pdf"]
    page_index = int(case.get("page_index", 0))
    pdf_bytes = pdf_path.read_bytes()

    benchmark_ids = {
        row["candidate_id"]
        for row in baseline.get("matches", [])
        if row.get("candidate_id")
    }

    detector = GdtFrameDetector()
    candidates = detector.detect_frames(pdf_bytes, page_index=page_index)

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        events = extract_page_text_events(doc, page_index=page_index)
    finally:
        doc.close()

    nonempty_events = [event for event in events if event.text.strip()]
    hidden_events = [event for event in nonempty_events if event.invisible]

    results = []
    for candidate in candidates:
        if candidate.candidate_id not in benchmark_ids:
            continue

        cell_rows = []
        for index, cell in enumerate(candidate.cells):
            strict = [
                event for event in nonempty_events
                if _contains(cell.bbox, event.page_origin, padding=0.0)
            ]
            padded = [
                event for event in nonempty_events
                if _contains(cell.bbox, event.page_origin, padding=PADDED_TOLERANCE_PT)
            ]
            strict_hidden = [event for event in strict if event.invisible]
            padded_hidden = [event for event in padded if event.invisible]

            nearest_hidden = sorted(
                hidden_events,
                key=lambda event: _distance_to_bbox(cell.bbox, event.page_origin),
            )[:NEAREST_LIMIT]

            cell_rows.append(
                {
                    "cell_index": index,
                    "bbox": [round(v, 3) for v in cell.bbox.to_list()],
                    "strict_text": "".join(event.text for event in strict),
                    "strict_hidden_text": "".join(event.text for event in strict_hidden),
                    "padded_text": "".join(event.text for event in padded),
                    "padded_hidden_text": "".join(event.text for event in padded_hidden),
                    "strict_events": [_event_row(event) for event in strict],
                    "padded_events": [_event_row(event) for event in padded],
                    "nearest_hidden": [
                        {
                            **_event_row(event),
                            "distance_to_cell": round(_distance_to_bbox(cell.bbox, event.page_origin), 4),
                        }
                        for event in nearest_hidden
                    ],
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
        "phase": "phase5_hidden_text_cell_diagnostic",
        "case_id": CASE_ID,
        "validation_status": "DIAGNOSTIC_ONLY",
        "ocr_used": False,
        "llm_used": False,
        "association_basis": "Tj/TJ origin reconstructed from content stream",
        "padded_tolerance_pt": PADDED_TOLERANCE_PT,
        "page_event_count": len(events),
        "page_nonempty_event_count": len(nonempty_events),
        "page_hidden_event_count": len(hidden_events),
        "benchmark_real_frame_count": len(results),
        "results": results,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("phase=phase5_hidden_text_cell_diagnostic")
    print("validation_status=DIAGNOSTIC_ONLY")
    print("ocr_used=False")
    print("llm_used=False")
    print(f"page_events={len(events)}")
    print(f"page_nonempty_events={len(nonempty_events)}")
    print(f"page_hidden_events={len(hidden_events)}")
    print(f"benchmark_real_frames={len(results)}")
    print("\nbenchmark_real_frame_hidden_text:")

    for row in results:
        print(f"  {row['candidate_id']} cells={row['cell_count']}")
        for cell in row["cells"]:
            strict_events = cell["strict_events"]
            padded_events = cell["padded_events"]
            print(
                f"    cell[{cell['cell_index']}] "
                f"strict={cell['strict_text']!r} "
                f"strict_hidden={cell['strict_hidden_text']!r} "
                f"padded={cell['padded_text']!r} "
                f"events={len(strict_events)}/{len(padded_events)}"
            )
            if strict_events:
                for event in strict_events:
                    text = str(event["text"])
                    x, y = event["page_origin"]
                    print(
                        f"      hit text={text!r} page=({x:.1f},{y:.1f}) "
                        f"xref={event['xref']} Tr={event['rendering_mode']}"
                    )
            else:
                nearest = cell["nearest_hidden"][:3]
                if nearest:
                    compact = []
                    for event in nearest:
                        x, y = event["page_origin"]
                        text = str(event["text"])
                        compact.append(
                            f"{text!r} d={event['distance_to_cell']:.1f} "
                            f"@({x:.1f},{y:.1f})"
                        )
                    print("      nearest_hidden=" + "; ".join(compact))

    print(f"\noutput={OUTPUT_PATH}")


if __name__ == "__main__":
    main()
