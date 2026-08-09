"""Fase 5: valida visualmente a transformação do texto invisível do PDF.

Motivação
---------
O parser de content stream reconstrói eventos ``Tj/TJ`` com duas posições:
``pdf_origin`` (coordenada após text matrix + CTM, ainda no espaço PDF) e
``page_origin`` (após ``page.transformation_matrix``). O diagnóstico anterior
mostrou que ``page_origin`` não caiu dentro das células GD&T, mas isso NÃO prova
que a camada textual não corresponda ao desenho: a transformação pode estar
incorreta ou incompleta.

Este script transforma os mesmos eventos por caminhos diferentes e pinta os
pontos sobre a página renderizada, junto com os frames/células já detectados.
Assim a validação passa a ser visual e geométrica, sem OCR e sem LLM.

Saídas principais
-----------------
- ``page_overlay_parser_page.png``: usa ``event.page_origin`` atual.
- ``page_overlay_pdf_raw.png``: usa ``event.pdf_origin`` sem transformação.
- ``page_overlay_pdf_yflip.png``: aplica flip Y simples pela altura da página.
- ``page_overlay_pdf_page_matrix.png``: aplica explicitamente
  ``page.transformation_matrix`` sobre ``event.pdf_origin``.
- ``*_short.png``: mesma variante, mas somente textos curtos para reduzir ruído.
- ``frame_zooms/``: zoom dos seis frames benchmark para cada variante.
- ``transform_overlay_summary.json``: contagens dentro de frames/células e
  vizinhos mais próximos.

Este arquivo é DIAGNOSTIC_ONLY. Não altera o ``frame_parser`` e não conclui
automaticamente qual transformação está correta.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Callable

import fitz
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gdt.detector import GdtFrameDetector
from src.gdt.pdf_hidden_text import PdfTextEvent, extract_page_text_events

CASE_ID = "case_41_rev8"
CASE_PATH = PROJECT_ROOT / "validation" / "gdt" / "cases" / f"{CASE_ID}.json"
GEOMETRY_BASELINE = PROJECT_ROOT / "validation" / "gdt" / "baselines" / f"{CASE_ID}.geometry.json"
OUTPUT_DIR = (
    PROJECT_ROOT
    / "validation"
    / "gdt"
    / "outputs"
    / "phase5"
    / CASE_ID
    / "hidden_text_transform_overlay"
)
SUMMARY_PATH = OUTPUT_DIR / "transform_overlay_summary.json"

DPI = 300
SCALE = DPI / 72.0
FRAME_PAD_PT = 18.0
NEAREST_LIMIT = 5
POINT_RADIUS_PX = 5


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _point_distance_to_bbox(bbox, point: tuple[float, float]) -> float:
    x, y = point
    dx = max(bbox.x0 - x, 0.0, x - bbox.x1)
    dy = max(bbox.y0 - y, 0.0, y - bbox.y1)
    return math.hypot(dx, dy)


def _contains(bbox, point: tuple[float, float], padding: float = 0.0) -> bool:
    x, y = point
    return (
        bbox.x0 - padding <= x <= bbox.x1 + padding
        and bbox.y0 - padding <= y <= bbox.y1 + padding
    )


def _fitz_point_tuple(point: fitz.Point) -> tuple[float, float]:
    return float(point.x), float(point.y)


def _variant_parser_page(event: PdfTextEvent, page: fitz.Page) -> tuple[float, float]:
    del page
    return float(event.page_origin[0]), float(event.page_origin[1])


def _variant_pdf_raw(event: PdfTextEvent, page: fitz.Page) -> tuple[float, float]:
    del page
    return float(event.pdf_origin[0]), float(event.pdf_origin[1])


def _variant_pdf_yflip(event: PdfTextEvent, page: fitz.Page) -> tuple[float, float]:
    x, y = event.pdf_origin
    return float(x), float(page.rect.height - y)


def _variant_pdf_page_matrix(event: PdfTextEvent, page: fitz.Page) -> tuple[float, float]:
    point = fitz.Point(float(event.pdf_origin[0]), float(event.pdf_origin[1]))
    return _fitz_point_tuple(point * page.transformation_matrix)


VariantFn = Callable[[PdfTextEvent, fitz.Page], tuple[float, float]]

VARIANTS: dict[str, VariantFn] = {
    "parser_page": _variant_parser_page,
    "pdf_raw": _variant_pdf_raw,
    "pdf_yflip": _variant_pdf_yflip,
    "pdf_page_matrix": _variant_pdf_page_matrix,
}


def _is_short_text(text: str) -> bool:
    value = text.strip()
    if not value:
        return False
    # Mantém tokens pequenos típicos de CAD/GD&T e anotações úteis para calibração.
    return len(value) <= 8 and "\n" not in value and "\r" not in value


def _render_page(page: fitz.Page) -> Image.Image:
    pix = page.get_pixmap(matrix=fitz.Matrix(SCALE, SCALE), alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def _pt_to_px(point: tuple[float, float]) -> tuple[float, float]:
    return point[0] * SCALE, point[1] * SCALE


def _bbox_to_px(bbox) -> tuple[float, float, float, float]:
    return bbox.x0 * SCALE, bbox.y0 * SCALE, bbox.x1 * SCALE, bbox.y1 * SCALE


def _draw_geometry(draw: ImageDraw.ImageDraw, candidates) -> None:
    font = ImageFont.load_default()
    for candidate in candidates:
        x0, y0, x1, y1 = _bbox_to_px(candidate.frame_bbox)
        # Azul = frame candidato benchmark.
        draw.rectangle((x0, y0, x1, y1), outline=(0, 90, 220), width=4)
        draw.text((x0 + 3, max(0, y0 - 13)), candidate.candidate_id, fill=(0, 70, 180), font=font)

        for index, cell in enumerate(candidate.cells):
            cx0, cy0, cx1, cy1 = _bbox_to_px(cell.bbox)
            # Verde = célula detectada.
            draw.rectangle((cx0, cy0, cx1, cy1), outline=(0, 165, 60), width=2)
            draw.text((cx0 + 2, cy0 + 2), f"c{index}", fill=(0, 125, 40), font=font)


def _draw_events(
    image: Image.Image,
    page: fitz.Page,
    candidates,
    events: list[PdfTextEvent],
    variant_name: str,
    *,
    short_only: bool,
) -> tuple[Image.Image, dict]:
    transform = VARIANTS[variant_name]
    output = image.copy()
    draw = ImageDraw.Draw(output)
    font = ImageFont.load_default()
    _draw_geometry(draw, candidates)

    transformed_rows = []
    inside_any_frame = 0
    inside_any_cell = 0

    for event in events:
        if not event.text.strip():
            continue
        if short_only and not _is_short_text(event.text):
            continue

        point = transform(event, page)
        frame_ids = []
        cell_ids = []
        for candidate in candidates:
            if _contains(candidate.frame_bbox, point):
                frame_ids.append(candidate.candidate_id)
            for index, cell in enumerate(candidate.cells):
                if _contains(cell.bbox, point):
                    cell_ids.append(f"{candidate.candidate_id}:cell[{index}]")

        if frame_ids:
            inside_any_frame += 1
        if cell_ids:
            inside_any_cell += 1

        px, py = _pt_to_px(point)
        radius = POINT_RADIUS_PX
        # Vermelho = origem reconstruída do evento textual invisível.
        draw.ellipse(
            (px - radius, py - radius, px + radius, py + radius),
            fill=(220, 30, 30),
            outline=(150, 0, 0),
        )
        label = event.text.strip().replace("\n", "\\n").replace("\r", "\\r")
        if len(label) > 22:
            label = label[:19] + "..."
        draw.text((px + 7, py - 7), label, fill=(170, 0, 0), font=font)

        transformed_rows.append(
            {
                "sequence": event.sequence,
                "xref": event.xref,
                "text": event.text,
                "rendering_mode": event.rendering_mode,
                "invisible": event.invisible,
                "pdf_origin": [round(v, 4) for v in event.pdf_origin],
                "parser_page_origin": [round(v, 4) for v in event.page_origin],
                "variant_page_point": [round(v, 4) for v in point],
                "inside_frames": frame_ids,
                "inside_cells": cell_ids,
            }
        )

    return output, {
        "variant": variant_name,
        "short_only": short_only,
        "plotted_event_count": len(transformed_rows),
        "inside_any_frame": inside_any_frame,
        "inside_any_cell": inside_any_cell,
        "events": transformed_rows,
    }


def _crop_frame_zoom(
    overlay: Image.Image,
    candidate,
    *,
    padding_pt: float = FRAME_PAD_PT,
) -> Image.Image:
    x0 = max(0.0, (candidate.frame_bbox.x0 - padding_pt) * SCALE)
    y0 = max(0.0, (candidate.frame_bbox.y0 - padding_pt) * SCALE)
    x1 = min(float(overlay.width), (candidate.frame_bbox.x1 + padding_pt) * SCALE)
    y1 = min(float(overlay.height), (candidate.frame_bbox.y1 + padding_pt) * SCALE)
    return overlay.crop((int(x0), int(y0), int(x1), int(y1)))


def _cell_summary(candidates, events: list[PdfTextEvent], page: fitz.Page, variant_name: str) -> list[dict]:
    transform = VARIANTS[variant_name]
    transformed = [(event, transform(event, page)) for event in events if event.text.strip()]
    rows = []

    for candidate in candidates:
        cell_rows = []
        for index, cell in enumerate(candidate.cells):
            inside = [(event, point) for event, point in transformed if _contains(cell.bbox, point)]
            nearest = sorted(
                transformed,
                key=lambda pair: _point_distance_to_bbox(cell.bbox, pair[1]),
            )[:NEAREST_LIMIT]
            cell_rows.append(
                {
                    "cell_index": index,
                    "bbox": [round(v, 3) for v in cell.bbox.to_list()],
                    "inside": [
                        {
                            "text": event.text,
                            "sequence": event.sequence,
                            "xref": event.xref,
                            "point": [round(v, 4) for v in point],
                        }
                        for event, point in inside
                    ],
                    "nearest": [
                        {
                            "text": event.text,
                            "sequence": event.sequence,
                            "xref": event.xref,
                            "point": [round(v, 4) for v in point],
                            "distance_to_cell": round(_point_distance_to_bbox(cell.bbox, point), 4),
                        }
                        for event, point in nearest
                    ],
                }
            )
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "frame_bbox": [round(v, 3) for v in candidate.frame_bbox.to_list()],
                "cells": cell_rows,
            }
        )
    return rows


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
    all_candidates = detector.detect_frames(pdf_bytes, page_index=page_index)
    candidates = [c for c in all_candidates if c.candidate_id in benchmark_ids]

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        page = doc[page_index]
        events = extract_page_text_events(doc, page_index=page_index)
        events = [event for event in events if event.text.strip()]
        hidden_events = [event for event in events if event.invisible]
        base_image = _render_page(page)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        zoom_dir = OUTPUT_DIR / "frame_zooms"
        zoom_dir.mkdir(parents=True, exist_ok=True)

        variant_payload = {}
        full_overlays: dict[str, Image.Image] = {}

        for variant_name in VARIANTS:
            full_overlay, full_stats = _draw_events(
                base_image,
                page,
                candidates,
                hidden_events,
                variant_name,
                short_only=False,
            )
            short_overlay, short_stats = _draw_events(
                base_image,
                page,
                candidates,
                hidden_events,
                variant_name,
                short_only=True,
            )

            full_path = OUTPUT_DIR / f"page_overlay_{variant_name}.png"
            short_path = OUTPUT_DIR / f"page_overlay_{variant_name}_short.png"
            full_overlay.save(full_path)
            short_overlay.save(short_path)
            full_overlays[variant_name] = full_overlay

            for candidate in candidates:
                zoom = _crop_frame_zoom(full_overlay, candidate)
                zoom.save(zoom_dir / f"{candidate.candidate_id}_{variant_name}.png")

            variant_payload[variant_name] = {
                "full": full_stats,
                "short": short_stats,
                "cells": _cell_summary(candidates, hidden_events, page, variant_name),
                "page_overlay": str(full_path),
                "short_overlay": str(short_path),
            }

        # Consistência interna: o parser hoje define page_origin justamente como
        # pdf_origin * page.transformation_matrix. Este delta deve ficar ~0 se a
        # implementação estiver aplicando essa matriz como esperado.
        deltas = []
        for event in hidden_events:
            a = _variant_parser_page(event, page)
            b = _variant_pdf_page_matrix(event, page)
            deltas.append(math.hypot(a[0] - b[0], a[1] - b[1]))

        payload = {
            "schema_version": 1,
            "phase": "phase5_hidden_text_transform_overlay",
            "case_id": CASE_ID,
            "validation_status": "DIAGNOSTIC_ONLY",
            "ocr_used": False,
            "llm_used": False,
            "dpi": DPI,
            "page_rect": [round(v, 4) for v in (page.rect.x0, page.rect.y0, page.rect.x1, page.rect.y1)],
            "page_transformation_matrix": [
                round(page.transformation_matrix.a, 6),
                round(page.transformation_matrix.b, 6),
                round(page.transformation_matrix.c, 6),
                round(page.transformation_matrix.d, 6),
                round(page.transformation_matrix.e, 6),
                round(page.transformation_matrix.f, 6),
            ],
            "page_event_count": len(events),
            "page_hidden_event_count": len(hidden_events),
            "benchmark_real_frame_count": len(candidates),
            "parser_page_vs_explicit_page_matrix_max_delta_pt": round(max(deltas) if deltas else 0.0, 8),
            "variants": variant_payload,
            "interpretation_note": (
                "Visual diagnostic only. A variant with points over known visible labels validates the coordinate "
                "mapping better than inside-cell counts alone. Do not conclude that hidden text is absent from GD&T "
                "until the overlay itself is reviewed."
            ),
        }
        SUMMARY_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        print("phase=phase5_hidden_text_transform_overlay")
        print("validation_status=DIAGNOSTIC_ONLY")
        print("ocr_used=False")
        print("llm_used=False")
        print(f"dpi={DPI}")
        print(f"page_events={len(events)}")
        print(f"page_hidden_events={len(hidden_events)}")
        print(f"benchmark_real_frames={len(candidates)}")
        print(f"page_transformation_matrix={payload['page_transformation_matrix']}")
        print(
            "parser_page_vs_explicit_page_matrix_max_delta_pt="
            f"{payload['parser_page_vs_explicit_page_matrix_max_delta_pt']}"
        )
        print("\nvariant_stats:")
        for variant_name in VARIANTS:
            stats = variant_payload[variant_name]["full"]
            print(
                f"  {variant_name}: plotted={stats['plotted_event_count']} "
                f"inside_any_frame={stats['inside_any_frame']} "
                f"inside_any_cell={stats['inside_any_cell']}"
            )

        print("\nbenchmark_real_frame_transform_nearest:")
        for variant_name in VARIANTS:
            print(f"  variant={variant_name}")
            for row in variant_payload[variant_name]["cells"]:
                print(f"    {row['candidate_id']}")
                for cell in row["cells"]:
                    nearest = "; ".join(
                        f"{entry['text']!r} d={entry['distance_to_cell']:.1f} @({entry['point'][0]:.1f},{entry['point'][1]:.1f})"
                        for entry in cell["nearest"][:3]
                    )
                    inside_text = "".join(entry["text"] for entry in cell["inside"])
                    print(
                        f"      cell[{cell['cell_index']}] inside={inside_text!r} "
                        f"nearest={nearest}"
                    )

        print(f"\noutput_dir={OUTPUT_DIR}")
        print(f"summary={SUMMARY_PATH}")
        print(f"frame_zooms={zoom_dir}")
    finally:
        doc.close()


if __name__ == "__main__":
    main()
