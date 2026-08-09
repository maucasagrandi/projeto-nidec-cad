"""Fase 5: diagnóstico de bbox estimada da camada textual invisível.

Depois de validar visualmente que as origens ``Tj/TJ`` têm relação espacial com
o desenho, este script deixa de testar apenas um ponto. Ele estima a área
ocupada por cada string usando a fonte do PDF e mede a interseção dessa bbox com
os frames/células GD&T.

O diagnóstico compara três sistemas de coordenadas:
- ``pdf_raw``: quad estimado no espaço PDF sem transformação de página;
- ``pdf_yflip``: flip Y simples pela altura da página;
- ``pdf_page_matrix``: ``page.transformation_matrix`` do PyMuPDF.

Não há OCR, LLM, filtro semântico de conteúdo nem PASS/FAIL de Fase 5 aqui.
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
from src.gdt.pdf_hidden_text_geometry import (
    PdfTextGeometryEvent,
    extract_page_text_geometry_events,
)

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
    / "hidden_text_bbox_diagnostic"
)
OUTPUT_PATH = OUTPUT_DIR / "hidden_text_bbox_summary.json"

DPI = 300
SCALE = DPI / 72.0
FRAME_PAD_PT = 22.0
NEAREST_LIMIT = 8
CELL_PADDING_SWEEP_PT = (0.0, 0.5, 1.0, 2.0, 3.0, 5.0)

BBox = tuple[float, float, float, float]
Quad = tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _quad_bbox(quad: Quad) -> BBox:
    xs = [point[0] for point in quad]
    ys = [point[1] for point in quad]
    return min(xs), min(ys), max(xs), max(ys)


def _transform_quad(quad: Quad, matrix: fitz.Matrix) -> Quad:
    rows = []
    for x, y in quad:
        point = fitz.Point(x, y) * matrix
        rows.append((float(point.x), float(point.y)))
    return tuple(rows)  # type: ignore[return-value]


def _bbox_area(bbox: BBox) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _intersection_bbox(a: BBox, b: BBox) -> BBox | None:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _intersection_area(a: BBox, b: BBox) -> float:
    intersection = _intersection_bbox(a, b)
    return _bbox_area(intersection) if intersection else 0.0


def _iou(a: BBox, b: BBox) -> float:
    inter = _intersection_area(a, b)
    if inter <= 0.0:
        return 0.0
    union = _bbox_area(a) + _bbox_area(b) - inter
    return inter / union if union > 0.0 else 0.0


def _bbox_distance(a: BBox, b: BBox) -> float:
    dx = max(a[0] - b[2], b[0] - a[2], 0.0)
    dy = max(a[1] - b[3], b[1] - a[3], 0.0)
    return math.hypot(dx, dy)


def _expand_bbox(bbox: BBox, padding: float) -> BBox:
    return (
        bbox[0] - padding,
        bbox[1] - padding,
        bbox[2] + padding,
        bbox[3] + padding,
    )


def _detector_bbox(bbox) -> BBox:
    return float(bbox.x0), float(bbox.y0), float(bbox.x1), float(bbox.y1)


def _variant_pdf_raw(event: PdfTextGeometryEvent, page: fitz.Page) -> Quad:
    del page
    return event.pdf_quad


def _variant_pdf_yflip(event: PdfTextGeometryEvent, page: fitz.Page) -> Quad:
    matrix = fitz.Matrix(1, 0, 0, -1, 0, float(page.rect.height))
    return _transform_quad(event.pdf_quad, matrix)


def _variant_pdf_page_matrix(event: PdfTextGeometryEvent, page: fitz.Page) -> Quad:
    del page
    return event.page_quad


VariantFn = Callable[[PdfTextGeometryEvent, fitz.Page], Quad]
VARIANTS: dict[str, VariantFn] = {
    "pdf_raw": _variant_pdf_raw,
    "pdf_yflip": _variant_pdf_yflip,
    "pdf_page_matrix": _variant_pdf_page_matrix,
}


def _is_short_text(text: str) -> bool:
    value = text.strip()
    return bool(value) and len(value) <= 8 and "\n" not in value and "\r" not in value


def _render_page(page: fitz.Page) -> Image.Image:
    pix = page.get_pixmap(matrix=fitz.Matrix(SCALE, SCALE), alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def _bbox_to_px(bbox: BBox) -> tuple[float, float, float, float]:
    return tuple(value * SCALE for value in bbox)  # type: ignore[return-value]


def _draw_geometry(draw: ImageDraw.ImageDraw, candidates) -> None:
    font = ImageFont.load_default()
    for candidate in candidates:
        frame = _bbox_to_px(_detector_bbox(candidate.frame_bbox))
        draw.rectangle(frame, outline=(0, 90, 220), width=4)
        draw.text(
            (frame[0] + 3, max(0, frame[1] - 13)),
            candidate.candidate_id,
            fill=(0, 70, 180),
            font=font,
        )
        for index, cell in enumerate(candidate.cells):
            cell_px = _bbox_to_px(_detector_bbox(cell.bbox))
            draw.rectangle(cell_px, outline=(0, 165, 60), width=2)
            draw.text((cell_px[0] + 2, cell_px[1] + 2), f"c{index}", fill=(0, 125, 40), font=font)


def _draw_text_boxes(
    base_image: Image.Image,
    candidates,
    transformed: list[tuple[PdfTextGeometryEvent, Quad, BBox]],
    *,
    short_only: bool,
) -> Image.Image:
    output = base_image.copy()
    draw = ImageDraw.Draw(output)
    font = ImageFont.load_default()
    _draw_geometry(draw, candidates)

    for event, _quad, bbox in transformed:
        if short_only and not _is_short_text(event.text):
            continue
        px_bbox = _bbox_to_px(bbox)
        draw.rectangle(px_bbox, outline=(220, 30, 30), width=2)
        label = event.text.strip().replace("\n", "\\n").replace("\r", "\\r")
        if len(label) > 20:
            label = label[:17] + "..."
        draw.text((px_bbox[0] + 2, max(0, px_bbox[1] - 10)), label, fill=(170, 0, 0), font=font)

    return output


def _event_metrics(event: PdfTextGeometryEvent, text_bbox: BBox, cell_bbox: BBox) -> dict:
    inter = _intersection_area(text_bbox, cell_bbox)
    text_area = _bbox_area(text_bbox)
    cell_area = _bbox_area(cell_bbox)
    return {
        "sequence": event.sequence,
        "xref": event.xref,
        "text": event.text,
        "font_resource": event.font_resource,
        "font_size": event.font_size,
        "font_xref": event.font_xref,
        "font_metrics_source": event.font_metrics_source,
        "bbox_quality": event.bbox_quality,
        "text_bbox": [round(value, 4) for value in text_bbox],
        "intersection_area": round(inter, 4),
        "overlap_text": round(inter / text_area, 4) if text_area > 0.0 else 0.0,
        "overlap_cell": round(inter / cell_area, 4) if cell_area > 0.0 else 0.0,
        "iou": round(_iou(text_bbox, cell_bbox), 4),
        "distance_to_cell": round(_bbox_distance(text_bbox, cell_bbox), 4),
    }


def _rank_for_cell(row: dict) -> tuple:
    # Diagnóstico apenas: interseções primeiro; depois maior fração da bbox do
    # texto dentro da célula; por fim menor distância. Não é classificação.
    intersects = row["intersection_area"] > 0.0
    return (
        0 if intersects else 1,
        -float(row["overlap_text"]),
        float(row["distance_to_cell"]),
        int(row["sequence"]),
    )


def _cell_analysis(candidates, transformed) -> list[dict]:
    candidate_rows = []
    for candidate in candidates:
        cells = []
        for index, cell in enumerate(candidate.cells):
            cell_bbox = _detector_bbox(cell.bbox)
            rows = [_event_metrics(event, bbox, cell_bbox) for event, _quad, bbox in transformed]
            rows.sort(key=_rank_for_cell)
            intersecting = [row for row in rows if row["intersection_area"] > 0.0]

            padding_sweep = {}
            for padding in CELL_PADDING_SWEEP_PT:
                expanded = _expand_bbox(cell_bbox, padding)
                hits = [
                    event.text
                    for event, _quad, bbox in transformed
                    if _intersection_area(bbox, expanded) > 0.0
                ]
                padding_sweep[str(padding)] = {
                    "hit_count": len(hits),
                    "texts": hits[:20],
                }

            cells.append(
                {
                    "cell_index": index,
                    "cell_bbox": [round(value, 4) for value in cell_bbox],
                    "intersection_count": len(intersecting),
                    "intersections": intersecting[:20],
                    "nearest_or_best": rows[:NEAREST_LIMIT],
                    "padding_sweep_pt": padding_sweep,
                }
            )

        candidate_rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "frame_bbox": [round(value, 4) for value in _detector_bbox(candidate.frame_bbox)],
                "cells": cells,
            }
        )
    return candidate_rows


def _variant_summary(candidates, transformed) -> dict:
    frames_with_intersection: set[str] = set()
    cells_with_intersection: set[str] = set()
    event_sequences_in_frames: set[int] = set()
    event_sequences_in_cells: set[int] = set()

    for event, _quad, text_bbox in transformed:
        for candidate in candidates:
            if _intersection_area(text_bbox, _detector_bbox(candidate.frame_bbox)) > 0.0:
                frames_with_intersection.add(candidate.candidate_id)
                event_sequences_in_frames.add(event.sequence)
            for index, cell in enumerate(candidate.cells):
                if _intersection_area(text_bbox, _detector_bbox(cell.bbox)) > 0.0:
                    cells_with_intersection.add(f"{candidate.candidate_id}:cell[{index}]")
                    event_sequences_in_cells.add(event.sequence)

    return {
        "event_count": len(transformed),
        "events_intersecting_any_frame": len(event_sequences_in_frames),
        "events_intersecting_any_cell": len(event_sequences_in_cells),
        "frames_with_text_bbox_intersection": len(frames_with_intersection),
        "cells_with_text_bbox_intersection": len(cells_with_intersection),
        "frame_ids": sorted(frames_with_intersection),
        "cell_ids": sorted(cells_with_intersection),
    }


def _crop_zoom(image: Image.Image, frame_bbox, padding_pt: float = FRAME_PAD_PT) -> Image.Image:
    bbox = _detector_bbox(frame_bbox)
    x0 = max(0.0, (bbox[0] - padding_pt) * SCALE)
    y0 = max(0.0, (bbox[1] - padding_pt) * SCALE)
    x1 = min(float(image.width), (bbox[2] + padding_pt) * SCALE)
    y1 = min(float(image.height), (bbox[3] + padding_pt) * SCALE)
    return image.crop((int(x0), int(y0), int(x1), int(y1)))


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
    candidates = [
        candidate
        for candidate in detector.detect_frames(pdf_bytes, page_index=page_index)
        if candidate.candidate_id in benchmark_ids
    ]

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        page = doc[page_index]
        events, font_resolver = extract_page_text_geometry_events(doc, page_index=page_index)
        hidden_events = [event for event in events if event.invisible and event.text.strip()]
        base_image = _render_page(page)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        zoom_dir = OUTPUT_DIR / "frame_zooms"
        zoom_dir.mkdir(parents=True, exist_ok=True)

        variants_payload = {}
        terminal_rows = []

        for variant_name, transform in VARIANTS.items():
            transformed = []
            for event in hidden_events:
                quad = transform(event, page)
                transformed.append((event, quad, _quad_bbox(quad)))

            full_overlay = _draw_text_boxes(
                base_image,
                candidates,
                transformed,
                short_only=False,
            )
            short_overlay = _draw_text_boxes(
                base_image,
                candidates,
                transformed,
                short_only=True,
            )

            full_path = OUTPUT_DIR / f"page_bbox_{variant_name}.png"
            short_path = OUTPUT_DIR / f"page_bbox_{variant_name}_short.png"
            full_overlay.save(full_path)
            short_overlay.save(short_path)

            for candidate in candidates:
                zoom = _crop_zoom(short_overlay, candidate.frame_bbox)
                zoom.save(zoom_dir / f"{candidate.candidate_id}_{variant_name}.png")

            summary = _variant_summary(candidates, transformed)
            cells = _cell_analysis(candidates, transformed)
            variants_payload[variant_name] = {
                "summary": summary,
                "cells": cells,
                "page_overlay": str(full_path),
                "short_overlay": str(short_path),
            }
            terminal_rows.append((variant_name, summary, cells))

        payload = {
            "schema_version": 1,
            "phase": "phase5_hidden_text_bbox_diagnostic",
            "case_id": CASE_ID,
            "validation_status": "DIAGNOSTIC_ONLY",
            "ocr_used": False,
            "llm_used": False,
            "association_basis": "estimated text bbox from PDF font metrics + text matrix + CTM",
            "bbox_is_ground_truth": False,
            "page_rect": [
                round(page.rect.x0, 4),
                round(page.rect.y0, 4),
                round(page.rect.x1, 4),
                round(page.rect.y1, 4),
            ],
            "page_hidden_event_count": len(hidden_events),
            "benchmark_real_frame_count": len(candidates),
            "font_resources": font_resolver.to_dict(),
            "cell_padding_sweep_pt": list(CELL_PADDING_SWEEP_PT),
            "variants": variants_payload,
            "interpretation_note": (
                "Interseção de bbox estimada é evidência geométrica, não validação semântica. "
                "Não tratar o texto melhor ranqueado como conteúdo correto sem revisão independente."
            ),
        }
        OUTPUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        print("phase=phase5_hidden_text_bbox_diagnostic")
        print("validation_status=DIAGNOSTIC_ONLY")
        print("ocr_used=False")
        print("llm_used=False")
        print(f"page_hidden_events={len(hidden_events)}")
        print(f"benchmark_real_frames={len(candidates)}")
        print("\nfont_resources:")
        for resource, row in font_resolver.to_dict().items():
            print(
                f"  {resource}: xref={row['xref']} basefont={row['basefont']!r} "
                f"source={row['source']} font={row['font_name']!r} "
                f"asc={row['ascender']:.4f} desc={row['descender']:.4f}"
            )

        print("\nvariant_stats:")
        for variant_name, summary, _cells in terminal_rows:
            print(
                f"  {variant_name}: events_in_frames={summary['events_intersecting_any_frame']} "
                f"events_in_cells={summary['events_intersecting_any_cell']} "
                f"frames_hit={summary['frames_with_text_bbox_intersection']}/6 "
                f"cells_hit={summary['cells_with_text_bbox_intersection']}"
            )

        print("\nbenchmark_real_frame_bbox_candidates:")
        for variant_name, _summary, candidates_rows in terminal_rows:
            print(f"  variant={variant_name}")
            for candidate in candidates_rows:
                print(f"    {candidate['candidate_id']}")
                for cell in candidate["cells"]:
                    best = cell["nearest_or_best"][0] if cell["nearest_or_best"] else None
                    if best is None:
                        print(f"      cell[{cell['cell_index']}] no_text_events")
                        continue
                    print(
                        f"      cell[{cell['cell_index']}] intersections={cell['intersection_count']} "
                        f"best={best['text']!r} overlap_text={best['overlap_text']:.3f} "
                        f"distance={best['distance_to_cell']:.3f}"
                    )

        print(f"\noutput={OUTPUT_PATH}")
        print(f"overlays={OUTPUT_DIR}")
    finally:
        doc.close()


if __name__ == "__main__":
    main()
