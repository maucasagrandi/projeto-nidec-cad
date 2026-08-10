"""Phase 5 diagnostic: isolate plausible text glyphs inside GD&T content cells.

This diagnostic intentionally starts from the already-detected cell bboxes.
It does NOT revisit frame or cell localization.

Compared with the earlier glyph-segmentation diagnostic, this version:
- renders the FULL cell bbox (no 0.8 pt inset, which clipped some A/B/D glyphs);
- keeps connected components intact;
- labels each component deterministically as structural_line, arrow_like,
  text_candidate, or other;
- saves a candidate-only mask so we can see what remains after ignoring obvious
  frame / leader geometry.

No OCR, no LLM, and no character classification are performed here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import fitz
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gdt.cell_visual_content import analyze_components, binarize_cell, build_text_candidate_mask
from src.gdt.detector import GdtFrameDetector

CASE_ID = "case_41_rev8"
CASE_PATH = PROJECT_ROOT / "validation" / "gdt" / "cases" / f"{CASE_ID}.json"
GEOMETRY_BASELINE = PROJECT_ROOT / "validation" / "gdt" / "baselines" / f"{CASE_ID}.geometry.json"
OUTPUT_DIR = PROJECT_ROOT / "validation" / "gdt" / "outputs" / "phase5" / CASE_ID / "cell_content_filter"
OUTPUT_PATH = OUTPUT_DIR / "cell_content_filter.json"
CONTACT_SHEET_PATH = OUTPUT_DIR / "cell_content_filter_contact_sheet.png"

RENDER_DPI = 1200
MIN_COMPONENT_AREA_PX = 8
CONTACT_TILE_WIDTH = 960
CONTACT_TILE_HEIGHT = 285

CLASS_COLORS_BGR = {
    "text_candidate": (0, 160, 0),
    "structural_line": (255, 120, 0),
    "arrow_like": (0, 0, 255),
    "other": (160, 0, 160),
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _render_gray(page: fitz.Page, rect: fitz.Rect) -> np.ndarray:
    scale = RENDER_DPI / 72.0
    pix = page.get_pixmap(
        matrix=fitz.Matrix(scale, scale),
        clip=rect,
        colorspace=fitz.csGRAY,
        alpha=False,
    )
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width).copy()


def _save_visible_binary(binary: np.ndarray, path: Path) -> None:
    cv2.imwrite(str(path), 255 - binary)


def _save_annotated(gray: np.ndarray, components, path: Path) -> None:
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for index, component in enumerate(components, start=1):
        x0, y0, x1, y1 = component.bbox_px
        color = CLASS_COLORS_BGR.get(component.component_class, (0, 0, 0))
        cv2.rectangle(image, (x0, y0), (max(x0, x1 - 1), max(y0, y1 - 1)), color, 2)
        label = f"{index}:{component.component_class}"
        cv2.putText(
            image,
            label,
            (x0, max(14, y0 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            color,
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(path), image)


def _fit(image: Image.Image, max_width: int, max_height: int) -> Image.Image:
    if image.width <= 0 or image.height <= 0:
        return image
    scale = min(max_width / image.width, max_height / image.height)
    scale = max(0.01, scale)
    size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def _build_contact_sheet(rows: list[dict]) -> None:
    if not rows:
        return

    sheet = Image.new("RGB", (CONTACT_TILE_WIDTH, CONTACT_TILE_HEIGHT * len(rows)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for row_index, row in enumerate(rows):
        top = row_index * CONTACT_TILE_HEIGHT
        title = (
            f"{row['candidate_id']} cell[{row['cell_index']}] role={row['expected_role']} "
            f"text_candidates={row['text_candidate_count']} "
            f"structural={row['structural_count']} arrows={row['arrow_like_count']} other={row['other_count']}"
        )
        draw.text((10, top + 8), title, fill="black", font=font)

        original = Image.open(OUTPUT_DIR / row["original_crop"]).convert("RGB")
        annotated = Image.open(OUTPUT_DIR / row["annotated_crop"]).convert("RGB")
        candidate_only = Image.open(OUTPUT_DIR / row["text_candidate_crop"]).convert("RGB")

        original = _fit(original, 290, 190)
        annotated = _fit(annotated, 290, 190)
        candidate_only = _fit(candidate_only, 290, 190)

        sheet.paste(original, (10, top + 50))
        sheet.paste(annotated, (330, top + 50))
        sheet.paste(candidate_only, (650, top + 50))
        draw.text((10, top + 245), "full cell", fill="black", font=font)
        draw.text((330, top + 245), "classified components", fill="black", font=font)
        draw.text((650, top + 245), "text candidates only", fill="black", font=font)

        if row_index < len(rows) - 1:
            draw.line((0, top + CONTACT_TILE_HEIGHT - 1, CONTACT_TILE_WIDTH, top + CONTACT_TILE_HEIGHT - 1), fill="gray")

    sheet.save(CONTACT_SHEET_PATH)


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
    candidates = [candidate for candidate in all_candidates if candidate.candidate_id in benchmark_ids]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        page = doc[page_index]
        for candidate in candidates:
            for cell_index, cell in enumerate(candidate.cells[1:], start=1):
                rect = fitz.Rect(cell.bbox.to_list())
                gray = _render_gray(page, rect)
                binary = binarize_cell(gray)
                components = analyze_components(binary, min_area_px=MIN_COMPONENT_AREA_PX)
                text_mask = build_text_candidate_mask(binary, components)

                prefix = f"{candidate.candidate_id}_cell_{cell_index:02d}"
                original_name = f"{prefix}_full.png"
                annotated_name = f"{prefix}_classified.png"
                text_name = f"{prefix}_text_candidates.png"

                cv2.imwrite(str(OUTPUT_DIR / original_name), gray)
                _save_annotated(gray, components, OUTPUT_DIR / annotated_name)
                _save_visible_binary(text_mask, OUTPUT_DIR / text_name)

                counts = {
                    key: sum(component.component_class == key for component in components)
                    for key in ("text_candidate", "structural_line", "arrow_like", "other")
                }

                rows.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "cell_index": cell_index,
                        "expected_role": "tolerance" if cell_index == 1 else "datum_or_modifier",
                        "cell_bbox": [round(value, 4) for value in cell.bbox.to_list()],
                        "render_dpi": RENDER_DPI,
                        "image_size_px": [int(gray.shape[1]), int(gray.shape[0])],
                        "component_count": len(components),
                        "text_candidate_count": counts["text_candidate"],
                        "structural_count": counts["structural_line"],
                        "arrow_like_count": counts["arrow_like"],
                        "other_count": counts["other"],
                        "components": [component.to_dict() for component in components],
                        "original_crop": original_name,
                        "annotated_crop": annotated_name,
                        "text_candidate_crop": text_name,
                    }
                )
    finally:
        doc.close()

    payload = {
        "schema_version": 1,
        "phase": "phase5_cell_content_filter_diagnostic",
        "case_id": CASE_ID,
        "validation_status": "DIAGNOSTIC_ONLY",
        "ocr_used": False,
        "llm_used": False,
        "character_classification_performed": False,
        "cell_localization_revisited": False,
        "render_dpi": RENDER_DPI,
        "benchmark_real_frame_count": len(candidates),
        "content_cell_count": len(rows),
        "rows": rows,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _build_contact_sheet(rows)

    print("phase=phase5_cell_content_filter_diagnostic")
    print("validation_status=DIAGNOSTIC_ONLY")
    print("ocr_used=False")
    print("llm_used=False")
    print("character_classification_performed=False")
    print("cell_localization_revisited=False")
    print(f"content_cells={len(rows)}")
    print("\nbenchmark_real_frame_cell_content_filter:")
    for row in rows:
        print(
            f"  {row['candidate_id']} cell[{row['cell_index']}] role={row['expected_role']} "
            f"text_candidates={row['text_candidate_count']} "
            f"structural={row['structural_count']} arrows={row['arrow_like_count']} other={row['other_count']}"
        )
        for index, component in enumerate(row["components"], start=1):
            print(
                f"    {index}: class={component['component_class']} bbox={component['bbox_px']} "
                f"holes={component['hole_count']} solidity={component['solidity']:.3f} "
                f"vertices={component['approx_vertices']} reasons={component['reasons']}"
            )

    print(f"\noutput={OUTPUT_PATH}")
    print(f"contact_sheet={CONTACT_SHEET_PATH}")
    print(f"crops={OUTPUT_DIR}")


if __name__ == "__main__":
    main()
