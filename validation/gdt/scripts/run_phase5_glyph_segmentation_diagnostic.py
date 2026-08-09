"""Fase 5: diagnóstico visual determinístico do conteúdo interno das células GD&T.

Motivação:
- a geometria dos quadros/células já é conhecida;
- a camada textual invisível do PDF não cai espacialmente sobre o conteúdo dos
  quadros GD&T no caso 41;
- o conteúdo visível das células existe como desenho vetorial e pode ser
  rasterizado em alta resolução sem perda relevante de forma.

Este script NÃO faz OCR, NÃO usa LLM e NÃO classifica caracteres.
Ele apenas:
1. recorta o INTERIOR das células 1+ (exclui a célula do símbolo GD&T);
2. remove as bordas do frame por inset geométrico antes da rasterização;
3. binariza o crop;
4. segmenta componentes conectados internos;
5. salva crops limpos, componentes individuais e uma contact sheet.

Uso:
    python validation/gdt/scripts/run_phase5_glyph_segmentation_diagnostic.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import fitz
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gdt.detector import GdtFrameDetector

CASE_ID = "case_41_rev8"
CASE_PATH = PROJECT_ROOT / "validation" / "gdt" / "cases" / f"{CASE_ID}.json"
GEOMETRY_BASELINE = PROJECT_ROOT / "validation" / "gdt" / "baselines" / f"{CASE_ID}.geometry.json"
OUTPUT_DIR = PROJECT_ROOT / "validation" / "gdt" / "outputs" / "phase5" / CASE_ID / "glyph_segmentation"
OUTPUT_PATH = OUTPUT_DIR / "glyph_segmentation.json"
CONTACT_SHEET_PATH = OUTPUT_DIR / "glyph_segmentation_contact_sheet.png"

RENDER_DPI = 1200
CELL_INSET_PT = 0.8
MIN_COMPONENT_AREA_PX = 8
MAX_COMPONENT_AREA_FRACTION = 0.90
CONTACT_TILE_WIDTH = 620
CONTACT_TILE_HEIGHT = 250


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _inset_bbox(bbox, amount: float) -> fitz.Rect:
    x0, y0, x1, y1 = [float(v) for v in bbox.to_list()]
    if x1 - x0 <= 2 * amount or y1 - y0 <= 2 * amount:
        return fitz.Rect(x0, y0, x1, y1)
    return fitz.Rect(x0 + amount, y0 + amount, x1 - amount, y1 - amount)


def _render_gray(page: fitz.Page, rect: fitz.Rect) -> np.ndarray:
    zoom = RENDER_DPI / 72.0
    pix = page.get_pixmap(
        matrix=fitz.Matrix(zoom, zoom),
        clip=rect,
        colorspace=fitz.csGRAY,
        alpha=False,
    )
    image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    return image.copy()


def _binarize(gray: np.ndarray) -> np.ndarray:
    # Vetores pretos em fundo branco. THRESH_BINARY_INV deixa tinta=255.
    _threshold, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
    )
    return binary


def _component_rows(binary: np.ndarray) -> tuple[list[dict], np.ndarray]:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    height, width = binary.shape[:2]
    image_area = float(max(1, width * height))

    rows: list[dict] = []
    for label in range(1, count):
        x, y, w, h, area = [int(v) for v in stats[label]]
        if area < MIN_COMPONENT_AREA_PX:
            continue
        if area / image_area > MAX_COMPONENT_AREA_FRACTION:
            continue

        cx, cy = [float(v) for v in centroids[label]]
        rows.append(
            {
                "label": label,
                "bbox_px": [x, y, x + w, y + h],
                "width_px": w,
                "height_px": h,
                "area_px": area,
                "area_fraction": round(area / image_area, 6),
                "centroid_px": [round(cx, 2), round(cy, 2)],
                "center_x_fraction": round(cx / max(1.0, width), 4),
                "center_y_fraction": round(cy / max(1.0, height), 4),
            }
        )

    rows.sort(key=lambda row: (row["bbox_px"][0], row["bbox_px"][1]))
    return rows, labels


def _save_binary(binary: np.ndarray, path: Path) -> None:
    # Para inspeção humana: fundo branco, tinta preta.
    visible = 255 - binary
    cv2.imwrite(str(path), visible)


def _save_annotated(gray: np.ndarray, components: list[dict], path: Path) -> None:
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for index, row in enumerate(components, start=1):
        x0, y0, x1, y1 = row["bbox_px"]
        cv2.rectangle(rgb, (x0, y0), (x1 - 1, y1 - 1), (0, 0, 255), 2)
        cv2.putText(
            rgb,
            str(index),
            (x0, max(14, y0 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(path), rgb)


def _save_components(binary: np.ndarray, components: list[dict], prefix: str) -> list[str]:
    names: list[str] = []
    for index, row in enumerate(components, start=1):
        x0, y0, x1, y1 = row["bbox_px"]
        pad = 5
        sx0 = max(0, x0 - pad)
        sy0 = max(0, y0 - pad)
        sx1 = min(binary.shape[1], x1 + pad)
        sy1 = min(binary.shape[0], y1 + pad)
        crop = 255 - binary[sy0:sy1, sx0:sx1]
        name = f"{prefix}_component_{index:02d}.png"
        cv2.imwrite(str(OUTPUT_DIR / name), crop)
        row["component_crop"] = name
        names.append(name)
    return names


def _fit_image(image: Image.Image, max_width: int, max_height: int) -> Image.Image:
    if image.width <= 0 or image.height <= 0:
        return image
    scale = min(max_width / image.width, max_height / image.height)
    scale = max(scale, 0.01)
    size = (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale))))
    return image.resize(size, Image.Resampling.LANCZOS)


def _build_contact_sheet(rows: list[dict]) -> None:
    if not rows:
        return

    width = CONTACT_TILE_WIDTH
    height = CONTACT_TILE_HEIGHT * len(rows)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for idx, row in enumerate(rows):
        top = idx * CONTACT_TILE_HEIGHT
        title = (
            f"{row['candidate_id']} cell[{row['cell_index']}] "
            f"role={row['expected_role']} components={row['component_count']}"
        )
        draw.text((10, top + 8), title, fill="black", font=font)

        binary_img = Image.open(OUTPUT_DIR / row["binary_crop"]).convert("RGB")
        annotated_img = Image.open(OUTPUT_DIR / row["annotated_crop"]).convert("RGB")
        fitted_binary = _fit_image(binary_img, 280, 175)
        fitted_annotated = _fit_image(annotated_img, 280, 175)

        sheet.paste(fitted_binary, (10, top + 45))
        sheet.paste(fitted_annotated, (320, top + 45))
        draw.text((10, top + 225), "clean", fill="black", font=font)
        draw.text((320, top + 225), "components", fill="black", font=font)

        if idx < len(rows) - 1:
            draw.line((0, top + CONTACT_TILE_HEIGHT - 1, width, top + CONTACT_TILE_HEIGHT - 1), fill="gray")

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
    candidates = detector.detect_frames(pdf_bytes, page_index=page_index)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        page = doc[page_index]
        for candidate in candidates:
            if candidate.candidate_id not in benchmark_ids:
                continue

            for cell_index, cell in enumerate(candidate.cells[1:], start=1):
                rect = _inset_bbox(cell.bbox, CELL_INSET_PT)
                gray = _render_gray(page, rect)
                binary = _binarize(gray)
                components, _labels = _component_rows(binary)

                prefix = f"{candidate.candidate_id}_cell_{cell_index:02d}"
                gray_name = f"{prefix}_gray.png"
                binary_name = f"{prefix}_clean.png"
                annotated_name = f"{prefix}_components.png"

                cv2.imwrite(str(OUTPUT_DIR / gray_name), gray)
                _save_binary(binary, OUTPUT_DIR / binary_name)
                _save_annotated(gray, components, OUTPUT_DIR / annotated_name)
                _save_components(binary, components, prefix)

                ink_pixels = int(np.count_nonzero(binary))
                image_area = int(binary.shape[0] * binary.shape[1])
                expected_role = "tolerance" if cell_index == 1 else "datum_or_modifier"

                results.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "cell_index": cell_index,
                        "expected_role": expected_role,
                        "original_bbox": [round(v, 4) for v in cell.bbox.to_list()],
                        "inset_bbox": [round(float(v), 4) for v in (rect.x0, rect.y0, rect.x1, rect.y1)],
                        "render_dpi": RENDER_DPI,
                        "image_size_px": [int(binary.shape[1]), int(binary.shape[0])],
                        "ink_pixels": ink_pixels,
                        "ink_fraction": round(ink_pixels / max(1, image_area), 6),
                        "component_count": len(components),
                        "components": components,
                        "gray_crop": gray_name,
                        "binary_crop": binary_name,
                        "annotated_crop": annotated_name,
                    }
                )
    finally:
        doc.close()

    payload = {
        "schema_version": 1,
        "phase": "phase5_glyph_segmentation_diagnostic",
        "case_id": CASE_ID,
        "validation_status": "DIAGNOSTIC_ONLY",
        "ocr_used": False,
        "llm_used": False,
        "classification_performed": False,
        "render_dpi": RENDER_DPI,
        "cell_inset_pt": CELL_INSET_PT,
        "min_component_area_px": MIN_COMPONENT_AREA_PX,
        "benchmark_real_frame_count": len(benchmark_ids),
        "content_cell_count": len(results),
        "results": results,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _build_contact_sheet(results)

    print("phase=phase5_glyph_segmentation_diagnostic")
    print("validation_status=DIAGNOSTIC_ONLY")
    print("ocr_used=False")
    print("llm_used=False")
    print("classification_performed=False")
    print(f"render_dpi={RENDER_DPI}")
    print(f"cell_inset_pt={CELL_INSET_PT}")
    print(f"content_cells={len(results)}")
    print("\nbenchmark_real_frame_glyph_segmentation:")
    for row in results:
        component_summary = [
            f"{index}:bbox={component['bbox_px']} area={component['area_px']}"
            for index, component in enumerate(row["components"], start=1)
        ]
        print(
            f"  {row['candidate_id']} cell[{row['cell_index']}] "
            f"role={row['expected_role']} components={row['component_count']} "
            f"ink_fraction={row['ink_fraction']:.4f}"
        )
        if component_summary:
            print("    " + "; ".join(component_summary))
        else:
            print("    no_components")

    print(f"\noutput={OUTPUT_PATH}")
    print(f"contact_sheet={CONTACT_SHEET_PATH}")
    print(f"crops={OUTPUT_DIR}")


if __name__ == "__main__":
    main()
