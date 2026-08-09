"""Gera uma folha de contato legível das células internas dos quadros GD&T reais.

Escopo:
- usa apenas os seis frames do benchmark geométrico do caso 41;
- ignora cell[0], pois a característica já é tratada pela Fase 4;
- recorta o interior de cell[1:] com pequeno inset para remover as bordas;
- renderiza em alta resolução e monta uma única imagem para revisão humana;
- NÃO usa OCR, LLM ou classificação automática do conteúdo.

Uso:
    python validation/gdt/scripts/build_phase5_cell_contact_sheet.py
"""

from __future__ import annotations

import json
import sys
from io import BytesIO
from pathlib import Path

import fitz
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gdt.detector import GdtFrameDetector

CASE_ID = "case_41_rev8"
CASE_PATH = PROJECT_ROOT / "validation" / "gdt" / "cases" / f"{CASE_ID}.json"
GEOMETRY_BASELINE = PROJECT_ROOT / "validation" / "gdt" / "baselines" / f"{CASE_ID}.geometry.json"
OUTPUT_DIR = PROJECT_ROOT / "validation" / "gdt" / "outputs" / "phase5" / CASE_ID
OUTPUT_PATH = OUTPUT_DIR / "phase5_cell_contact_sheet.png"
CROP_DIR = OUTPUT_DIR / "interior_cells"
CROP_DPI = 900
CELL_INSET_PT = 0.8
TARGET_HEIGHT = 150
LABEL_WIDTH = 310
ROW_PADDING = 12
BACKGROUND = 255


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _inner_rect(bbox) -> fitz.Rect:
    rect = fitz.Rect(bbox.to_list())
    inset = min(CELL_INSET_PT, rect.width * 0.12, rect.height * 0.12)
    inner = fitz.Rect(
        rect.x0 + inset,
        rect.y0 + inset,
        rect.x1 - inset,
        rect.y1 - inset,
    )
    if inner.width <= 1 or inner.height <= 1:
        return rect
    return inner


def _render_crop(page: fitz.Page, bbox) -> Image.Image:
    zoom = CROP_DPI / 72.0
    pix = page.get_pixmap(
        matrix=fitz.Matrix(zoom, zoom),
        clip=_inner_rect(bbox),
        colorspace=fitz.csGRAY,
        alpha=False,
    )
    image = Image.open(BytesIO(pix.tobytes("png"))).convert("L")
    scale = TARGET_HEIGHT / max(1, image.height)
    width = max(1, int(round(image.width * scale)))
    return image.resize((width, TARGET_HEIGHT), Image.Resampling.LANCZOS)


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
    CROP_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, Image.Image]] = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        page = doc[page_index]
        for candidate in candidates:
            if candidate.candidate_id not in benchmark_ids:
                continue
            for index, cell in enumerate(candidate.cells[1:], start=1):
                crop = _render_crop(page, cell.bbox)
                crop_name = f"{candidate.candidate_id}_cell_{index:02d}_interior.png"
                crop.save(CROP_DIR / crop_name)
                label = f"{candidate.candidate_id}  cell[{index}]"
                rows.append((label, crop))
    finally:
        doc.close()

    if not rows:
        raise RuntimeError("Nenhuma célula interna encontrada para o benchmark.")

    max_crop_width = max(image.width for _, image in rows)
    row_height = TARGET_HEIGHT + 2 * ROW_PADDING
    sheet_width = LABEL_WIDTH + max_crop_width + 3 * ROW_PADDING
    header_height = 56
    sheet_height = header_height + row_height * len(rows)

    sheet = Image.new("L", (sheet_width, sheet_height), BACKGROUND)
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((ROW_PADDING, 14), "Phase 5 - case 41 - interior GD&T cells (cell[1:])", fill=0, font=font)

    y = header_height
    for label, crop in rows:
        draw.text((ROW_PADDING, y + TARGET_HEIGHT // 2), label, fill=0, font=font, anchor="lm")
        x = LABEL_WIDTH + ROW_PADDING
        sheet.paste(crop, (x, y + ROW_PADDING))
        y += row_height

    sheet.save(OUTPUT_PATH)

    print("phase=phase5_cell_contact_sheet")
    print("validation_status=HUMAN_REVIEW_INPUT")
    print("ocr_used=False")
    print("llm_used=False")
    print(f"benchmark_internal_cells={len(rows)}")
    print(f"crop_dpi={CROP_DPI}")
    print(f"cell_inset_pt={CELL_INSET_PT}")
    print(f"contact_sheet={OUTPUT_PATH}")
    print(f"individual_crops={CROP_DIR}")


if __name__ == "__main__":
    main()
