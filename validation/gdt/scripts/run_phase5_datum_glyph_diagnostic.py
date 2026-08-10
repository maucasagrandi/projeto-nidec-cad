"""Phase 5 diagnostic: deterministic recognition of isolated datum letters.

This script starts from the already-detected datum cells. It does not revisit
frame/cell localization and it does not use OCR or an LLM.

Methodology for case 41 is explicitly bootstrap-only:
- one visually labelled A cell is registered as the A template;
- the other two visually labelled A cells are holdouts;
- B and D each have only one real example in this CAD, so they are template
  sources only and are NOT counted as independent validation.

The goal is to validate the normalization/scoring path without inflating an
accuracy claim from self-matches.
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

from src.gdt.cell_visual_content import analyze_components, binarize_cell
from src.gdt.datum_glyph import DatumGlyphTemplateClassifier, normalized_component_from_cell
from src.gdt.detector import GdtFrameDetector

CASE_ID = "case_41_rev8"
CASE_PATH = PROJECT_ROOT / "validation" / "gdt" / "cases" / f"{CASE_ID}.json"
CONFIG_PATH = PROJECT_ROOT / "validation" / "gdt" / "configs" / "case_41_datum_bootstrap.json"
OUTPUT_DIR = PROJECT_ROOT / "validation" / "gdt" / "outputs" / "phase5" / CASE_ID / "datum_glyph"
OUTPUT_PATH = OUTPUT_DIR / "datum_glyph_diagnostic.json"
CONTACT_SHEET_PATH = OUTPUT_DIR / "datum_glyph_contact_sheet.png"

RENDER_DPI = 1200
CANVAS_SIZE = 96
PADDING = 10


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


def _cell_key(candidate_id: str, cell_index: int) -> str:
    return f"{candidate_id}:cell[{cell_index}]"


def _extract_cell_sample(page: fitz.Page, candidate, cell_index: int) -> dict:
    if cell_index < 0 or cell_index >= len(candidate.cells):
        return {
            "status": "invalid_cell_index",
            "text_candidate_count": 0,
        }

    cell = candidate.cells[cell_index]
    gray = _render_gray(page, cell.bbox)
    binary = binarize_cell(gray)
    components = analyze_components(binary)
    text_candidates = [row for row in components if row.component_class == "text_candidate"]

    result = {
        "status": "ok" if len(text_candidates) == 1 else "ambiguous_component_count",
        "cell_bbox": [round(v, 4) for v in cell.bbox.to_list()],
        "image_size_px": [int(binary.shape[1]), int(binary.shape[0])],
        "component_count": len(components),
        "text_candidate_count": len(text_candidates),
        "components": [row.to_dict() for row in components],
        "binary": binary,
        "normalized": None,
    }

    if len(text_candidates) == 1:
        normalized = normalized_component_from_cell(
            binary,
            text_candidates[0],
            canvas_size=CANVAS_SIZE,
            padding=PADDING,
        )
        result["normalized"] = normalized
    return result


def _save_mask(mask: np.ndarray, path: Path) -> None:
    cv2.imwrite(str(path), 255 - mask)


def _build_contact_sheet(rows: list[dict]) -> None:
    if not rows:
        return

    tile_width = 760
    tile_height = 180
    sheet = Image.new("RGB", (tile_width, tile_height * len(rows)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for index, row in enumerate(rows):
        top = index * tile_height
        expected = row["expected_label"]
        predicted = row.get("predicted_label") or "-"
        role = row["evaluation_role"]
        status = row["status"]
        margin = row.get("margin")
        margin_text = "-" if margin is None else f"{margin:.4f}"
        title = (
            f"{row['cell_key']} expected={expected} predicted={predicted} "
            f"role={role} status={status} margin={margin_text}"
        )
        draw.text((10, top + 8), title, fill="black", font=font)

        norm_path = row.get("normalized_crop")
        if norm_path:
            image = Image.open(OUTPUT_DIR / norm_path).convert("RGB")
            image = image.resize((128, 128), Image.Resampling.NEAREST)
            sheet.paste(image, (20, top + 38))

        y = top + 42
        for rank, match in enumerate(row.get("ranking", [])[:3], start=1):
            text = (
                f"#{rank} {match['label']} score={match['score']:.4f} "
                f"dice={match['dice']:.3f} chamfer={match['chamfer']:.3f} "
                f"contour={match['contour']:.3f} holes={match['hole_agreement']:.0f}"
            )
            draw.text((180, y), text, fill="black", font=font)
            y += 27

        if index < len(rows) - 1:
            draw.line((0, top + tile_height - 1, tile_width, top + tile_height - 1), fill="gray")

    sheet.save(CONTACT_SHEET_PATH)


def main() -> None:
    case = _load(CASE_PATH)
    config = _load(CONFIG_PATH)
    pdf_path = PROJECT_ROOT / case["pdf"]
    page_index = int(case.get("page_index", 0))
    pdf_bytes = pdf_path.read_bytes()

    detector = GdtFrameDetector()
    candidates = detector.detect_frames(pdf_bytes, page_index=page_index)
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    requested_cells: dict[str, dict] = {}
    for row in config.get("expected_datum_cells", []):
        requested_cells[_cell_key(row["candidate_id"], int(row["cell_index"]))] = row
    for row in config.get("template_sources", []):
        requested_cells.setdefault(
            _cell_key(row["candidate_id"], int(row["cell_index"])),
            {
                "candidate_id": row["candidate_id"],
                "cell_index": int(row["cell_index"]),
                "label": row["label"],
                "evaluation_role": "template_source",
            },
        )

    samples: dict[str, dict] = {}
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        page = doc[page_index]
        for key, row in requested_cells.items():
            candidate = candidate_by_id.get(row["candidate_id"])
            if candidate is None:
                sample = {"status": "candidate_not_found", "text_candidate_count": 0}
            else:
                sample = _extract_cell_sample(page, candidate, int(row["cell_index"]))
            sample["candidate_id"] = row["candidate_id"]
            sample["cell_index"] = int(row["cell_index"])
            sample["cell_key"] = key
            sample["expected_label"] = str(row["label"]).upper()
            sample["evaluation_role"] = row.get("evaluation_role", "diagnostic")
            samples[key] = sample
    finally:
        doc.close()

    classifier = DatumGlyphTemplateClassifier()
    template_rows = []
    for source in config.get("template_sources", []):
        key = _cell_key(source["candidate_id"], int(source["cell_index"]))
        sample = samples.get(key)
        status = "registered"
        if sample is None or sample.get("normalized") is None:
            status = "unavailable"
        else:
            classifier.register(
                str(source["label"]),
                sample["normalized"],
                source_id=key,
            )
        template_rows.append(
            {
                "label": str(source["label"]).upper(),
                "cell_key": key,
                "status": status,
            }
        )

    result_rows: list[dict] = []
    holdout_total = 0
    holdout_resolved = 0
    holdout_ranking_correct = 0

    for key, sample in samples.items():
        normalized = sample.pop("normalized", None)
        binary = sample.pop("binary", None)

        normalized_name = None
        if normalized is not None:
            normalized_name = key.replace(":", "_").replace("[", "_").replace("]", "") + "_normalized.png"
            _save_mask(normalized, OUTPUT_DIR / normalized_name)

        ranking = classifier.rank(normalized) if normalized is not None else []
        ranking_dicts = [match.to_dict() for match in ranking]
        predicted = ranking[0].label if ranking else None
        margin = None
        if ranking:
            second = ranking[1].score if len(ranking) > 1 else 0.0
            margin = float(ranking[0].score - second)

        row = {
            **sample,
            "normalized_crop": normalized_name,
            "predicted_label": predicted,
            "ranking": ranking_dicts,
            "margin": margin,
            "ranking_correct": predicted == sample["expected_label"] if predicted is not None else False,
        }
        result_rows.append(row)

        if sample["evaluation_role"] == "holdout":
            holdout_total += 1
            if ranking:
                holdout_resolved += 1
                if predicted == sample["expected_label"]:
                    holdout_ranking_correct += 1

    payload = {
        "schema_version": 1,
        "phase": "phase5_datum_glyph_diagnostic",
        "case_id": CASE_ID,
        "validation_status": "DIAGNOSTIC_ONLY",
        "ocr_used": False,
        "llm_used": False,
        "template_methodology": config.get("methodology"),
        "template_sources_are_validation": False,
        "acceptance_threshold_calibrated": False,
        "render_dpi": RENDER_DPI,
        "canvas_size": CANVAS_SIZE,
        "padding": PADDING,
        "registered_template_count": classifier.template_count,
        "registered_labels": list(classifier.labels),
        "template_sources": template_rows,
        "holdout_total": holdout_total,
        "holdout_resolved": holdout_resolved,
        "holdout_ranking_correct": holdout_ranking_correct,
        "holdout_ranking_accuracy": (
            holdout_ranking_correct / holdout_resolved if holdout_resolved else None
        ),
        "results": result_rows,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _build_contact_sheet(result_rows)

    print("phase=phase5_datum_glyph_diagnostic")
    print("validation_status=DIAGNOSTIC_ONLY")
    print("ocr_used=False")
    print("llm_used=False")
    print("template_sources_are_validation=False")
    print("acceptance_threshold_calibrated=False")
    print(f"registered_templates={classifier.template_count} labels={list(classifier.labels)}")
    print(
        f"holdout={holdout_ranking_correct}/{holdout_resolved} "
        f"resolved (total_holdout={holdout_total})"
    )
    print("\nbenchmark_real_frame_datum_glyphs:")
    for row in result_rows:
        ranking_text = ", ".join(
            f"{match['label']}:{match['score']:.3f}"
            for match in row["ranking"][:3]
        ) or "-"
        margin = row["margin"]
        margin_text = "-" if margin is None else f"{margin:.3f}"
        print(
            f"  {row['cell_key']} expected={row['expected_label']} "
            f"role={row['evaluation_role']} candidates={row['text_candidate_count']} "
            f"predicted={row['predicted_label'] or '-'} margin={margin_text} ranking=[{ranking_text}]"
        )

    print(f"\noutput={OUTPUT_PATH}")
    print(f"contact_sheet={CONTACT_SHEET_PATH}")


if __name__ == "__main__":
    main()
