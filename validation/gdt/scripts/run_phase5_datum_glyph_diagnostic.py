"""Phase 5 diagnostic: deterministic recognition of isolated datum letters.

This script consumes the output of the immediately preceding Phase 5 cell
content filter. It does NOT re-run connected-component classification.

That separation is intentional:
- cell-content filtering decides which visible components are plausible text;
- this diagnostic receives the saved ``text candidates only`` mask;
- datum recognition normalizes and ranks that already-filtered glyph.

Methodology for case 41 remains bootstrap-only:
- one visually labelled A cell is registered as the A template;
- the other two visually labelled A cells are holdouts;
- B and D each have only one real example in this CAD, so they are template
  sources only and are NOT counted as independent validation.

No OCR and no LLM are used.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gdt.datum_glyph import DatumGlyphTemplateClassifier, normalize_glyph_mask

CASE_ID = "case_41_rev8"
CONFIG_PATH = PROJECT_ROOT / "validation" / "gdt" / "configs" / "case_41_datum_bootstrap.json"
FILTER_DIR = PROJECT_ROOT / "validation" / "gdt" / "outputs" / "phase5" / CASE_ID / "cell_content_filter"
FILTER_OUTPUT_PATH = FILTER_DIR / "cell_content_filter.json"
OUTPUT_DIR = PROJECT_ROOT / "validation" / "gdt" / "outputs" / "phase5" / CASE_ID / "datum_glyph"
OUTPUT_PATH = OUTPUT_DIR / "datum_glyph_diagnostic.json"
CONTACT_SHEET_PATH = OUTPUT_DIR / "datum_glyph_contact_sheet.png"

EXPECTED_RENDER_DPI = 1200
CANVAS_SIZE = 96
PADDING = 10


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _cell_key(candidate_id: str, cell_index: int) -> str:
    return f"{candidate_id}:cell[{cell_index}]"


def _load_filter_payload() -> dict:
    if not FILTER_OUTPUT_PATH.exists():
        raise FileNotFoundError(
            "Phase 5 cell-content filter output not found. Run first: "
            "python validation/gdt/scripts/run_phase5_cell_content_filter_diagnostic.py"
        )

    payload = _load(FILTER_OUTPUT_PATH)
    if payload.get("case_id") != CASE_ID:
        raise ValueError(
            f"cell-content filter case mismatch: expected {CASE_ID!r}, "
            f"got {payload.get('case_id')!r}"
        )
    if int(payload.get("render_dpi", -1)) != EXPECTED_RENDER_DPI:
        raise ValueError(
            "cell-content filter render DPI mismatch: "
            f"expected {EXPECTED_RENDER_DPI}, got {payload.get('render_dpi')!r}"
        )
    return payload


def _load_text_candidate_mask(row: dict) -> np.ndarray:
    """Load the exact candidate-only mask produced by the previous diagnostic.

    ``run_phase5_cell_content_filter_diagnostic.py`` stores it for human viewing
    as black ink on white background. Convert it back to binary ink=255 here.
    """

    name = row.get("text_candidate_crop")
    if not name:
        raise ValueError("filter row has no text_candidate_crop")
    path = FILTER_DIR / str(name)
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"filtered text-candidate crop not found: {path}")
    _threshold, mask = cv2.threshold(
        image,
        0,
        255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
    )
    return mask


def _component_count(mask: np.ndarray) -> int:
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    valid = 0
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= 8:
            valid += 1
    return valid


def _sample_from_filter(row: dict) -> dict:
    filter_candidate_count = int(row.get("text_candidate_count", 0))
    mask = _load_text_candidate_mask(row)
    mask_component_count = _component_count(mask)
    ink_pixels = int(np.count_nonzero(mask))

    status = "ok"
    if filter_candidate_count != 1:
        status = "ambiguous_filter_candidate_count"
    elif ink_pixels == 0:
        status = "empty_filtered_mask"

    normalized = None
    if status == "ok":
        # Normalize the complete candidate-only mask. Do not re-run the visual
        # component classifier here: the previous stage already made that
        # decision and saved the exact resulting mask.
        normalized = normalize_glyph_mask(
            mask,
            canvas_size=CANVAS_SIZE,
            padding=PADDING,
        )

    return {
        "status": status,
        "cell_bbox": row.get("cell_bbox"),
        "image_size_px": row.get("image_size_px"),
        "filter_component_count": int(row.get("component_count", 0)),
        "text_candidate_count": filter_candidate_count,
        "filtered_mask_component_count": mask_component_count,
        "filtered_mask_ink_pixels": ink_pixels,
        "filter_components": row.get("components", []),
        "filter_text_candidate_crop": row.get("text_candidate_crop"),
        "filtered_mask": mask,
        "normalized": normalized,
    }


def _save_mask(mask: np.ndarray, path: Path) -> None:
    cv2.imwrite(str(path), 255 - mask)


def _build_contact_sheet(rows: list[dict]) -> None:
    if not rows:
        return

    tile_width = 820
    tile_height = 190
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

        filtered_path = row.get("filter_text_candidate_crop")
        if filtered_path:
            image = Image.open(FILTER_DIR / filtered_path).convert("RGB")
            image.thumbnail((145, 135), Image.Resampling.NEAREST)
            sheet.paste(image, (15, top + 38))
            draw.text((15, top + 170), "filtered mask", fill="black", font=font)

        norm_path = row.get("normalized_crop")
        if norm_path:
            image = Image.open(OUTPUT_DIR / norm_path).convert("RGB")
            image = image.resize((128, 128), Image.Resampling.NEAREST)
            sheet.paste(image, (175, top + 38))
            draw.text((175, top + 170), "normalized", fill="black", font=font)

        y = top + 42
        for rank, match in enumerate(row.get("ranking", [])[:3], start=1):
            text = (
                f"#{rank} {match['label']} score={match['score']:.4f} "
                f"dice={match['dice']:.3f} chamfer={match['chamfer']:.3f} "
                f"contour={match['contour']:.3f} holes={match['hole_agreement']:.0f}"
            )
            draw.text((330, y), text, fill="black", font=font)
            y += 27

        if index < len(rows) - 1:
            draw.line((0, top + tile_height - 1, tile_width, top + tile_height - 1), fill="gray")

    sheet.save(CONTACT_SHEET_PATH)


def main() -> None:
    config = _load(CONFIG_PATH)
    filter_payload = _load_filter_payload()
    filter_rows = {
        _cell_key(row["candidate_id"], int(row["cell_index"])): row
        for row in filter_payload.get("rows", [])
    }

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
    for key, requested in requested_cells.items():
        filter_row = filter_rows.get(key)
        if filter_row is None:
            sample = {
                "status": "filter_row_not_found",
                "text_candidate_count": 0,
                "filtered_mask_component_count": 0,
                "filtered_mask_ink_pixels": 0,
                "filter_components": [],
                "filtered_mask": None,
                "normalized": None,
            }
        else:
            sample = _sample_from_filter(filter_row)

        sample["candidate_id"] = requested["candidate_id"]
        sample["cell_index"] = int(requested["cell_index"])
        sample["cell_key"] = key
        sample["expected_label"] = str(requested["label"]).upper()
        sample["evaluation_role"] = requested.get("evaluation_role", "diagnostic")
        samples[key] = sample

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
        filtered_mask = sample.pop("filtered_mask", None)

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
        "schema_version": 2,
        "phase": "phase5_datum_glyph_diagnostic",
        "case_id": CASE_ID,
        "validation_status": "DIAGNOSTIC_ONLY",
        "ocr_used": False,
        "llm_used": False,
        "input_stage": "phase5_cell_content_filter_diagnostic",
        "resegmentation_performed": False,
        "template_methodology": config.get("methodology"),
        "template_sources_are_validation": False,
        "acceptance_threshold_calibrated": False,
        "render_dpi": EXPECTED_RENDER_DPI,
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
    print("input_stage=phase5_cell_content_filter_diagnostic")
    print("resegmentation_performed=False")
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
            f"role={row['evaluation_role']} filter_candidates={row['text_candidate_count']} "
            f"mask_components={row['filtered_mask_component_count']} "
            f"predicted={row['predicted_label'] or '-'} margin={margin_text} "
            f"status={row['status']} ranking=[{ranking_text}]"
        )
        if row["status"] != "ok":
            rejected = [
                f"{component.get('component_class')}:{component.get('bbox_px')}"
                for component in row.get("filter_components", [])
            ]
            print("    filter_components=" + ", ".join(rejected))

    print(f"\noutput={OUTPUT_PATH}")
    print(f"contact_sheet={CONTACT_SHEET_PATH}")


if __name__ == "__main__":
    main()
