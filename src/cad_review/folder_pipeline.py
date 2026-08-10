"""Generic per-PDF processor used by the folder validation batch.

This module is intentionally labelled validation-oriented while Phase 3 remains
open. Detector outputs are candidates, symbol scoring has no calibrated global
acceptance threshold, and unresolved fields stay unresolved.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import fitz

from src.cad_review.compliance_engine import build_cad_review_result
from src.cad_review.orchestrator import run_part_classification_branch
from src.cad_review.visual_output import render_visual_evidence
from src.gdt.datum_consistency import assess_referenced_datum_definitions
from src.gdt.datum_feature import detect_datum_feature_indicators
from src.gdt.detector import GdtFrameDetector
from src.gdt.frame_parser import parse_feature_control_frame
from src.gdt.iso1101_reference import assess_iso1101_datum_rule
from src.gdt.symbol_classifier import load_template_catalog, render_page_gray, score_candidates


def _bbox(value: Any) -> list[float]:
    return [round(float(v), 3) for v in value.to_list()]


def _template_catalog_status(template_root: Path, reference_catalog_path: Path) -> tuple[list, dict]:
    templates = load_template_catalog(template_root)
    classes = sorted({row.class_name for row in templates})
    expected: set[str] = {"position", "profile"}
    if reference_catalog_path.exists():
        config = json.loads(reference_catalog_path.read_text(encoding="utf-8"))
        expected.update(
            str(row.get("class_name", "")).strip().lower()
            for row in config.get("entries", [])
            if str(row.get("status", "active")).lower() == "active"
        )
    expected.discard("")
    missing = sorted(expected - set(classes))
    return templates, {
        "template_classes": classes,
        "expected_classes": sorted(expected),
        "missing_classes": missing,
        "complete": not missing,
        "threshold_calibrated": False,
    }


def process_cad_pdf(
    pdf_path: str | Path,
    *,
    output_dir: str | Path,
    classification_prompt: str,
    normas_path: str | Path,
    template_root: str | Path,
    iso1101_rules_path: str | Path,
    reference_catalog_path: str | Path,
    visual_dpi: int = 180,
    symbol_dpi: int = 300,
    allow_incomplete_symbol_catalog: bool = False,
) -> dict:
    """Process one CAD and write its JSON + visual evidence directory."""

    pdf = Path(pdf_path)
    cad_output = Path(output_dir)
    cad_output.mkdir(parents=True, exist_ok=True)
    pdf_bytes = pdf.read_bytes()

    part_branch = run_part_classification_branch(
        pdf_bytes,
        classification_prompt=classification_prompt,
        page_index=0,
        compressor_series_context="ALL",
        compressor_series_source="temporary_default_until_windchill",
        normas_path=Path(normas_path),
    )

    rules_config = json.loads(Path(iso1101_rules_path).read_text(encoding="utf-8"))
    iso_rules = rules_config.get("rules", [])
    templates, catalog_status = _template_catalog_status(Path(template_root), Path(reference_catalog_path))
    classification_enabled = bool(templates) and (catalog_status["complete"] or allow_incomplete_symbol_catalog)

    detector = GdtFrameDetector()
    raw_frames: list[dict] = []
    phase5_frames: list[dict] = []
    phase6_frames: list[dict] = []
    parsed_by_candidate: list[tuple[str, list[str]]] = []

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        page_count = len(doc)

    # First pass: definitions are drawing-level evidence. A datum referenced on
    # page 1 can legitimately be defined on page 2, so collect every definition
    # before evaluating any reference.
    datum_definitions: list[dict] = []
    for page_index in range(page_count):
        definitions = detect_datum_feature_indicators(pdf_bytes, page_index=page_index)
        datum_definitions.extend(row.to_dict() for row in definitions)

    # Second pass: GD&T detection / parsing / ISO 1101 assessment.
    for page_index in range(page_count):
        candidates = detector.detect_frames(pdf_bytes, page_index=page_index)
        score_rows: list[tuple[Any, Any]] = []
        if classification_enabled and candidates:
            page_gray, zoom = render_page_gray(pdf_bytes, page_index=page_index, dpi=symbol_dpi)
            score_rows = score_candidates(candidates, page_gray, zoom, templates)

        score_by_id = {}
        for candidate, scored in zip(candidates, score_rows):
            score, _crop = scored
            score_by_id[candidate.candidate_id] = score

        for candidate in candidates:
            score = score_by_id.get(candidate.candidate_id)
            characteristic = score.best_class if score is not None else None
            parsed = parse_feature_control_frame(candidate, characteristic=characteristic)
            parsed_dict = parsed.to_dict()

            candidate_row = {
                "candidate_id": candidate.candidate_id,
                "page": candidate.page,
                "frame_bbox": _bbox(candidate.frame_bbox),
                "symbol_bbox": _bbox(candidate.symbol_bbox),
                "cell_bboxes": [_bbox(cell.bbox) for cell in candidate.cells],
                "cell_texts": [list(cell.texts) for cell in candidate.cells],
                "detection_status": "candidate_unvalidated",
                "characteristic": characteristic,
                "referenced_datums": list(parsed.referenced_datums),
                "tolerance_raw": parsed.tolerance_raw,
                "tolerance_value": parsed.tolerance_value,
                "unresolved_fields": list(parsed.unresolved_fields),
                "symbol_scoring": (
                    {
                        **score.to_dict(),
                        "decision_policy": "ranking_only_no_global_threshold",
                        "catalog_complete": catalog_status["complete"],
                    }
                    if score is not None
                    else {
                        "best_class": None,
                        "decision_policy": "not_evaluated",
                        "reason": (
                            "symbol catalog incomplete"
                            if not catalog_status["complete"]
                            else "no active templates"
                        ),
                    }
                ),
            }
            raw_frames.append(candidate_row)
            phase5_frames.append({"candidate_id": candidate.candidate_id, "parsed": parsed_dict})
            parsed_by_candidate.append((candidate.candidate_id, list(parsed.referenced_datums)))

            iso_finding = assess_iso1101_datum_rule(
                characteristic=characteristic,
                referenced_datums=parsed.referenced_datums,
                rules=iso_rules,
                edition=int(rules_config.get("edition", 2017)),
                mode="reference",
            )
            phase6_frames.append({"candidate_id": candidate.candidate_id, "finding": iso_finding.to_dict()})

    phase7_frames: list[dict] = []
    for candidate_id, referenced_datums in parsed_by_candidate:
        datum_findings = assess_referenced_datum_definitions(
            referenced_datums=referenced_datums,
            defined_indicators=datum_definitions,
            mode="reference",
            standard="ISO 5459",
            source_ref="Datum Feature Indicator -> ISO 5459 (reference baseline; exact edition/clause unresolved)",
        )
        phase7_frames.append(
            {"candidate_id": candidate_id, "findings": [row.to_dict() for row in datum_findings]}
        )

    phase5_payload = {
        "phase": "batch_phase5_structured_candidates",
        "validation_status": "BATCH_VALIDATION_ONLY",
        "frames": phase5_frames,
    }
    phase6_payload = {
        "phase": "batch_phase6_iso1101_reference",
        "validation_status": "BATCH_VALIDATION_ONLY",
        "mode": "reference",
        "normative_applicability_established": False,
        "frames": phase6_frames,
    }
    phase7_payload = {
        "phase": "batch_phase7_datum_definition",
        "validation_status": "BATCH_VALIDATION_ONLY",
        "mode": "reference",
        "normative_applicability_established": False,
        "standard": "ISO 5459",
        "definitions": datum_definitions,
        "frames": phase7_frames,
    }

    integrated = build_cad_review_result(
        drawing={"name": pdf.name, "source_path": str(pdf)},
        part_branch=part_branch,
        phase5_payload=phase5_payload,
        phase6_payload=phase6_payload,
        phase7_payload=phase7_payload,
        compressor_series="ALL",
        compressor_series_source="temporary_default_until_windchill",
    ).model_dump()

    # Preserve detector geometry and scoring in the final contract; the generic
    # compliance adapter intentionally keeps only normalized Phase-5 fields.
    integrated["gdt_frames"] = raw_frames
    integrated["validation_status"] = "BATCH_VALIDATION_ONLY"
    integrated["production_claim"] = False
    integrated["limitations"] = [
        "Phase 3 multi-CAD validation is deferred.",
        "GD&T geometry items are detector candidates until independently validated.",
        "Symbol classification is ranking-only; no global acceptance threshold is calibrated.",
        "Datum references depend on recoverable frame content; unresolved content is not guessed.",
        "ISO 1101:2017 and ISO 5459 are used as reference baselines unless normative applicability is independently established.",
    ]
    integrated["symbol_catalog"] = catalog_status

    visual = render_visual_evidence(
        pdf_bytes,
        output_dir=cad_output,
        gdt_candidates=raw_frames,
        datum_definitions=datum_definitions,
        findings=integrated.get("findings", []),
        dpi=visual_dpi,
        save_crops=True,
    )
    integrated["artifacts"] = {"visual_evidence": visual}

    result_path = cad_output / "result.json"
    integrated["artifacts"]["result_json"] = result_path.name
    result_path.write_text(json.dumps(integrated, indent=2, ensure_ascii=False), encoding="utf-8")
    return integrated


__all__ = ["process_cad_pdf"]
