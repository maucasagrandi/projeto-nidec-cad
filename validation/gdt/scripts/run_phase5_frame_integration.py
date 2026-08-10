"""Final case-41 integration diagnostic for Phase 5 GD&T frame content.

This script combines the stable pieces developed during Phase 5:

- detected frame / cell geometry;
- characteristic label (from independent case-41 GT *only to isolate Phase 5*);
- visual tolerance-cell assessment from ``cell_content_filter.json``;
- visual datum ranking from ``datum_glyph_diagnostic.json``;
- ``frame_parser`` merge / unresolved-field contract.

It does NOT perform ISO compliance, datum-definition validation, OCR or LLM
inference. It also does not claim an end-to-end symbol-classification metric:
characteristics are supplied from independent GT because Phase 2/4 already
validate that subsystem separately.

Two gates are intentionally distinct:

``integration_gate``
    PASS when every benchmark frame can be converted to one structured
    ``ParsedGdtFrame`` without guessing missing values.

``content_complete``
    True only when no parsed frame has unresolved fields. An unresolved
    tolerance therefore does not make integration fail; it remains explicit
    evidence that Phase 5 could not recover that field from this drawing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gdt.detector import GdtFrameDetector
from src.gdt.frame_parser import FrameVisualEvidence, parse_feature_control_frame
from src.gdt.tolerance_cell import assessment_from_filter_row

CASE_ID = "case_41_rev8"
CASE_PATH = PROJECT_ROOT / "validation" / "gdt" / "cases" / f"{CASE_ID}.json"
GT_PATH = PROJECT_ROOT / "validation" / "gdt" / "ground_truth" / f"{CASE_ID}.json"
GEOMETRY_BASELINE = PROJECT_ROOT / "validation" / "gdt" / "baselines" / f"{CASE_ID}.geometry.json"
CELL_FILTER_PATH = (
    PROJECT_ROOT
    / "validation"
    / "gdt"
    / "outputs"
    / "phase5"
    / CASE_ID
    / "cell_content_filter"
    / "cell_content_filter.json"
)
DATUM_DIAGNOSTIC_PATH = (
    PROJECT_ROOT
    / "validation"
    / "gdt"
    / "outputs"
    / "phase5"
    / CASE_ID
    / "datum_glyph"
    / "datum_glyph_diagnostic.json"
)
OUTPUT_DIR = PROJECT_ROOT / "validation" / "gdt" / "outputs" / "phase5" / CASE_ID / "frame_integration"
OUTPUT_PATH = OUTPUT_DIR / "phase5_structured_frames.json"


def _load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"required Phase 5 artifact not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_artifact(payload: dict, *, phase: str) -> None:
    if payload.get("case_id") != CASE_ID:
        raise ValueError(f"artifact case mismatch: expected {CASE_ID}, got {payload.get('case_id')}")
    if payload.get("phase") != phase:
        raise ValueError(f"artifact phase mismatch: expected {phase}, got {payload.get('phase')}")


def _characteristics_by_candidate(gt: dict, baseline: dict) -> dict[str, str]:
    gt_by_id = {
        row["id"]: str(row["characteristic"])
        for row in gt.get("frames", [])
        if row.get("id") and row.get("characteristic")
    }
    output: dict[str, str] = {}
    for match in baseline.get("matches", []):
        candidate_id = match.get("candidate_id")
        gt_id = match.get("ground_truth_id")
        if candidate_id and gt_id in gt_by_id:
            output[str(candidate_id)] = gt_by_id[str(gt_id)]
    return output


def _filter_rows_by_key(payload: dict) -> dict[str, dict]:
    rows = {}
    for row in payload.get("rows", []):
        candidate_id = row.get("candidate_id")
        cell_index = row.get("cell_index")
        if candidate_id is None or cell_index is None:
            continue
        rows[f"{candidate_id}:cell[{int(cell_index)}]"] = row
    return rows


def _datum_rows_by_candidate(payload: dict) -> dict[str, list[dict]]:
    output: dict[str, list[dict]] = {}
    for row in payload.get("results", []):
        candidate_id = row.get("candidate_id")
        if candidate_id is None:
            continue
        output.setdefault(str(candidate_id), []).append(row)
    return output


def main() -> None:
    case = _load(CASE_PATH)
    gt = _load(GT_PATH)
    baseline = _load(GEOMETRY_BASELINE)
    cell_filter = _load(CELL_FILTER_PATH)
    datum_diag = _load(DATUM_DIAGNOSTIC_PATH)

    _assert_artifact(cell_filter, phase="phase5_cell_content_filter_diagnostic")
    _assert_artifact(datum_diag, phase="phase5_datum_glyph_diagnostic")

    pdf_path = PROJECT_ROOT / case["pdf"]
    page_index = int(case.get("page_index", 0))
    pdf_bytes = pdf_path.read_bytes()

    benchmark_ids = [
        str(row["candidate_id"])
        for row in baseline.get("matches", [])
        if row.get("candidate_id")
    ]
    characteristic_by_candidate = _characteristics_by_candidate(gt, baseline)
    filter_by_key = _filter_rows_by_key(cell_filter)
    datum_by_candidate = _datum_rows_by_candidate(datum_diag)

    detector = GdtFrameDetector()
    detected = detector.detect_frames(pdf_bytes, page_index=page_index)
    candidate_by_id = {candidate.candidate_id: candidate for candidate in detected}

    output_rows: list[dict] = []
    integration_errors: list[str] = []

    for candidate_id in benchmark_ids:
        candidate = candidate_by_id.get(candidate_id)
        if candidate is None:
            integration_errors.append(f"{candidate_id}: detector candidate missing")
            continue

        characteristic = characteristic_by_candidate.get(candidate_id)
        tolerance_key = f"{candidate_id}:cell[1]"
        tolerance_row = filter_by_key.get(tolerance_key)
        tolerance_assessment = (
            assessment_from_filter_row(tolerance_row)
            if tolerance_row is not None
            else None
        )

        datum_cells: dict[int, str] = {}
        datum_evidence_rows: list[dict] = []
        for row in datum_by_candidate.get(candidate_id, []):
            predicted = row.get("predicted_label")
            status = row.get("status")
            cell_index = row.get("cell_index")
            if status != "ok" or not predicted or cell_index is None:
                continue
            datum_cells[int(cell_index)] = str(predicted).upper()
            datum_evidence_rows.append(
                {
                    "cell_index": int(cell_index),
                    "predicted_label": str(predicted).upper(),
                    "score": (row.get("ranking") or [{}])[0].get("score") if row.get("ranking") else None,
                    "margin": row.get("margin"),
                    "evaluation_role": row.get("evaluation_role"),
                    "acceptance_threshold_calibrated": bool(
                        datum_diag.get("acceptance_threshold_calibrated", False)
                    ),
                }
            )

        notes = [
            "characteristic supplied from independent ground truth to isolate Phase 5 integration",
        ]
        if datum_cells and not datum_diag.get("acceptance_threshold_calibrated", False):
            notes.append(
                "datum glyph ranking used diagnostically; global acceptance threshold is not calibrated"
            )
        if tolerance_assessment is None:
            notes.append("tolerance filter evidence missing for cell[1]")

        evidence = FrameVisualEvidence(
            datum_by_cell=datum_cells,
            tolerance_assessment=tolerance_assessment,
            source="phase5_datum_glyph_diagnostic",
            notes=notes,
        )

        try:
            parsed = parse_feature_control_frame(
                candidate,
                characteristic=characteristic,
                visual_evidence=evidence,
            )
        except Exception as exc:  # diagnostic must surface, never hide, integration failure
            integration_errors.append(f"{candidate_id}: {type(exc).__name__}: {exc}")
            continue

        output_rows.append(
            {
                "candidate_id": candidate_id,
                "characteristic_input_source": "independent_ground_truth_isolation",
                "tolerance_assessment": (
                    tolerance_assessment.to_dict() if tolerance_assessment is not None else None
                ),
                "datum_evidence": datum_evidence_rows,
                "visual_evidence": evidence.to_dict(),
                "parsed": parsed.to_dict(),
            }
        )

    structured_count = len(output_rows)
    benchmark_count = len(benchmark_ids)
    tolerance_resolved_count = sum(
        row["parsed"].get("tolerance_value") is not None for row in output_rows
    )
    datum_frame_count = sum(bool(row["parsed"].get("referenced_datums")) for row in output_rows)
    datum_reference_count = sum(
        len(row["parsed"].get("referenced_datums", [])) for row in output_rows
    )
    unresolved_frame_count = sum(
        bool(row["parsed"].get("unresolved_fields")) for row in output_rows
    )

    integration_gate_passed = structured_count == benchmark_count and not integration_errors
    content_complete = integration_gate_passed and unresolved_frame_count == 0

    payload = {
        "schema_version": 1,
        "phase": "phase5_frame_integration",
        "case_id": CASE_ID,
        "validation_status": "CASE41_DIAGNOSTIC_ONLY",
        "ocr_used": False,
        "llm_used": False,
        "characteristic_input_source": "independent_ground_truth_isolation",
        "end_to_end_symbol_validation": False,
        "datum_acceptance_threshold_calibrated": bool(
            datum_diag.get("acceptance_threshold_calibrated", False)
        ),
        "integration_gate": "PASS" if integration_gate_passed else "FAIL",
        "content_complete": content_complete,
        "benchmark_frame_count": benchmark_count,
        "structured_frame_count": structured_count,
        "tolerance_resolved_count": tolerance_resolved_count,
        "tolerance_unresolved_count": structured_count - tolerance_resolved_count,
        "frames_with_datum_references": datum_frame_count,
        "datum_reference_count": datum_reference_count,
        "unresolved_frame_count": unresolved_frame_count,
        "integration_errors": integration_errors,
        "frames": output_rows,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("phase=phase5_frame_integration")
    print("validation_status=CASE41_DIAGNOSTIC_ONLY")
    print("ocr_used=False")
    print("llm_used=False")
    print("characteristic_input_source=independent_ground_truth_isolation")
    print("end_to_end_symbol_validation=False")
    print(f"integration_gate={'PASS' if integration_gate_passed else 'FAIL'}")
    print(f"content_complete={content_complete}")
    print(f"structured_frames={structured_count}/{benchmark_count}")
    print(f"tolerance_resolved={tolerance_resolved_count}/{structured_count}")
    print(f"datum_references={datum_reference_count} across_frames={datum_frame_count}")
    print(f"unresolved_frames={unresolved_frame_count}")

    if integration_errors:
        print("\nintegration_errors:")
        for error in integration_errors:
            print(f"  {error}")

    print("\nbenchmark_real_frame_structured:")
    for row in output_rows:
        parsed = row["parsed"]
        tolerance = parsed.get("tolerance_raw")
        tolerance_text = tolerance if tolerance is not None else "UNRESOLVED"
        datums = parsed.get("referenced_datums") or []
        unresolved = parsed.get("unresolved_fields") or []
        print(
            f"  {row['candidate_id']} characteristic={parsed.get('characteristic')} "
            f"tolerance={tolerance_text} datums={datums} unresolved={unresolved}"
        )
        assessment = row.get("tolerance_assessment")
        if assessment is not None:
            print(
                f"    tolerance_status={assessment.get('status')} "
                f"selected_text={assessment.get('selected_text_candidate_count')} "
                f"structural={assessment.get('structural_count')} "
                f"arrows={assessment.get('arrow_like_count')}"
            )
        for datum in row.get("datum_evidence", []):
            print(
                f"    datum cell[{datum['cell_index']}]={datum['predicted_label']} "
                f"margin={datum.get('margin')} role={datum.get('evaluation_role')}"
            )

    print(f"\noutput={OUTPUT_PATH}")


if __name__ == "__main__":
    main()
