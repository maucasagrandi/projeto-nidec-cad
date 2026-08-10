"""Phase 6 diagnostic: resolve ISO 1101 edition before datum semantics.

This script consumes the structured Phase 5 case-41 output.  It does not guess
an ISO edition and it does not embed normative datum rules.

Inputs are explicit CLI/context values:
- --citation can be repeated and should contain the actual drawing/customer
  citation (for example "ISO 1101" or "ISO 1101:2017");
- --compressor-series supplies external PLM/PDM/frontend context when needed;
- --applicability-json optionally supplies customer applicability rows;
- --semantic-rules-json optionally supplies edition-specific characteristic
  rules, each with source_ref provenance.

Without a resolved edition and an explicit semantic rule, the frame is reported
as NOT_EVALUATED rather than passed/failed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gdt.iso1101 import assess_datum_reference_semantics, resolve_iso1101_edition

CASE_ID = "case_41_rev8"
PHASE5_PATH = (
    PROJECT_ROOT
    / "validation"
    / "gdt"
    / "outputs"
    / "phase5"
    / CASE_ID
    / "frame_integration"
    / "phase5_structured_frames.json"
)
OUTPUT_DIR = PROJECT_ROOT / "validation" / "gdt" / "outputs" / "phase6" / CASE_ID
OUTPUT_PATH = OUTPUT_DIR / "iso1101_semantics_diagnostic.json"


def _load(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows_from_json(path: Path | None) -> list[dict]:
    if path is None:
        return []
    payload = _load(path)
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("rows", "rules", "applicability", "semantic_rules"):
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
    raise ValueError(f"could not find rule rows in {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--citation",
        action="append",
        default=[],
        help="Actual ISO citation text. Repeat for multiple citations.",
    )
    parser.add_argument("--compressor-series", default=None)
    parser.add_argument("--applicability-json", type=Path, default=None)
    parser.add_argument("--semantic-rules-json", type=Path, default=None)
    args = parser.parse_args()

    phase5 = _load(PHASE5_PATH)
    if phase5.get("phase") != "phase5_frame_integration":
        raise ValueError("required Phase 5 structured-frame artifact has wrong phase")

    applicability = _rows_from_json(args.applicability_json)
    semantic_rules = _rows_from_json(args.semantic_rules_json)
    resolution = resolve_iso1101_edition(
        args.citation,
        applicability_rules=applicability,
        compressor_series=args.compressor_series,
    )

    rows = []
    evaluated_count = 0
    needs_context_count = 0
    not_evaluated_count = 0

    for frame_row in phase5.get("frames", []):
        parsed = frame_row.get("parsed") or {}
        assessment = assess_datum_reference_semantics(
            characteristic=parsed.get("characteristic"),
            referenced_datums=parsed.get("referenced_datums") or [],
            edition_resolution=resolution,
            rules=semantic_rules,
        )
        if assessment.status == "evaluated":
            evaluated_count += 1
        elif assessment.status == "needs_condition_context":
            needs_context_count += 1
        else:
            not_evaluated_count += 1
        rows.append(
            {
                "candidate_id": frame_row.get("candidate_id"),
                "phase5_unresolved_fields": parsed.get("unresolved_fields") or [],
                "assessment": assessment.to_dict(),
            }
        )

    payload = {
        "schema_version": 1,
        "phase": "phase6_iso1101_diagnostic",
        "case_id": CASE_ID,
        "validation_status": "DIAGNOSTIC_ONLY",
        "phase5_input": str(PHASE5_PATH.relative_to(PROJECT_ROOT)),
        "citations": list(args.citation),
        "compressor_series": args.compressor_series,
        "applicability_source": str(args.applicability_json) if args.applicability_json else None,
        "semantic_rules_source": str(args.semantic_rules_json) if args.semantic_rules_json else None,
        "edition_resolution": resolution.to_dict(),
        "evaluated_frame_count": evaluated_count,
        "needs_condition_context_count": needs_context_count,
        "not_evaluated_frame_count": not_evaluated_count,
        "frames": rows,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("phase=phase6_iso1101_diagnostic")
    print("validation_status=DIAGNOSTIC_ONLY")
    print(
        f"edition_status={resolution.status} edition={resolution.edition or '-'} "
        f"source={resolution.source or '-'}"
    )
    print(f"edition_reason={resolution.reason}")
    print(
        f"frames evaluated={evaluated_count} needs_context={needs_context_count} "
        f"not_evaluated={not_evaluated_count}"
    )
    print("\nframe_semantics:")
    for row in rows:
        assessment = row["assessment"]
        print(
            f"  {row['candidate_id']} characteristic={assessment['characteristic']} "
            f"datums={assessment['referenced_datums']} status={assessment['status']} "
            f"result={assessment['result']} requirement={assessment['datum_requirement'] or '-'}"
        )
    print(f"\noutput={OUTPUT_PATH}")


if __name__ == "__main__":
    main()
