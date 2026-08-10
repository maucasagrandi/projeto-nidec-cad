"""Phase 6 diagnostic: user-facing ISO 1101 GD&T rule findings.

Default behavior matches the current CAD Review requirement:
- ISO 1101:2017 is used as an explicit *reference baseline*;
- the rule table comes from the reviewed ISO 1101:2017 Table 1 excerpt;
- findings are PASS, WARNING, NEEDS_CONTEXT or NOT_EVALUATED;
- WARNING text cites the ISO edition and configured table/subclause;
- reference mode says "Potential violation" because the drawing/TSS chain has
  not yet established contractual applicability of ISO 1101:2017.

This diagnostic consumes Phase 5 structured frames. It does not validate that a
referenced datum is actually defined elsewhere in the drawing; that belongs to
Phase 7 / ISO 5459.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gdt.iso1101_reference import (
    FINDING_NEEDS_CONTEXT,
    FINDING_NOT_EVALUATED,
    FINDING_PASS,
    FINDING_WARNING,
    MODE_NORMATIVE,
    MODE_REFERENCE,
    assess_iso1101_datum_rule,
)

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
DEFAULT_RULES_PATH = PROJECT_ROOT / "validation" / "gdt" / "configs" / "iso1101_2017_reference_rules.json"
OUTPUT_DIR = PROJECT_ROOT / "validation" / "gdt" / "outputs" / "phase6" / CASE_ID
OUTPUT_PATH = OUTPUT_DIR / "iso1101_violation_diagnostic.json"


def _load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=[MODE_REFERENCE, MODE_NORMATIVE], default=MODE_REFERENCE)
    parser.add_argument("--rules-json", type=Path, default=DEFAULT_RULES_PATH)
    parser.add_argument("--edition", type=int, default=2017)
    args = parser.parse_args()

    phase5 = _load(PHASE5_PATH)
    if phase5.get("phase") != "phase5_frame_integration":
        raise ValueError("required Phase 5 artifact has wrong phase")

    rule_payload = _load(args.rules_json)
    rules = rule_payload.get("rules")
    if not isinstance(rules, list):
        raise ValueError("rules JSON must contain a 'rules' list")
    configured_edition = int(rule_payload.get("edition", args.edition))
    if configured_edition != int(args.edition):
        raise ValueError(
            f"requested edition {args.edition} does not match rule configuration edition {configured_edition}"
        )

    rows = []
    counts = {
        FINDING_PASS: 0,
        FINDING_WARNING: 0,
        FINDING_NEEDS_CONTEXT: 0,
        FINDING_NOT_EVALUATED: 0,
    }

    for frame in phase5.get("frames", []):
        parsed = frame.get("parsed") or {}
        finding = assess_iso1101_datum_rule(
            characteristic=parsed.get("characteristic"),
            referenced_datums=parsed.get("referenced_datums") or [],
            rules=rules,
            edition=args.edition,
            mode=args.mode,
        )
        counts[finding.status] = counts.get(finding.status, 0) + 1
        rows.append(
            {
                "candidate_id": frame.get("candidate_id"),
                "phase5_unresolved_fields": parsed.get("unresolved_fields") or [],
                "finding": finding.to_dict(),
            }
        )

    payload = {
        "schema_version": 1,
        "phase": "phase6_iso1101_violation_diagnostic",
        "case_id": CASE_ID,
        "validation_status": "CASE41_DIAGNOSTIC_ONLY",
        "phase5_input": str(PHASE5_PATH.relative_to(PROJECT_ROOT)),
        "standard": "ISO 1101",
        "edition": args.edition,
        "mode": args.mode,
        "normative_applicability_established": args.mode == MODE_NORMATIVE,
        "rules_source": str(args.rules_json.relative_to(PROJECT_ROOT)) if args.rules_json.is_relative_to(PROJECT_ROOT) else str(args.rules_json),
        "rules_scope_note": rule_payload.get("scope_note"),
        "counts": counts,
        "frames": rows,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("phase=phase6_iso1101_violation_diagnostic")
    print("validation_status=CASE41_DIAGNOSTIC_ONLY")
    print(f"standard=ISO 1101:{args.edition}")
    print(f"mode={args.mode}")
    print(f"normative_applicability_established={args.mode == MODE_NORMATIVE}")
    print(
        f"findings pass={counts[FINDING_PASS]} warning={counts[FINDING_WARNING]} "
        f"needs_context={counts[FINDING_NEEDS_CONTEXT]} not_evaluated={counts[FINDING_NOT_EVALUATED]}"
    )
    print("\niso_findings:")
    for row in rows:
        finding = row["finding"]
        print(
            f"  {row['candidate_id']} characteristic={finding['characteristic']} "
            f"datums={finding['referenced_datums']} status={finding['status']} "
            f"code={finding['code']}"
        )
        print(f"    source={finding['source_ref'] or '-'}")
        print(f"    finding={finding['finding']}")
        if finding.get("recommended_action"):
            print(f"    recommended_action={finding['recommended_action']}")

    print(f"\noutput={OUTPUT_PATH}")


if __name__ == "__main__":
    main()
