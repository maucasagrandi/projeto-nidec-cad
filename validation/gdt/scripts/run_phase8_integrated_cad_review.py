"""Phase 8 diagnostic: assemble one integrated CAD Review result for case 41.

By default this runner executes the existing LLM part-classification branch,
uses compressor_series=ALL as temporary external context, and combines it with
Phase 5/6/7 artifacts. A precomputed ``--part-branch-json`` may be supplied to
exercise only the deterministic integration without calling Vertex/Gemini.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prompts import classificacao_enriquecida_prompt
from src.cad_review.compliance_engine import build_cad_review_result
from src.cad_review.orchestrator import run_part_classification_branch

CASE_ID = "case_41_rev8"
CASE_PATH = PROJECT_ROOT / "validation" / "gdt" / "cases" / f"{CASE_ID}.json"
PHASE5_PATH = PROJECT_ROOT / "validation" / "gdt" / "outputs" / "phase5" / CASE_ID / "frame_integration" / "phase5_structured_frames.json"
PHASE6_PATH = PROJECT_ROOT / "validation" / "gdt" / "outputs" / "phase6" / CASE_ID / "iso1101_violation_diagnostic.json"
PHASE7_PATH = PROJECT_ROOT / "validation" / "gdt" / "outputs" / "phase7" / CASE_ID / "datum_definition_diagnostic.json"
OUTPUT_DIR = PROJECT_ROOT / "validation" / "gdt" / "outputs" / "phase8" / CASE_ID
OUTPUT_PATH = OUTPUT_DIR / "integrated_cad_review.json"


def _load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"required artifact not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part-branch-json", type=Path, default=None)
    parser.add_argument("--normas-xlsx", type=Path, default=PROJECT_ROOT / "Normas.xlsx")
    args = parser.parse_args()

    case = _load(CASE_PATH)
    phase5 = _load(PHASE5_PATH)
    phase6 = _load(PHASE6_PATH)
    phase7 = _load(PHASE7_PATH)

    pdf_path = PROJECT_ROOT / case["pdf"]
    pdf_bytes = pdf_path.read_bytes()

    if args.part_branch_json is not None:
        part_branch = _load(args.part_branch_json)
        part_mode = "precomputed_part_branch"
    else:
        part_branch = run_part_classification_branch(
            pdf_bytes,
            classification_prompt=classificacao_enriquecida_prompt,
            page_index=int(case.get("page_index", 0)),
            compressor_series_context="ALL",
            compressor_series_source="temporary_default_until_windchill",
            normas_path=args.normas_xlsx,
        )
        part_mode = "existing_llm_plus_applicability"

    result = build_cad_review_result(
        drawing={
            "name": pdf_path.name,
            "case_id": CASE_ID,
            "source_path": case["pdf"],
        },
        part_branch=part_branch,
        phase5_payload=phase5,
        phase6_payload=phase6,
        phase7_payload=phase7,
        compressor_series="ALL",
        compressor_series_source="temporary_default_until_windchill",
    )

    payload = result.model_dump()
    payload["validation_status"] = "CASE41_INTEGRATION_DIAGNOSTIC"
    payload["part_branch_mode"] = part_mode
    payload["production_claim"] = False
    payload["production_limitation"] = (
        "Phase 3 multi-CAD validation is still deferred; case-41 GD&T artifacts retain their recorded diagnostic limitations."
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    classification = payload.get("part_classification") or {}
    component = (classification.get("component") or {}).get("value")
    llm_series = (classification.get("compressor_series") or {}).get("value")
    context = payload["review_context"]
    summary = payload["summary"]

    print("phase=phase8_integrated_cad_review")
    print("validation_status=CASE41_INTEGRATION_DIAGNOSTIC")
    print(f"part_branch_mode={part_mode}")
    print(f"component={component or '-'}")
    print(f"llm_compressor_series={llm_series or '-'}")
    print(
        f"review_compressor_series={context['compressor_series']} "
        f"source={context['compressor_series_source']}"
    )
    print(
        "findings "
        f"pass={summary['PASS']} warning={summary['WARNING']} "
        f"needs_context={summary['NEEDS_CONTEXT']} not_evaluated={summary['NOT_EVALUATED']}"
    )
    print("\nreview_findings:")
    for finding in payload["findings"]:
        print(
            f"  {finding['finding_id']} domain={finding['domain']} status={finding['status']} "
            f"code={finding['code']} standard={finding.get('standard') or '-'}"
        )
        print(f"    finding={finding['finding']}")
        if finding.get("recommended_action"):
            print(f"    recommended_action={finding['recommended_action']}")
    print(f"\noutput={OUTPUT_PATH}")


if __name__ == "__main__":
    main()
