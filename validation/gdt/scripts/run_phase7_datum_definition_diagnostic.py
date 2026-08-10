"""Phase 7 diagnostic: verify that referenced datums are defined in the drawing.

Consumes the structured Phase 5 output and compares referenced datum letters
against deterministic Datum Feature Indicator detections on the original PDF.

No OCR and no LLM are used.  The ISO 5459 reference is currently a reference
baseline only; this script does not claim that a specific ISO 5459 edition is
contractually applicable to the drawing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import fitz
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gdt.datum_consistency import assess_referenced_datum_definitions
from src.gdt.datum_feature import detect_datum_feature_indicators

CASE_ID = "case_41_rev8"
CASE_PATH = PROJECT_ROOT / "validation" / "gdt" / "cases" / f"{CASE_ID}.json"
DEFAULT_PHASE5_PATH = (
    PROJECT_ROOT
    / "validation"
    / "gdt"
    / "outputs"
    / "phase5"
    / CASE_ID
    / "frame_integration"
    / "phase5_structured_frames.json"
)
OUTPUT_DIR = PROJECT_ROOT / "validation" / "gdt" / "outputs" / "phase7" / CASE_ID
OUTPUT_PATH = OUTPUT_DIR / "datum_definition_diagnostic.json"
OVERLAY_PATH = OUTPUT_DIR / "datum_feature_indicators.png"
SOURCE_REF = "Datum related symbols table: Datum Feature Indicator -> ISO 5459"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _render_overlay(pdf_bytes: bytes, page_index: int, indicators) -> None:
    dpi = 200
    scale = dpi / 72.0
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        page = doc[page_index]
        pix = page.get_pixmap(
            matrix=fitz.Matrix(scale, scale),
            colorspace=fitz.csRGB,
            alpha=False,
        )
        image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3).copy()
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        for indicator in indicators:
            x0, y0, x1, y1 = indicator.box_bbox
            mx0, my0, mx1, my1 = indicator.marker_bbox
            cv2.rectangle(
                image,
                (int(round(x0 * scale)), int(round(y0 * scale))),
                (int(round(x1 * scale)), int(round(y1 * scale))),
                (0, 0, 255),
                4,
            )
            cv2.rectangle(
                image,
                (int(round(mx0 * scale)), int(round(my0 * scale))),
                (int(round(mx1 * scale)), int(round(my1 * scale))),
                (255, 0, 0),
                3,
            )
            cv2.putText(
                image,
                f"datum {indicator.label}",
                (int(round(x0 * scale)), max(20, int(round((y0 - 5) * scale)))),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        cv2.imwrite(str(OVERLAY_PATH), image)
    finally:
        doc.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase5-json", type=Path, default=DEFAULT_PHASE5_PATH)
    args = parser.parse_args()

    case = _load(CASE_PATH)
    phase5 = _load(args.phase5_json)
    if phase5.get("phase") != "phase5_frame_integration":
        raise ValueError("Phase 7 requires phase5_frame_integration structured output")

    pdf_path = PROJECT_ROOT / case["pdf"]
    pdf_bytes = pdf_path.read_bytes()
    page_index = int(case.get("page_index", 0))

    indicators = detect_datum_feature_indicators(pdf_bytes, page_index=page_index)
    defined_labels = sorted({row.label for row in indicators})

    frame_rows = []
    referenced_all: list[str] = []
    for frame in phase5.get("frames", []):
        parsed = frame.get("parsed") or {}
        refs = [str(value).strip().upper() for value in parsed.get("referenced_datums", []) if str(value).strip()]
        referenced_all.extend(refs)
        frame_findings = assess_referenced_datum_definitions(
            referenced_datums=refs,
            defined_indicators=indicators,
            mode="reference",
            standard="ISO 5459",
            source_ref=SOURCE_REF,
        )
        frame_rows.append(
            {
                "candidate_id": frame.get("candidate_id"),
                "referenced_datums": refs,
                "findings": [row.to_dict() for row in frame_findings],
            }
        )

    aggregate = assess_referenced_datum_definitions(
        referenced_datums=referenced_all,
        defined_indicators=indicators,
        mode="reference",
        standard="ISO 5459",
        source_ref=SOURCE_REF,
    )
    referenced_labels = [row.datum for row in aggregate]
    missing_labels = [row.datum for row in aggregate if row.status == "WARNING"]

    occurrence_pass = 0
    occurrence_warning = 0
    for frame in frame_rows:
        for finding in frame["findings"]:
            if finding["status"] == "PASS":
                occurrence_pass += 1
            elif finding["status"] == "WARNING":
                occurrence_warning += 1

    payload = {
        "schema_version": 1,
        "phase": "phase7_datum_definition_diagnostic",
        "case_id": CASE_ID,
        "validation_status": "CASE41_DIAGNOSTIC_ONLY",
        "ocr_used": False,
        "llm_used": False,
        "standard": "ISO 5459",
        "iso5459_edition_resolved": False,
        "mode": "reference",
        "normative_applicability_established": False,
        "source_ref": SOURCE_REF,
        "phase5_input": str(args.phase5_json),
        "detector_method": "single uppercase PDF token + enclosing small box + filled triangular marker + connected stem",
        "referenced_labels": referenced_labels,
        "defined_labels": defined_labels,
        "missing_definition_labels": missing_labels,
        "definition_count": len(indicators),
        "aggregate_counts": {
            "PASS": sum(row.status == "PASS" for row in aggregate),
            "WARNING": sum(row.status == "WARNING" for row in aggregate),
        },
        "reference_occurrence_counts": {
            "PASS": occurrence_pass,
            "WARNING": occurrence_warning,
        },
        "definitions": [row.to_dict() for row in indicators],
        "aggregate_findings": [row.to_dict() for row in aggregate],
        "frames": frame_rows,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _render_overlay(pdf_bytes, page_index, indicators)

    print("phase=phase7_datum_definition_diagnostic")
    print("validation_status=CASE41_DIAGNOSTIC_ONLY")
    print("ocr_used=False")
    print("llm_used=False")
    print("standard=ISO 5459")
    print("mode=reference")
    print(f"referenced_labels={referenced_labels}")
    print(f"defined_labels={defined_labels}")
    print(f"missing_definition_labels={missing_labels}")
    print(f"definition_count={len(indicators)}")
    print(f"aggregate_findings pass={payload['aggregate_counts']['PASS']} warning={payload['aggregate_counts']['WARNING']}")
    print(f"reference_occurrences pass={occurrence_pass} warning={occurrence_warning}")
    print("\ndatum_definitions:")
    for row in indicators:
        print(
            f"  {row.label}: box={tuple(round(v, 2) for v in row.box_bbox)} "
            f"marker={row.marker_side} stem_coverage={row.stem_coverage:.3f}"
        )
    print("\naggregate_findings:")
    for row in aggregate:
        print(f"  datum={row.datum} status={row.status} code={row.code}")
        print(f"    finding={row.finding}")
        print(f"    recommended_action={row.recommended_action}")
    print(f"\noutput={OUTPUT_PATH}")
    print(f"overlay={OVERLAY_PATH}")


if __name__ == "__main__":
    main()
