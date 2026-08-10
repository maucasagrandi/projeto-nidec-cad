from src.cad_review.compliance_engine import build_cad_review_result


def test_phase8_aggregates_standards_iso1101_and_iso5459_findings():
    part_branch = {
        "classification": {
            "component": {"value": "GASKET - VALVE PLATE", "confidence": 0.99},
            "compressor_series": {"value": None, "confidence": 0.0},
        },
        "cited_standards": [
            {"standard": "TSS 002513", "standard_raw": "TSS 002513", "source_text": "note 1"},
            {"standard": "NTB 2611", "standard_raw": "NTB2611", "source_text": "note x"},
        ],
        "applicable_standards": [
            {"standard": "TSS 002513", "reason": "All", "source": "component_match"},
            {"standard": "TSS 002420", "reason": "All", "source": "component_match"},
        ],
        "standards_comparison": {
            "expected": ["TSS 002420", "TSS 002513"],
            "cited": ["NTB 2611", "TSS 002513"],
            "matching": ["TSS 002513"],
            "missing": ["TSS 002420"],
            "unexpected": ["NTB 2611"],
            "applicability_status": "RESOLVED",
        },
        "provenance": {"classification_source": "existing_llm_part_classification"},
    }
    phase5 = {
        "phase": "phase5_frame_integration",
        "validation_status": "CASE41_DIAGNOSTIC_ONLY",
        "frames": [
            {
                "candidate_id": "GDT-CAND-P01-004",
                "parsed": {
                    "characteristic": "position",
                    "tolerance_raw": None,
                    "tolerance_value": None,
                    "referenced_datums": ["A", "D"],
                    "unresolved_fields": ["tolerance_value"],
                    "field_sources": {},
                },
            }
        ],
    }
    phase6 = {
        "phase": "phase6_iso1101_violation_diagnostic",
        "validation_status": "CASE41_DIAGNOSTIC_ONLY",
        "mode": "reference",
        "normative_applicability_established": False,
        "frames": [
            {
                "candidate_id": "GDT-CAND-P01-004",
                "finding": {
                    "status": "NEEDS_CONTEXT",
                    "code": "ISO1101_DATUM_REQUIREMENT_CONDITIONAL",
                    "standard_label": "ISO 1101:2017",
                    "source_ref": "ISO 1101:2017 Table 1, subclause 18.12",
                    "characteristic": "position",
                    "referenced_datums": ["A", "D"],
                    "datum_requirement": "conditional",
                    "finding": "Position datum requirement is context-dependent.",
                    "recommended_action": "Review feature context.",
                    "normative_claim": False,
                },
            }
        ],
    }
    phase7 = {
        "phase": "phase7_datum_definition_diagnostic",
        "validation_status": "CASE41_DIAGNOSTIC_ONLY",
        "mode": "reference",
        "normative_applicability_established": False,
        "standard": "ISO 5459",
        "source_ref": "Datum Feature Indicator -> ISO 5459",
        "definitions": [
            {"label": "A", "page": 1},
            {"label": "B", "page": 1},
        ],
        "frames": [
            {
                "candidate_id": "GDT-CAND-P01-004",
                "findings": [
                    {
                        "status": "PASS",
                        "code": "ISO5459_DATUM_DEFINITION_FOUND",
                        "datum": "A",
                        "standard": "ISO 5459",
                        "finding": "Datum A is referenced and a corresponding datum feature indicator was identified in the drawing.",
                        "recommended_action": "No action required for datum-definition presence.",
                        "normative_claim": False,
                        "definition_count": 1,
                        "definition_evidence": [{"label": "A", "page": 1}],
                    },
                    {
                        "status": "WARNING",
                        "code": "ISO5459_REFERENCED_DATUM_NOT_DEFINED",
                        "datum": "D",
                        "standard": "ISO 5459",
                        "finding": "Potential violation of ISO 5459: datum D is referenced but not defined.",
                        "recommended_action": "Verify that datum D is correctly defined in the drawing or correct the GD&T datum reference.",
                        "normative_claim": False,
                        "definition_count": 0,
                        "definition_evidence": [],
                    },
                ],
            }
        ],
    }

    result = build_cad_review_result(
        drawing={"name": "case41.pdf"},
        part_branch=part_branch,
        phase5_payload=phase5,
        phase6_payload=phase6,
        phase7_payload=phase7,
    )

    assert result.review_context.compressor_series == "ALL"
    assert result.review_context.compressor_series_source == "temporary_default_until_windchill"
    assert result.part_classification["compressor_series"]["value"] is None
    assert result.summary.PASS == 2
    assert result.summary.WARNING == 2
    assert result.summary.NEEDS_CONTEXT == 2
    assert result.summary.NOT_EVALUATED == 0

    by_code = {row.code: row for row in result.findings}
    assert by_code["APPLICABLE_STANDARD_MISSING"].standard == "TSS 002420"
    assert by_code["ISO1101_DATUM_REQUIREMENT_CONDITIONAL"].candidate_id == "GDT-CAND-P01-004"
    assert by_code["ISO5459_REFERENCED_DATUM_NOT_DEFINED"].datum == "D"
    assert by_code["ISO5459_REFERENCED_DATUM_NOT_DEFINED"].standard == "ISO 5459"
    assert by_code["ISO5459_DATUM_DEFINITION_FOUND"].evidence["definition_count"] == 1
