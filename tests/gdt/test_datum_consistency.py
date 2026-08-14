from src.gdt.datum_consistency import (
    MODE_NORMATIVE,
    STATUS_PASS,
    STATUS_WARNING,
    assess_referenced_datum_definitions,
)


def _indicator(label: str) -> dict:
    return {
        "label": label,
        "page": 1,
        "text_bbox": [1, 1, 2, 2],
        "box_bbox": [0, 0, 3, 3],
        "marker_bbox": [1, 4, 2, 5],
        "marker_side": "bottom",
        "stem_coverage": 1.0,
    }


def test_referenced_datum_with_definition_passes():
    findings = assess_referenced_datum_definitions(
        referenced_datums=["A"],
        defined_indicators=[_indicator("A")],
    )
    assert len(findings) == 1
    assert findings[0].status == STATUS_PASS
    assert findings[0].code == "ISO5459_DATUM_DEFINITION_FOUND"


def test_missing_referenced_datum_generates_reference_warning():
    findings = assess_referenced_datum_definitions(
        referenced_datums=["A", "D"],
        defined_indicators=[_indicator("A")],
    )
    by_label = {row.datum: row for row in findings}
    assert by_label["A"].status == STATUS_PASS
    assert by_label["D"].status == STATUS_WARNING
    assert by_label["D"].code == "ISO5459_REFERENCED_DATUM_NOT_DEFINED"
    assert by_label["D"].finding.startswith("Potential violation of ISO 5459")
    assert by_label["D"].normative_claim is False


def test_normative_mode_changes_only_strength_of_claim():
    finding = assess_referenced_datum_definitions(
        referenced_datums=["D"],
        defined_indicators=[],
        mode=MODE_NORMATIVE,
    )[0]
    assert finding.status == STATUS_WARNING
    assert finding.finding.startswith("Violation of ISO 5459")
    assert finding.normative_claim is True


def test_duplicate_references_generate_one_aggregate_finding():
    findings = assess_referenced_datum_definitions(
        referenced_datums=["A", "A", "B"],
        defined_indicators=[_indicator("A"), _indicator("B")],
    )
    assert [row.datum for row in findings] == ["A", "B"]
