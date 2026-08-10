import json
from pathlib import Path

from src.gdt.iso1101_reference import (
    FINDING_NEEDS_CONTEXT,
    FINDING_PASS,
    FINDING_WARNING,
    MODE_NORMATIVE,
    MODE_REFERENCE,
    assess_iso1101_datum_rule,
)


RULE_PATH = Path("validation/gdt/configs/iso1101_2017_reference_rules.json")


def _rules():
    return json.loads(RULE_PATH.read_text(encoding="utf-8"))["rules"]


def test_perpendicularity_without_datum_generates_iso_warning():
    finding = assess_iso1101_datum_rule(
        characteristic="perpendicularity",
        referenced_datums=[],
        rules=_rules(),
        edition=2017,
        mode=MODE_REFERENCE,
    )
    assert finding.status == FINDING_WARNING
    assert finding.code == "ISO1101_REQUIRED_DATUM_MISSING"
    assert finding.standard_label == "ISO 1101:2017"
    assert finding.source_ref == "ISO 1101:2017 Table 1, subclause 18.10"
    assert "Potential violation of ISO 1101:2017" in finding.finding
    assert finding.normative_claim is False


def test_perpendicularity_with_datum_passes_reference_rule():
    finding = assess_iso1101_datum_rule(
        characteristic="perpendicularity",
        referenced_datums=["A"],
        rules=_rules(),
    )
    assert finding.status == FINDING_PASS
    assert finding.code == "ISO1101_DATUM_RULE_OK"


def test_flatness_without_datum_passes():
    finding = assess_iso1101_datum_rule(
        characteristic="flatness",
        referenced_datums=[],
        rules=_rules(),
    )
    assert finding.status == FINDING_PASS


def test_flatness_with_datum_generates_warning():
    finding = assess_iso1101_datum_rule(
        characteristic="flatness",
        referenced_datums=["A"],
        rules=_rules(),
    )
    assert finding.status == FINDING_WARNING
    assert finding.code == "ISO1101_DATUM_NOT_USED_BY_CHARACTERISTIC"
    assert finding.source_ref == "ISO 1101:2017 Table 1, subclause 18.2"


def test_position_does_not_generate_false_violation_from_datum_presence_alone():
    no_datum = assess_iso1101_datum_rule(
        characteristic="position",
        referenced_datums=[],
        rules=_rules(),
    )
    with_datum = assess_iso1101_datum_rule(
        characteristic="position",
        referenced_datums=["A", "B"],
        rules=_rules(),
    )
    assert no_datum.status == FINDING_NEEDS_CONTEXT
    assert with_datum.status == FINDING_NEEDS_CONTEXT
    assert no_datum.code == "ISO1101_DATUM_REQUIREMENT_CONDITIONAL"


def test_profile_is_conditional_because_table_has_form_and_datum_related_entries():
    finding = assess_iso1101_datum_rule(
        characteristic="profile",
        referenced_datums=[],
        rules=_rules(),
    )
    assert finding.status == FINDING_NEEDS_CONTEXT
    assert "context-dependent" in finding.finding


def test_normative_mode_uses_violation_wording_only_when_caller_selects_it():
    finding = assess_iso1101_datum_rule(
        characteristic="parallelism",
        referenced_datums=[],
        rules=_rules(),
        mode=MODE_NORMATIVE,
    )
    assert finding.status == FINDING_WARNING
    assert finding.finding.startswith("Violation of ISO 1101:2017")
    assert finding.normative_claim is True


def test_reference_mode_never_makes_normative_claim():
    finding = assess_iso1101_datum_rule(
        characteristic="parallelism",
        referenced_datums=[],
        rules=_rules(),
        mode=MODE_REFERENCE,
    )
    assert finding.status == FINDING_WARNING
    assert finding.finding.startswith("Potential violation of ISO 1101:2017")
    assert finding.normative_claim is False
