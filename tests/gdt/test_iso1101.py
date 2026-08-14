from src.gdt.iso1101 import (
    SEMANTIC_EVALUATED,
    SEMANTIC_NEEDS_CONTEXT,
    SEMANTIC_NOT_EVALUATED,
    STATUS_AMBIGUOUS,
    STATUS_MISSING_CONTEXT,
    STATUS_NO_CITATION,
    STATUS_RESOLVED,
    assess_datum_reference_semantics,
    parse_iso1101_citation,
    resolve_iso1101_edition,
)


def test_parses_explicit_iso1101_year():
    citation = parse_iso1101_citation("Drawing according to ISO 1101:2017")
    assert citation is not None
    assert citation.year == 2017


def test_explicit_supported_year_resolves_without_applicability():
    result = resolve_iso1101_edition(["ISO 1101:2012"])
    assert result.status == STATUS_RESOLVED
    assert result.edition == 2012
    assert result.source == "drawing_citation"


def test_missing_iso1101_citation_stays_unresolved():
    result = resolve_iso1101_edition(["ISO 5459:2011"])
    assert result.status == STATUS_NO_CITATION
    assert result.edition is None


def test_yearless_citation_resolves_from_exact_series_rule():
    result = resolve_iso1101_edition(
        ["ISO 1101"],
        compressor_series="VCC",
        applicability_rules=[
            {"Standard": "ISO 1101", "edition": 2012, "Compressor Series": "FFU", "Applicability": "Applicable"},
            {"Standard": "ISO 1101", "edition": 2017, "Compressor Series": "VCC", "Applicability": "Applicable"},
        ],
    )
    assert result.status == STATUS_RESOLVED
    assert result.edition == 2017
    assert result.source == "customer_applicability_exact_series"


def test_yearless_series_specific_rule_requires_series_context():
    result = resolve_iso1101_edition(
        ["ISO 1101"],
        applicability_rules=[
            {"Standard": "ISO 1101", "edition": 2017, "Compressor Series": "VCC", "Applicability": "Applicable"},
        ],
    )
    assert result.status == STATUS_MISSING_CONTEXT
    assert result.edition is None


def test_conflicting_applicable_editions_stay_ambiguous():
    result = resolve_iso1101_edition(
        ["ISO 1101"],
        applicability_rules=[
            {"Standard": "ISO 1101", "edition": 2012, "Compressor Series": "*", "Applicability": True},
            {"Standard": "ISO 1101", "edition": 2017, "Compressor Series": "*", "Applicability": True},
        ],
    )
    assert result.status == STATUS_AMBIGUOUS
    assert result.edition is None


def test_semantics_not_evaluated_until_edition_is_resolved():
    edition = resolve_iso1101_edition(["ISO 1101"])
    result = assess_datum_reference_semantics(
        characteristic="position",
        referenced_datums=["A"],
        edition_resolution=edition,
        rules=[],
    )
    assert result.status == SEMANTIC_NOT_EVALUATED
    assert result.result == "edition_unresolved"


def test_required_datum_rule_reports_missing_reference():
    edition = resolve_iso1101_edition(["ISO 1101:2017"])
    result = assess_datum_reference_semantics(
        characteristic="position",
        referenced_datums=[],
        edition_resolution=edition,
        rules=[
            {
                "characteristic": "position",
                "edition": 2017,
                "datum_requirement": "required",
                "source_ref": "test fixture only",
            }
        ],
    )
    assert result.status == SEMANTIC_EVALUATED
    assert result.result == "missing_required_datum_reference"


def test_conditional_rule_does_not_force_pass_or_fail():
    edition = resolve_iso1101_edition(["ISO 1101:2017"])
    result = assess_datum_reference_semantics(
        characteristic="profile",
        referenced_datums=[],
        edition_resolution=edition,
        rules=[
            {
                "characteristic": "profile",
                "edition": 2017,
                "datum_requirement": "conditional",
                "source_ref": "test fixture only",
            }
        ],
    )
    assert result.status == SEMANTIC_NEEDS_CONTEXT
    assert result.result == "condition_context_required"
