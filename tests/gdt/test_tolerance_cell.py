from src.gdt.tolerance_cell import (
    STATUS_INVALID_EVIDENCE,
    STATUS_RESOLVED_NUMERIC,
    STATUS_UNRESOLVED_NO_TEXT,
    STATUS_UNRESOLVED_UNRECOGNIZED,
    assessment_from_filter_row,
    resolved_numeric_assessment,
)


def test_zero_text_candidates_stays_unresolved_without_guessing():
    assessment = assessment_from_filter_row(
        {
            "cell_index": 1,
            "expected_role": "tolerance",
            "text_candidate_count": 0,
            "structural_count": 1,
            "arrow_like_count": 1,
            "other_count": 0,
        }
    )

    assert assessment.status == STATUS_UNRESOLVED_NO_TEXT
    assert assessment.resolved is False
    assert assessment.tolerance_raw is None
    assert assessment.tolerance_value is None
    assert assessment.structural_count == 1
    assert assessment.arrow_like_count == 1


def test_isolated_unrecognized_glyph_does_not_become_numeric_value():
    assessment = assessment_from_filter_row(
        {
            "cell_index": 1,
            "expected_role": "tolerance",
            "text_candidate_count": 1,
            "structural_count": 1,
            "arrow_like_count": 0,
            "other_count": 0,
        }
    )

    assert assessment.status == STATUS_UNRESOLVED_UNRECOGNIZED
    assert assessment.resolved is False
    assert assessment.tolerance_value is None


def test_non_tolerance_row_is_invalid_evidence():
    assessment = assessment_from_filter_row(
        {
            "cell_index": 2,
            "expected_role": "datum_or_modifier",
            "text_candidate_count": 1,
        }
    )

    assert assessment.status == STATUS_INVALID_EVIDENCE
    assert assessment.resolved is False


def test_resolved_numeric_evidence_is_explicit():
    assessment = resolved_numeric_assessment(raw="0,05", value=0.05, diameter_zone=True)

    assert assessment.status == STATUS_RESOLVED_NUMERIC
    assert assessment.resolved is True
    assert assessment.tolerance_raw == "0,05"
    assert assessment.tolerance_value == 0.05
    assert assessment.diameter_zone is True
