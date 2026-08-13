from src.gdt.detector import BBox, GdtCell, GdtFrameCandidate
from src.gdt.frame_parser import FrameVisualEvidence, parse_feature_control_frame
from src.gdt.tolerance_cell import assessment_from_filter_row, resolved_numeric_assessment


def _candidate(cell_texts: list[list[str]]) -> GdtFrameCandidate:
    cells = []
    x = 0.0
    for texts in cell_texts:
        cells.append(GdtCell(bbox=BBox(x, 0.0, x + 10.0, 10.0), texts=texts))
        x += 10.0
    return GdtFrameCandidate(
        candidate_id="GDT-CAND-TEST-001",
        page=1,
        frame_bbox=BBox(0.0, 0.0, x, 10.0),
        symbol_bbox=cells[0].bbox,
        cells=cells,
    )


def test_parses_position_tolerance_diameter_and_datums():
    candidate = _candidate([[], ["⌀", "0.05"], ["A"], ["B"], ["C"]])

    parsed = parse_feature_control_frame(candidate, characteristic="position")

    assert parsed.characteristic == "position"
    assert parsed.tolerance_raw == "0.05"
    assert parsed.tolerance_value == 0.05
    assert parsed.diameter_zone is True
    assert parsed.referenced_datums == ["A", "B", "C"]
    assert parsed.unresolved_tokens == []
    assert parsed.unresolved_fields == []
    assert parsed.field_sources["tolerance"] == "pdf_cell_text"


def test_accepts_decimal_comma_without_changing_raw_value():
    candidate = _candidate([[], ["0,03"]])

    parsed = parse_feature_control_frame(candidate, characteristic="profile")

    assert parsed.tolerance_raw == "0,03"
    assert parsed.tolerance_value == 0.03
    assert parsed.diameter_zone is None
    assert parsed.referenced_datums == []


def test_accepts_leading_dot_tolerance_without_changing_raw_value():
    candidate = _candidate([[], [".05"]])

    parsed = parse_feature_control_frame(candidate, characteristic="profile")

    assert parsed.tolerance_raw == ".05"
    assert parsed.tolerance_value == 0.05


def test_diameter_absence_in_text_is_unknown_not_false():
    candidate = _candidate([[], ["0.10"], ["A"]])

    parsed = parse_feature_control_frame(candidate, characteristic="position")

    assert parsed.diameter_zone is None


def test_datum_cell_must_be_textually_unambiguous():
    candidate = _candidate([[], ["0.05"], ["A", "M"], ["B"]])

    parsed = parse_feature_control_frame(candidate, characteristic="position")

    assert parsed.referenced_datums == ["B"]
    assert "A" in parsed.unresolved_tokens
    assert "M" in parsed.unresolved_tokens


def test_missing_tolerance_is_reported_as_unresolved():
    candidate = _candidate([[], ["ABC"]])

    parsed = parse_feature_control_frame(candidate, characteristic="profile")

    assert parsed.tolerance_raw is None
    assert parsed.tolerance_value is None
    assert "tolerance_value" in parsed.unresolved_fields
    assert parsed.unresolved_tokens == ["ABC"]


def test_missing_characteristic_is_not_guessed():
    candidate = _candidate([[], ["0.05"]])

    parsed = parse_feature_control_frame(candidate)

    assert parsed.characteristic is None
    assert "characteristic" in parsed.unresolved_fields


def test_visual_datum_fills_empty_pdf_cell_without_overwriting_text():
    candidate = _candidate([[], [], [], []])
    evidence = FrameVisualEvidence(
        datum_by_cell={2: "A", 3: "B"},
        source="datum_glyph_classifier",
    )

    parsed = parse_feature_control_frame(
        candidate,
        characteristic="position",
        visual_evidence=evidence,
    )

    assert parsed.referenced_datums == ["A", "B"]
    assert parsed.field_sources["datum_cells"] == {
        "2": "datum_glyph_classifier",
        "3": "datum_glyph_classifier",
    }
    assert "tolerance_value" in parsed.unresolved_fields


def test_text_datum_has_priority_when_visual_evidence_agrees():
    candidate = _candidate([[], ["0.05"], ["A"]])
    evidence = FrameVisualEvidence(datum_by_cell={2: "A"}, source="datum_glyph_classifier")

    parsed = parse_feature_control_frame(
        candidate,
        characteristic="position",
        visual_evidence=evidence,
    )

    assert parsed.referenced_datums == ["A"]
    assert parsed.field_sources["datum_cells"]["2"] == "pdf_cell_text"


def test_text_visual_datum_conflict_stays_unresolved():
    candidate = _candidate([[], ["0.05"], ["A"]])
    evidence = FrameVisualEvidence(datum_by_cell={2: "B"}, source="datum_glyph_classifier")

    parsed = parse_feature_control_frame(
        candidate,
        characteristic="position",
        visual_evidence=evidence,
    )

    assert parsed.referenced_datums == []
    assert "datum_cell_2_conflict" in parsed.unresolved_fields
    assert any("text=A visual=B" in note for note in parsed.evidence_notes)


def test_unresolved_visual_tolerance_is_preserved_as_unresolved_evidence():
    candidate = _candidate([[], [], ["A"]])
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
    evidence = FrameVisualEvidence(tolerance_assessment=assessment)

    parsed = parse_feature_control_frame(
        candidate,
        characteristic="position",
        visual_evidence=evidence,
    )

    assert parsed.tolerance_value is None
    assert "tolerance_value" in parsed.unresolved_fields
    assert any("unresolved_no_text_candidate" in note for note in parsed.evidence_notes)


def test_resolved_visual_tolerance_can_fill_missing_pdf_text():
    candidate = _candidate([[], [], ["A"]])
    evidence = FrameVisualEvidence(
        tolerance_assessment=resolved_numeric_assessment(
            raw="0,05",
            value=0.05,
            diameter_zone=True,
            source="visual_numeric_classifier",
        )
    )

    parsed = parse_feature_control_frame(
        candidate,
        characteristic="position",
        visual_evidence=evidence,
    )

    assert parsed.tolerance_raw == "0,05"
    assert parsed.tolerance_value == 0.05
    assert parsed.diameter_zone is True
    assert parsed.field_sources["tolerance"] == "visual_numeric_classifier"
    assert "tolerance_value" not in parsed.unresolved_fields
