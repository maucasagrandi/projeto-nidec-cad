from src.gdt.detector import BBox, GdtCell, GdtFrameCandidate
from src.gdt.frame_parser import parse_feature_control_frame


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


def test_accepts_decimal_comma_without_changing_raw_value():
    candidate = _candidate([[], ["0,03"]])

    parsed = parse_feature_control_frame(candidate, characteristic="profile")

    assert parsed.tolerance_raw == "0,03"
    assert parsed.tolerance_value == 0.03
    assert parsed.diameter_zone is None
    assert parsed.referenced_datums == []


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
