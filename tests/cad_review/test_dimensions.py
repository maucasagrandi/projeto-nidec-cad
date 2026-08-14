from __future__ import annotations

import fitz

from src.cad_review.dimensions import analyze_dimension_page


def _drawing_pdf() -> bytes:
    document = fitz.open()
    page = document.new_page(width=400, height=300)

    # Drawing grid markers, duplicated on opposite margins as in a CAD sheet.
    for x, marker in ((80, "1"), (300, "2")):
        page.insert_text((x, 15), marker)
        page.insert_text((x, 294), marker)
    for y, marker in ((70, "A"), (200, "B")):
        page.insert_text((8, y), marker)
        page.insert_text((388, y), marker)

    page.insert_text((100, 100), "10.5")
    page.insert_text((250, 220), "3,5")
    page.insert_text((230, 110), "R2.0±0.1")
    page.insert_text((100, 130), "NOTE 12")
    payload = document.tobytes()
    document.close()
    return payload


def test_dimensions_are_counted_and_mapped_to_drawing_grid() -> None:
    result = analyze_dimension_page(_drawing_pdf(), 0, dpi=100)

    assert [dimension.value for dimension in result.dimensions] == ["10.5", "R2.0±0.1", "3,5"]
    assert [dimension.quadrant for dimension in result.dimensions] == ["A1", "A2", "B2"]
    assert [dimension.dimension_id for dimension in result.dimensions] == [
        "DIM-P01-001",
        "DIM-P01-002",
        "DIM-P01-003",
    ]
    assert result.annotated_image is not None
    assert result.annotated_image.shape[:2] == (417, 556)
    assert result.to_dict()["count"] == 3


def test_grid_markers_and_unstructured_integers_are_not_dimensions() -> None:
    result = analyze_dimension_page(_drawing_pdf(), 0, dpi=72)

    values = {dimension.value for dimension in result.dimensions}
    assert "1" not in values
    assert "2" not in values
    assert "12" not in values

