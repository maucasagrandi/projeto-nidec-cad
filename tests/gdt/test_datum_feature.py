import fitz

from src.gdt.datum_feature import detect_datum_feature_indicators


def _synthetic_pdf() -> bytes:
    doc = fitz.open()
    page = doc.new_page(width=220, height=180)

    page.draw_rect(fitz.Rect(50, 40, 64, 54), color=(0, 0, 0), width=0.8)
    page.insert_text((54.3, 51.0), "A", fontsize=10)
    page.draw_line(fitz.Point(57, 54), fitz.Point(57, 68), color=(0, 0, 0), width=0.8)
    shape = page.new_shape()
    shape.draw_polyline([fitz.Point(53, 75), fitz.Point(57, 68), fitz.Point(61, 75)])
    shape.finish(color=(0, 0, 0), fill=(0, 0, 0), width=0.1, closePath=True)
    shape.commit()

    page.draw_rect(fitz.Rect(100, 40, 114, 54), color=(0, 0, 0), width=0.8)
    page.insert_text((104.1, 51.0), "B", fontsize=10)

    page.insert_text((10, 120), "D", fontsize=10)

    data = doc.tobytes()
    doc.close()
    return data


def test_detects_boxed_letter_with_connected_filled_marker():
    found = detect_datum_feature_indicators(_synthetic_pdf(), raster_dpi=240)
    assert [row.label for row in found] == ["A"]
    assert found[0].marker_side == "bottom"
    assert found[0].stem_coverage >= 0.55


def test_does_not_treat_boxed_or_isolated_letters_as_datum_definitions():
    found = detect_datum_feature_indicators(_synthetic_pdf(), raster_dpi=240)
    labels = {row.label for row in found}
    assert "B" not in labels
    assert "D" not in labels
