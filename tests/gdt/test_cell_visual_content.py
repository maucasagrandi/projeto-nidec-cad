import cv2
import numpy as np

from src.gdt.cell_visual_content import analyze_components


def _blank(width=220, height=140):
    return np.zeros((height, width), dtype=np.uint8)


def test_long_horizontal_line_is_structural():
    image = _blank()
    cv2.rectangle(image, (0, 20), (210, 24), 255, thickness=-1)
    rows = analyze_components(image)
    assert len(rows) == 1
    assert rows[0].component_class == "structural_line"


def test_filled_triangle_is_arrow_like():
    image = _blank()
    triangle = np.array([[70, 30], [135, 75], [90, 100]], dtype=np.int32)
    cv2.fillPoly(image, [triangle], 255)
    rows = analyze_components(image)
    assert len(rows) == 1
    assert rows[0].component_class == "arrow_like"


def test_small_triangle_clipped_at_top_edge_is_arrow_like():
    image = _blank(width=380, height=140)
    triangle = np.array([[245, 0], [278, 0], [250, 27]], dtype=np.int32)
    cv2.fillPoly(image, [triangle], 255)
    rows = analyze_components(image)
    assert len(rows) == 1
    assert rows[0].touches_top
    assert rows[0].component_class == "arrow_like"
    assert "edge_triangle_fragment" in rows[0].reasons


def test_letter_a_like_shape_is_text_candidate():
    image = _blank(width=160, height=180)
    cv2.putText(image, "A", (35, 145), cv2.FONT_HERSHEY_SIMPLEX, 3.6, 255, 9, cv2.LINE_8)
    rows = analyze_components(image)
    candidates = [row for row in rows if row.component_class == "text_candidate"]
    assert candidates


def test_vertical_border_and_letter_are_separated():
    image = _blank(width=200, height=180)
    cv2.rectangle(image, (2, 0), (6, 179), 255, thickness=-1)
    cv2.putText(image, "B", (55, 145), cv2.FONT_HERSHEY_SIMPLEX, 3.6, 255, 9, cv2.LINE_8)
    rows = analyze_components(image)
    classes = [row.component_class for row in rows]
    assert "structural_line" in classes
    assert "text_candidate" in classes
