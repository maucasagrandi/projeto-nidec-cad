import cv2
import numpy as np

from src.gdt.datum_glyph import DatumGlyphTemplateClassifier, normalize_glyph_mask


def _letter_mask(letter: str, *, scale: float = 3.4, thickness: int = 8, x: int = 20, y: int = 135):
    image = np.zeros((170, 130), dtype=np.uint8)
    cv2.putText(
        image,
        letter,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        255,
        thickness,
        cv2.LINE_8,
    )
    return image


def test_normalization_centers_and_preserves_nonempty_glyph():
    raw = _letter_mask("A", x=4, y=155)
    normalized = normalize_glyph_mask(raw, canvas_size=96, padding=10)
    assert normalized.shape == (96, 96)
    assert np.count_nonzero(normalized) > 0

    ys, xs = np.where(normalized > 0)
    center_x = (float(xs.min()) + float(xs.max())) / 2.0
    center_y = (float(ys.min()) + float(ys.max())) / 2.0
    assert abs(center_x - 47.5) <= 1.5
    assert abs(center_y - 47.5) <= 1.5


def test_classifier_ranks_same_letter_first_across_scale_change():
    classifier = DatumGlyphTemplateClassifier()
    for label in ("A", "B", "D"):
        template = normalize_glyph_mask(_letter_mask(label, scale=3.2, thickness=8))
        classifier.register(label, template, source_id=f"template-{label}")

    query = normalize_glyph_mask(_letter_mask("A", scale=3.7, thickness=9, x=10, y=145))
    ranking = classifier.rank(query)
    assert ranking
    assert ranking[0].label == "A"
    assert ranking[0].score > ranking[1].score


def test_classifier_uses_hole_topology_to_separate_b_from_a():
    classifier = DatumGlyphTemplateClassifier()
    classifier.register("A", normalize_glyph_mask(_letter_mask("A")), source_id="A")
    classifier.register("B", normalize_glyph_mask(_letter_mask("B")), source_id="B")

    query = normalize_glyph_mask(_letter_mask("B", scale=3.6, thickness=9))
    ranking = classifier.rank(query)
    assert ranking[0].label == "B"
    assert ranking[0].hole_agreement == 1.0
