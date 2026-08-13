import cv2
import numpy as np

from src.gdt.symbol_classifier import (
    GLOBAL_FAMILY_WEIGHT,
    LOCAL_FAMILY_WEIGHT,
    SCORE_COMPONENTS,
    TemplateImage,
    _combine_family_scores,
    _prepare_forms,
    score_crop,
)


def _crosshair() -> np.ndarray:
    img = np.full((64, 64), 255, np.uint8)
    cv2.circle(img, (32, 32), 18, 0, 2)
    cv2.line(img, (10, 32), (54, 32), 0, 2)
    cv2.line(img, (32, 10), (32, 54), 0, 2)
    return img


def _degraded_crosshair() -> np.ndarray:
    """Position parcialmente degradado + linha longa, parecido com o caso 004."""

    img = np.full((80, 80), 255, np.uint8)
    cv2.line(img, (8, 12), (72, 12), 0, 2)  # traço/borda que favorece classes lineares
    cv2.line(img, (18, 42), (62, 42), 0, 3)
    cv2.line(img, (40, 20), (40, 64), 0, 3)
    cv2.ellipse(img, (40, 42), (18, 18), 0, 0, 180, 0, 3)
    return img


def _arc() -> np.ndarray:
    img = np.full((64, 64), 255, np.uint8)
    cv2.ellipse(img, (32, 40), (20, 20), 0, 180, 360, 0, 2)
    return img


def _straightness() -> np.ndarray:
    img = np.full((64, 64), 255, np.uint8)
    cv2.line(img, (10, 32), (54, 32), 0, 3)
    return img


def _symmetry() -> np.ndarray:
    img = np.full((64, 64), 255, np.uint8)
    for y in (24, 32, 40):
        cv2.line(img, (12, y), (52, y), 0, 3)
    return img


def _flatness() -> np.ndarray:
    img = np.full((64, 64), 255, np.uint8)
    points = np.array([[14, 40], [24, 22], [50, 22], [40, 40]], np.int32)
    cv2.polylines(img, [points], True, 0, 3)
    return img


def _circularity() -> np.ndarray:
    img = np.full((64, 64), 255, np.uint8)
    cv2.circle(img, (32, 32), 18, 0, 3)
    return img


def _cylindricity() -> np.ndarray:
    # Forma sintética apenas para testar competição entre templates; a
    # referência real da classe vem da pasta cotas/.
    img = np.full((64, 64), 255, np.uint8)
    cv2.line(img, (14, 16), (14, 48), 0, 3)
    cv2.line(img, (50, 16), (50, 48), 0, 3)
    cv2.ellipse(img, (32, 32), (12, 18), 0, 0, 360, 0, 3)
    return img


def _template(class_name: str, name: str, image: np.ndarray) -> TemplateImage:
    _, tight = _prepare_forms(image, target_size=48, margin=10)
    return TemplateImage(
        class_name=class_name,
        template_name=name,
        path=f"{name}.png",
        representations=tight,
    )


def _phase4_templates() -> list[TemplateImage]:
    return [
        _template("position", "position_synthetic", _crosshair()),
        _template("profile", "profile_synthetic", _arc()),
        _template("straightness", "straightness_synthetic", _straightness()),
        _template("symmetry", "symmetry_synthetic", _symmetry()),
        _template("flatness", "flatness_synthetic", _flatness()),
        _template("circularity", "circularity_synthetic", _circularity()),
        _template("cylindricity", "cylindricity_synthetic", _cylindricity()),
    ]


def test_identical_shape_scores_above_different_shape():
    position = _template("position", "position_synthetic", _crosshair())
    profile = _template("profile", "profile_synthetic", _arc())

    class_scores, _ = score_crop(
        _crosshair(),
        [position, profile],
        target_size=48,
        margin=10,
    )

    assert class_scores["position"] > class_scores["profile"]


def test_scores_are_reported_for_every_component():
    templates = [
        _template("position", "position_synthetic", _crosshair()),
        _template("profile", "profile_synthetic", _arc()),
    ]
    class_scores, template_scores = score_crop(_arc(), templates)

    assert set(class_scores) == {"position", "profile"}
    assert len(template_scores) == 2
    assert all(set(item.scores) == set(SCORE_COMPONENTS) for item in template_scores)


def test_score_balances_local_and_global_families():
    components = {
        "gray": 0.9,
        "binary": 0.6,
        "edges": 0.3,
        "structure": 0.8,
        "hog": 0.4,
    }

    local_mean = np.mean([0.9, 0.6, 0.3])
    global_mean = np.mean([0.8, 0.4])
    expected = LOCAL_FAMILY_WEIGHT * local_mean + GLOBAL_FAMILY_WEIGHT * global_mean

    assert np.isclose(_combine_family_scores(components), expected)
    assert np.isclose(LOCAL_FAMILY_WEIGHT + GLOBAL_FAMILY_WEIGHT, 1.0)


def test_phase4_catalog_keeps_position_as_best_for_position_shape():
    class_scores, _ = score_crop(_crosshair(), _phase4_templates())
    best_class = max(class_scores, key=class_scores.get)
    assert best_class == "position"


def test_structure_prevents_linear_class_from_stealing_degraded_position():
    class_scores, _ = score_crop(_degraded_crosshair(), _phase4_templates())
    best_class = max(class_scores, key=class_scores.get)

    assert best_class == "position"
    assert class_scores["position"] > class_scores["straightness"]
    assert class_scores["position"] > class_scores["symmetry"]


def test_phase4_catalog_keeps_profile_as_best_for_profile_shape():
    class_scores, _ = score_crop(_arc(), _phase4_templates())
    best_class = max(class_scores, key=class_scores.get)
    assert best_class == "profile"


def test_circularity_is_a_valid_class_not_a_negative_control():
    class_scores, _ = score_crop(_circularity(), _phase4_templates())
    best_class = max(class_scores, key=class_scores.get)
    assert best_class == "circularity"
    assert "negative_controls" not in class_scores
