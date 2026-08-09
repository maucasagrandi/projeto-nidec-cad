import cv2
import numpy as np

from src.gdt.symbol_classifier import TemplateImage, _prepare_forms, score_crop


def _crosshair() -> np.ndarray:
    img = np.full((64, 64), 255, np.uint8)
    cv2.circle(img, (32, 32), 18, 0, 2)
    cv2.line(img, (10, 32), (54, 32), 0, 2)
    cv2.line(img, (32, 10), (32, 54), 0, 2)
    return img


def _arc() -> np.ndarray:
    img = np.full((64, 64), 255, np.uint8)
    cv2.ellipse(img, (32, 40), (20, 20), 0, 180, 360, 0, 2)
    return img


def _template(class_name: str, name: str, image: np.ndarray) -> TemplateImage:
    _, tight = _prepare_forms(image, target_size=48, margin=10)
    return TemplateImage(
        class_name=class_name,
        template_name=name,
        path=f"{name}.png",
        representations=tight,
    )


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


def test_scores_are_reported_for_every_class():
    templates = [
        _template("position", "position_synthetic", _crosshair()),
        _template("profile", "profile_synthetic", _arc()),
    ]
    class_scores, template_scores = score_crop(_arc(), templates)

    assert set(class_scores) == {"position", "profile"}
    assert len(template_scores) == 2
    assert all(set(item.scores) == {"gray", "binary", "edges"} for item in template_scores)
