"""Deterministic single-letter datum glyph normalization and template scoring.

This module receives an already-isolated ``text_candidate`` component from a
GD&T datum cell. It does not locate frames/cells and it does not use OCR or an
LLM.

The classifier is deliberately template based. A glyph is normalized to a
fixed canvas and compared against registered examples using complementary shape
signals:
- binary Dice overlap;
- bidirectional chamfer similarity;
- OpenCV contour shape similarity;
- aspect-ratio similarity;
- hole-count agreement.

No global acceptance threshold is defined here. Callers receive ranked scores
and decide whether the evidence is sufficient. This is important because the
initial case-41 templates are bootstrap references, not production ground
truth.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Optional

import cv2
import numpy as np

from src.gdt.cell_visual_content import VisualComponent


DEFAULT_CANVAS_SIZE = 96
DEFAULT_PADDING = 10


def _largest_contour(binary: np.ndarray) -> Optional[np.ndarray]:
    contours, _hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _hole_count(binary: np.ndarray) -> int:
    _contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return 0
    return sum(1 for row in hierarchy[0] if int(row[3]) >= 0)


def extract_component_mask(binary: np.ndarray, component: VisualComponent) -> np.ndarray:
    """Return a full-size mask containing only ``component.label``."""

    if binary.ndim != 2:
        raise ValueError("binary image must be single channel")
    count, labels = cv2.connectedComponents(binary, connectivity=8)
    if component.label <= 0 or component.label >= count:
        raise ValueError(f"component label {component.label} not present in binary image")
    output = np.zeros_like(binary)
    output[labels == component.label] = 255
    return output


def normalize_glyph_mask(
    component_mask: np.ndarray,
    *,
    canvas_size: int = DEFAULT_CANVAS_SIZE,
    padding: int = DEFAULT_PADDING,
) -> np.ndarray:
    """Crop, scale and center a single glyph on a square binary canvas."""

    if component_mask.ndim != 2:
        raise ValueError("component mask must be single channel")
    if canvas_size <= 2 * padding + 2:
        raise ValueError("canvas_size too small for requested padding")

    ys, xs = np.where(component_mask > 0)
    if xs.size == 0 or ys.size == 0:
        return np.zeros((canvas_size, canvas_size), dtype=np.uint8)

    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    crop = component_mask[y0:y1, x0:x1]

    target = canvas_size - 2 * padding
    scale = min(target / max(1, crop.shape[1]), target / max(1, crop.shape[0]))
    new_width = max(1, int(round(crop.shape[1] * scale)))
    new_height = max(1, int(round(crop.shape[0] * scale)))
    resized = cv2.resize(crop, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    left = (canvas_size - new_width) // 2
    top = (canvas_size - new_height) // 2
    canvas[top : top + new_height, left : left + new_width] = resized
    return canvas


def normalized_component_from_cell(
    binary: np.ndarray,
    component: VisualComponent,
    *,
    canvas_size: int = DEFAULT_CANVAS_SIZE,
    padding: int = DEFAULT_PADDING,
) -> np.ndarray:
    return normalize_glyph_mask(
        extract_component_mask(binary, component),
        canvas_size=canvas_size,
        padding=padding,
    )


def _dice_similarity(a: np.ndarray, b: np.ndarray) -> float:
    aa = a > 0
    bb = b > 0
    denom = int(aa.sum()) + int(bb.sum())
    if denom == 0:
        return 1.0
    intersection = int(np.logical_and(aa, bb).sum())
    return float(2.0 * intersection / denom)


def _distance_to_ink(binary: np.ndarray) -> np.ndarray:
    # distanceTransform measures distance from non-zero pixels to zero pixels.
    # Invert the ink mask so template/query ink becomes zero.
    inverted = np.where(binary > 0, 0, 255).astype(np.uint8)
    return cv2.distanceTransform(inverted, cv2.DIST_L2, 3)


def _chamfer_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_ink = a > 0
    b_ink = b > 0
    if not a_ink.any() or not b_ink.any():
        return 0.0

    dist_to_b = _distance_to_ink(b)
    dist_to_a = _distance_to_ink(a)
    mean_ab = float(dist_to_b[a_ink].mean())
    mean_ba = float(dist_to_a[b_ink].mean())
    mean_distance = 0.5 * (mean_ab + mean_ba)

    diagonal = math.hypot(float(a.shape[1]), float(a.shape[0]))
    normalized = mean_distance / max(1.0, diagonal)
    return float(math.exp(-12.0 * normalized))


def _contour_similarity(a: np.ndarray, b: np.ndarray) -> float:
    ca = _largest_contour(a)
    cb = _largest_contour(b)
    if ca is None or cb is None:
        return 0.0
    distance = float(cv2.matchShapes(ca, cb, cv2.CONTOURS_MATCH_I1, 0.0))
    return 1.0 / (1.0 + 4.0 * max(0.0, distance))


def _aspect_ratio(binary: np.ndarray) -> float:
    ys, xs = np.where(binary > 0)
    if xs.size == 0 or ys.size == 0:
        return 1.0
    width = float(xs.max() - xs.min() + 1)
    height = float(ys.max() - ys.min() + 1)
    return width / max(1.0, height)


def _aspect_similarity(a: np.ndarray, b: np.ndarray) -> float:
    ra = max(1e-6, _aspect_ratio(a))
    rb = max(1e-6, _aspect_ratio(b))
    return float(math.exp(-abs(math.log(ra / rb))))


@dataclass(frozen=True)
class DatumGlyphTemplate:
    label: str
    image: np.ndarray
    source_id: str
    hole_count: int


@dataclass(frozen=True)
class DatumGlyphMatch:
    label: str
    score: float
    source_id: str
    dice: float
    chamfer: float
    contour: float
    aspect: float
    hole_agreement: float

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "score": round(self.score, 6),
            "source_id": self.source_id,
            "dice": round(self.dice, 6),
            "chamfer": round(self.chamfer, 6),
            "contour": round(self.contour, 6),
            "aspect": round(self.aspect, 6),
            "hole_agreement": round(self.hole_agreement, 6),
        }


class DatumGlyphTemplateClassifier:
    """Rank isolated uppercase datum glyphs against registered templates."""

    def __init__(self) -> None:
        self._templates: list[DatumGlyphTemplate] = []

    @property
    def template_count(self) -> int:
        return len(self._templates)

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(sorted({template.label for template in self._templates}))

    def register(self, label: str, normalized_image: np.ndarray, *, source_id: str) -> None:
        normalized_label = str(label).strip().upper()
        if len(normalized_label) != 1 or not normalized_label.isalpha():
            raise ValueError("datum template label must be one alphabetic character")
        if normalized_image.ndim != 2:
            raise ValueError("normalized_image must be single channel")
        image = np.where(normalized_image > 0, 255, 0).astype(np.uint8)
        self._templates.append(
            DatumGlyphTemplate(
                label=normalized_label,
                image=image.copy(),
                source_id=str(source_id),
                hole_count=_hole_count(image),
            )
        )

    @staticmethod
    def _score(query: np.ndarray, template: DatumGlyphTemplate) -> DatumGlyphMatch:
        dice = _dice_similarity(query, template.image)
        chamfer = _chamfer_similarity(query, template.image)
        contour = _contour_similarity(query, template.image)
        aspect = _aspect_similarity(query, template.image)
        query_holes = _hole_count(query)
        hole_agreement = 1.0 if query_holes == template.hole_count else 0.0

        score = (
            0.30 * dice
            + 0.30 * chamfer
            + 0.22 * contour
            + 0.10 * aspect
            + 0.08 * hole_agreement
        )
        return DatumGlyphMatch(
            label=template.label,
            score=float(score),
            source_id=template.source_id,
            dice=float(dice),
            chamfer=float(chamfer),
            contour=float(contour),
            aspect=float(aspect),
            hole_agreement=float(hole_agreement),
        )

    def rank(self, normalized_query: np.ndarray) -> list[DatumGlyphMatch]:
        if not self._templates:
            return []
        query = np.where(normalized_query > 0, 255, 0).astype(np.uint8)
        raw = [self._score(query, template) for template in self._templates]

        # Multiple templates per letter are allowed. Keep the best exemplar for
        # each label so ranking is by datum letter, not by template count.
        best_by_label: dict[str, DatumGlyphMatch] = {}
        for match in raw:
            current = best_by_label.get(match.label)
            if current is None or match.score > current.score:
                best_by_label[match.label] = match
        return sorted(best_by_label.values(), key=lambda row: (-row.score, row.label))


__all__ = [
    "DEFAULT_CANVAS_SIZE",
    "DEFAULT_PADDING",
    "DatumGlyphMatch",
    "DatumGlyphTemplate",
    "DatumGlyphTemplateClassifier",
    "extract_component_mask",
    "normalize_glyph_mask",
    "normalized_component_from_cell",
]
