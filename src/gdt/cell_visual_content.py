"""Deterministic visual decomposition for GD&T cell interiors.

This module operates on rasterized cell crops only. It does not perform OCR or
character classification. The goal is to separate obvious frame / leader
geometry from components that are plausible glyphs.

The distinction is intentionally diagnostic at this stage:
- ``structural_line``: long thin border / leader-like linework;
- ``arrow_like``: compact filled triangular / convex component;
- ``text_candidate``: remaining component with glyph-like proportions;
- ``other``: component that does not satisfy any strong rule.

No category is treated as ground truth. The current purpose is to verify that
single-letter datum cells can be isolated without clipping their glyphs, and to
check whether tolerance cells contain any plausible text once structural
geometry is ignored.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class VisualComponent:
    label: int
    bbox_px: tuple[int, int, int, int]
    width_px: int
    height_px: int
    area_px: int
    centroid_px: tuple[float, float]
    width_fraction: float
    height_fraction: float
    area_fraction: float
    aspect_ratio: float
    extent: float
    solidity: float
    hole_count: int
    approx_vertices: int
    touches_left: bool
    touches_top: bool
    touches_right: bool
    touches_bottom: bool
    component_class: str
    reasons: tuple[str, ...]

    @property
    def edge_touch_count(self) -> int:
        return sum(
            (
                self.touches_left,
                self.touches_top,
                self.touches_right,
                self.touches_bottom,
            )
        )

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "bbox_px": list(self.bbox_px),
            "width_px": self.width_px,
            "height_px": self.height_px,
            "area_px": self.area_px,
            "centroid_px": [round(self.centroid_px[0], 2), round(self.centroid_px[1], 2)],
            "width_fraction": round(self.width_fraction, 5),
            "height_fraction": round(self.height_fraction, 5),
            "area_fraction": round(self.area_fraction, 6),
            "aspect_ratio": round(self.aspect_ratio, 5),
            "extent": round(self.extent, 5),
            "solidity": round(self.solidity, 5),
            "hole_count": self.hole_count,
            "approx_vertices": self.approx_vertices,
            "edge_touch_count": self.edge_touch_count,
            "touches": {
                "left": self.touches_left,
                "top": self.touches_top,
                "right": self.touches_right,
                "bottom": self.touches_bottom,
            },
            "component_class": self.component_class,
            "reasons": list(self.reasons),
        }


def binarize_cell(gray: np.ndarray) -> np.ndarray:
    """Return binary image with visible ink=255 and background=0."""

    if gray.ndim != 2:
        raise ValueError("gray image must be single channel")
    _threshold, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
    )
    return binary


def _contour_features(mask: np.ndarray) -> tuple[float, float, int, int]:
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, 0.0, 0, 0

    outer_indices: list[int] = []
    hole_count = 0
    if hierarchy is not None:
        hierarchy_row = hierarchy[0]
        for index, row in enumerate(hierarchy_row):
            parent = int(row[3])
            if parent < 0:
                outer_indices.append(index)
            else:
                hole_count += 1
    else:
        outer_indices = list(range(len(contours)))

    if not outer_indices:
        outer_indices = list(range(len(contours)))

    outer = max((contours[index] for index in outer_indices), key=cv2.contourArea)
    area = float(cv2.contourArea(outer))
    hull = cv2.convexHull(outer)
    hull_area = float(cv2.contourArea(hull))
    solidity = area / hull_area if hull_area > 0.0 else 0.0

    perimeter = float(cv2.arcLength(outer, True))
    epsilon = max(0.5, 0.025 * perimeter)
    approx = cv2.approxPolyDP(outer, epsilon, True)
    return area, solidity, hole_count, int(len(approx))


def _classify_component(
    *,
    width: int,
    height: int,
    area: int,
    image_width: int,
    image_height: int,
    solidity: float,
    hole_count: int,
    approx_vertices: int,
    touches: Sequence[bool],
) -> tuple[str, tuple[str, ...]]:
    width_fraction = width / max(1.0, float(image_width))
    height_fraction = height / max(1.0, float(image_height))
    area_fraction = area / max(1.0, float(image_width * image_height))
    thin_fraction = min(width_fraction, height_fraction)
    long_fraction = max(width_fraction, height_fraction)
    edge_touch_count = sum(bool(value) for value in touches)

    reasons: list[str] = []

    # Frame borders and leader segments are usually very long and very thin.
    if long_fraction >= 0.35 and thin_fraction <= 0.055:
        reasons.append("long_thin_component")
        return "structural_line", tuple(reasons)

    # Geometry connected across much of the crop is more likely frame/leader
    # structure than a single glyph. This rule deliberately tolerates one edge
    # touch because real CAD glyphs may sit close to a cell boundary.
    if edge_touch_count >= 2 and (width_fraction >= 0.72 or height_fraction >= 0.72):
        reasons.append("large_multi_edge_component")
        return "structural_line", tuple(reasons)

    # A leader arrow can be clipped by the cell boundary, leaving only a small
    # triangular fragment. In case 41 this happened in tolerance cell 004: the
    # fragment touched the top edge and was too small for the general arrow
    # area rule, so it was incorrectly labelled as text. Hole-bearing glyphs
    # are excluded, and the rule requires a compact convex 3/4-vertex polygon
    # touching an edge so it does not broadly suppress ordinary datum glyphs.
    if (
        hole_count == 0
        and edge_touch_count >= 1
        and solidity >= 0.82
        and 3 <= approx_vertices <= 4
        and area >= 20
        and width_fraction <= 0.35
        and height_fraction <= 0.40
    ):
        reasons.extend(("edge_triangle_fragment", "few_vertices", "no_holes"))
        return "arrow_like", tuple(reasons)

    # Compact filled arrow heads are convex and have few polygon vertices.
    # Hole-bearing glyphs such as A/B/D are explicitly excluded from this rule.
    if (
        hole_count == 0
        and solidity >= 0.86
        and 3 <= approx_vertices <= 6
        and area_fraction >= 0.004
        and width_fraction <= 0.65
        and height_fraction <= 0.70
    ):
        reasons.extend(("convex_filled_shape", "few_vertices", "no_holes"))
        return "arrow_like", tuple(reasons)

    glyph_like = (
        area >= 20
        and 0.025 <= width_fraction <= 0.78
        and 0.12 <= height_fraction <= 0.98
        and area_fraction <= 0.60
    )
    if glyph_like:
        if hole_count > 0:
            reasons.append("contains_hole")
        reasons.append("glyph_like_proportions")
        return "text_candidate", tuple(reasons)

    reasons.append("no_strong_class")
    return "other", tuple(reasons)


def analyze_components(binary: np.ndarray, min_area_px: int = 8) -> list[VisualComponent]:
    """Analyze connected components without modifying the binary image."""

    if binary.ndim != 2:
        raise ValueError("binary image must be single channel")

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    image_height, image_width = binary.shape[:2]
    image_area = max(1, image_width * image_height)
    rows: list[VisualComponent] = []

    for label in range(1, count):
        x, y, width, height, area = [int(value) for value in stats[label]]
        if area < int(min_area_px):
            continue

        x1 = x + width
        y1 = y + height
        component_mask = np.zeros((height, width), dtype=np.uint8)
        local = labels[y:y1, x:x1]
        component_mask[local == label] = 255

        contour_area, solidity, hole_count, approx_vertices = _contour_features(component_mask)
        bbox_area = max(1, width * height)
        extent = float(area) / float(bbox_area)
        cx, cy = [float(value) for value in centroids[label]]

        touches_left = x <= 0
        touches_top = y <= 0
        touches_right = x1 >= image_width
        touches_bottom = y1 >= image_height
        component_class, reasons = _classify_component(
            width=width,
            height=height,
            area=area,
            image_width=image_width,
            image_height=image_height,
            solidity=solidity,
            hole_count=hole_count,
            approx_vertices=approx_vertices,
            touches=(touches_left, touches_top, touches_right, touches_bottom),
        )

        rows.append(
            VisualComponent(
                label=label,
                bbox_px=(x, y, x1, y1),
                width_px=width,
                height_px=height,
                area_px=area,
                centroid_px=(cx, cy),
                width_fraction=float(width) / max(1.0, float(image_width)),
                height_fraction=float(height) / max(1.0, float(image_height)),
                area_fraction=float(area) / float(image_area),
                aspect_ratio=float(width) / max(1.0, float(height)),
                extent=extent,
                solidity=float(solidity),
                hole_count=int(hole_count),
                approx_vertices=int(approx_vertices),
                touches_left=touches_left,
                touches_top=touches_top,
                touches_right=touches_right,
                touches_bottom=touches_bottom,
                component_class=component_class,
                reasons=reasons,
            )
        )

    rows.sort(key=lambda row: (row.bbox_px[0], row.bbox_px[1]))
    return rows


def build_text_candidate_mask(binary: np.ndarray, components: Iterable[VisualComponent]) -> np.ndarray:
    """Return mask containing only components currently labelled text_candidate."""

    output = np.zeros_like(binary)
    count, labels = cv2.connectedComponents(binary, connectivity=8)
    valid_labels = {row.label for row in components if row.component_class == "text_candidate"}
    for label in valid_labels:
        if 0 < label < count:
            output[labels == label] = 255
    return output


__all__ = [
    "VisualComponent",
    "analyze_components",
    "binarize_cell",
    "build_text_candidate_mask",
]
