"""Deterministic extraction and annotation of drawing dimensions.

The detector reads the PDF text layer with PyMuPDF.  Grid letters and numbers
are identified only in the drawing margins, while dimension-like tokens are
collected from the drawing area and mapped to the nearest grid cell.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from typing import Any

import cv2
import fitz
import numpy as np

_GRID_LETTER = re.compile(r"^[A-H]$", re.IGNORECASE)
_GRID_NUMBER = re.compile(r"^(?:[1-9]|1[0-5])$")
_DIMENSION = re.compile(
    r"^\(?\s*(?:[⌀ØR□]\s*)?"
    r"(?:\d+[.,]\d*|\d+\s*(?:±|\+/-)\s*\d+(?:[.,]\d+)?)"
    r"(?:\s*(?:±|\+/-)\s*\d+(?:[.,]\d+)?)?\s*\)?$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DimensionRecord:
    """One dimension found in the revised drawing text layer."""

    dimension_id: str
    value: str
    page_index: int
    quadrant: str
    bbox: tuple[float, float, float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.dimension_id,
            "value": self.value,
            "page": self.page_index + 1,
            "quadrant": self.quadrant,
            "bbox": [round(value, 2) for value in self.bbox],
        }


@dataclass
class DimensionPageResult:
    """Structured and annotated dimension output for one revised page."""

    page_index: int
    dimensions: list[DimensionRecord]
    annotated_image: np.ndarray | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "page": self.page_index + 1,
            "count": len(self.dimensions),
            "items": [dimension.to_dict() for dimension in self.dimensions],
        }


def _grid_limits(markers: dict[str, list[float]]) -> list[tuple[str, float]]:
    consolidated = {
        marker: statistics.median(coordinates)
        for marker, coordinates in markers.items()
        if coordinates
    }
    ordered = sorted(consolidated.items(), key=lambda item: item[1])
    if not ordered:
        return []
    limits = [
        (marker, (coordinate + ordered[index + 1][1]) / 2.0)
        for index, (marker, coordinate) in enumerate(ordered[:-1])
    ]
    limits.append((ordered[-1][0], float("inf")))
    return limits


def _locate_quadrant(
    x: float,
    y: float,
    x_limits: list[tuple[str, float]],
    y_limits: list[tuple[str, float]],
) -> str:
    column = next((marker for marker, end in x_limits if x <= end), "?")
    row = next((marker for marker, end in y_limits if y <= end), "?")
    return f"{row}{column}"


def _iter_spans(page: fitz.Page):
    for block in page.get_text("dict").get("blocks", []):
        for line in block.get("lines", []):
            yield from line.get("spans", [])


def _extract_page_dimensions(page: fitz.Page, page_index: int) -> list[DimensionRecord]:
    width = page.rect.width
    height = page.rect.height
    top = height * 0.08
    bottom = height * 0.92
    left = width * 0.08
    right = width * 0.92

    letters_y: dict[str, list[float]] = {}
    numbers_x: dict[str, list[float]] = {}
    candidates: list[tuple[str, tuple[float, float, float, float], float, float]] = []

    for span in _iter_spans(page):
        text = str(span.get("text", "")).strip().upper()
        if not text:
            continue
        bbox = tuple(float(value) for value in span["bbox"])
        center_x = (bbox[0] + bbox[2]) / 2.0
        center_y = (bbox[1] + bbox[3]) / 2.0

        if _GRID_LETTER.fullmatch(text) and (center_x < left or center_x > right):
            letters_y.setdefault(text, []).append(center_y)
        elif _GRID_NUMBER.fullmatch(text) and (center_y < top or center_y > bottom):
            numbers_x.setdefault(text, []).append(center_x)
        elif _DIMENSION.fullmatch(text):
            candidates.append((text, bbox, center_x, center_y))

    x_limits = _grid_limits(numbers_x)
    y_limits = _grid_limits(letters_y)
    records: list[DimensionRecord] = []
    seen: set[tuple[str, int, int]] = set()
    for text, bbox, center_x, center_y in sorted(candidates, key=lambda item: (item[3], item[2])):
        # Some CAD exporters repeat the same vector text in overlapping layers.
        key = (text, round(center_x), round(center_y))
        if key in seen:
            continue
        seen.add(key)
        records.append(
            DimensionRecord(
                dimension_id=f"DIM-P{page_index + 1:02d}-{len(records) + 1:03d}",
                value=text,
                page_index=page_index,
                quadrant=_locate_quadrant(center_x, center_y, x_limits, y_limits),
                bbox=bbox,
            )
        )
    return records


def _render_annotations(
    page: fitz.Page,
    dimensions: list[DimensionRecord],
    dpi: int,
) -> np.ndarray:
    pixmap = page.get_pixmap(dpi=dpi, alpha=False)
    rgb = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, 3)
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    scale_x = pixmap.width / page.rect.width
    scale_y = pixmap.height / page.rect.height
    font_scale = max(0.42, min(0.8, dpi / 250.0))
    thickness = max(1, round(dpi / 100))

    for dimension in dimensions:
        x0, y0, x1, y1 = dimension.bbox
        padding = 4.0
        start = (max(0, round((x0 - padding) * scale_x)), max(0, round((y0 - padding) * scale_y)))
        end = (
            min(pixmap.width - 1, round((x1 + padding) * scale_x)),
            min(pixmap.height - 1, round((y1 + padding) * scale_y)),
        )
        cv2.rectangle(image, start, end, (0, 0, 255), thickness)
        short_id = dimension.dimension_id.rsplit("-", 1)[-1]
        label = f"D{short_id} [{dimension.quadrant}]"
        (label_width, label_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        label_x = start[0]
        label_y = max(label_height + 3, start[1] - 4)
        cv2.rectangle(
            image,
            (label_x, label_y - label_height - 3),
            (min(pixmap.width - 1, label_x + label_width + 4), label_y + 2),
            (255, 255, 255),
            -1,
        )
        cv2.putText(
            image,
            label,
            (label_x + 2, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 255),
            thickness,
            cv2.LINE_AA,
        )
    return image


def analyze_dimension_page(
    revised_pdf: bytes,
    page_index: int,
    *,
    dpi: int = 150,
) -> DimensionPageResult:
    """Extract, locate and annotate dimensions from one revised PDF page."""

    if dpi < 72:
        raise ValueError("dimension annotation dpi must be at least 72")
    with fitz.open(stream=revised_pdf, filetype="pdf") as document:
        if page_index < 0 or page_index >= len(document):
            raise IndexError(f"page_index {page_index} is outside the PDF")
        page = document[page_index]
        dimensions = _extract_page_dimensions(page, page_index)
        image = _render_annotations(page, dimensions, dpi)
    return DimensionPageResult(page_index, dimensions, image)


__all__ = [
    "DimensionPageResult",
    "DimensionRecord",
    "analyze_dimension_page",
]
