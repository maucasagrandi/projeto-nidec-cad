"""Deterministic detection of datum feature indicators in vector CAD PDFs.

Phase 7 question: a GD&T frame references datum ``A`` / ``B`` / ..., but is
that datum actually defined somewhere in the drawing?

This detector intentionally does not treat an isolated uppercase letter as a
datum definition. A candidate must combine three independent signals:

1. a single uppercase PDF text token;
2. a small near-square outline enclosing the token;
3. a nearby filled triangular marker connected to the box by a visible stem.

The implementation uses PyMuPDF text/vector geometry plus a deterministic
OpenCV raster check. It does not use OCR or an LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Optional

import cv2
import fitz
import numpy as np

_SINGLE_DATUM_RE = re.compile(r"^[A-Z]$")


@dataclass(frozen=True)
class DatumFeatureIndicatorCandidate:
    label: str
    page: int
    text_bbox: tuple[float, float, float, float]
    box_bbox: tuple[float, float, float, float]
    marker_bbox: tuple[float, float, float, float]
    marker_side: str
    stem_coverage: float
    box_rectangularity: float
    source: str = "pdf_text+raster_box+vector_filled_marker"

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "page": self.page,
            "text_bbox": [round(value, 4) for value in self.text_bbox],
            "box_bbox": [round(value, 4) for value in self.box_bbox],
            "marker_bbox": [round(value, 4) for value in self.marker_bbox],
            "marker_side": self.marker_side,
            "stem_coverage": round(self.stem_coverage, 4),
            "box_rectangularity": round(self.box_rectangularity, 4),
            "source": self.source,
        }


@dataclass(frozen=True)
class _RectCandidate:
    bbox: tuple[float, float, float, float]
    rectangularity: float


@dataclass(frozen=True)
class _MarkerCandidate:
    bbox: tuple[float, float, float, float]
    drawing_index: int


def _render_binary(page: fitz.Page, dpi: int) -> tuple[np.ndarray, float]:
    scale = float(dpi) / 72.0
    pix = page.get_pixmap(
        matrix=fitz.Matrix(scale, scale),
        colorspace=fitz.csGRAY,
        alpha=False,
    )
    gray = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width).copy()
    _threshold, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    return binary, scale


def _detect_small_boxes(
    binary: np.ndarray,
    *,
    scale: float,
    min_size_pt: float,
    max_size_pt: float,
    min_aspect: float,
    max_aspect: float,
    min_rectangularity: float,
) -> list[_RectCandidate]:
    contours, _hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    raw: list[_RectCandidate] = []

    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        width_pt = width / scale
        height_pt = height / scale
        if not (min_size_pt <= width_pt <= max_size_pt):
            continue
        if not (min_size_pt <= height_pt <= max_size_pt):
            continue

        aspect = width_pt / max(height_pt, 1e-6)
        if not (min_aspect <= aspect <= max_aspect):
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) != 4:
            continue

        contour_area = float(cv2.contourArea(contour))
        rectangularity = contour_area / max(1.0, float(width * height))
        if rectangularity < min_rectangularity:
            continue

        raw.append(
            _RectCandidate(
                bbox=(
                    float(x / scale),
                    float(y / scale),
                    float((x + width) / scale),
                    float((y + height) / scale),
                ),
                rectangularity=rectangularity,
            )
        )

    deduped: list[_RectCandidate] = []
    for candidate in sorted(raw, key=lambda row: -row.rectangularity):
        x0, y0, x1, y1 = candidate.bbox
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
        duplicate = False
        for existing in deduped:
            ex0, ey0, ex1, ey1 = existing.bbox
            ecx = 0.5 * (ex0 + ex1)
            ecy = 0.5 * (ey0 + ey1)
            if abs(cx - ecx) <= 1.0 and abs(cy - ecy) <= 1.0:
                duplicate = True
                break
        if not duplicate:
            deduped.append(candidate)
    return deduped


def _detect_filled_triangle_markers(
    page: fitz.Page,
    *,
    max_marker_size_pt: float,
    max_fill_channel: float,
) -> list[_MarkerCandidate]:
    output: list[_MarkerCandidate] = []
    for drawing_index, drawing in enumerate(page.get_drawings()):
        fill = drawing.get("fill")
        if fill is None:
            continue
        try:
            if max(float(channel) for channel in fill) > max_fill_channel:
                continue
        except TypeError:
            continue

        rect = fitz.Rect(drawing.get("rect"))
        if rect.width <= 0 or rect.height <= 0:
            continue
        if rect.width > max_marker_size_pt or rect.height > max_marker_size_pt:
            continue
        if rect.width < 1.5 or rect.height < 1.5:
            continue

        line_items = [item for item in drawing.get("items", []) if item and item[0] == "l"]
        if not (3 <= len(line_items) <= 5):
            continue

        output.append(
            _MarkerCandidate(
                bbox=(float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)),
                drawing_index=drawing_index,
            )
        )
    return output


def _corridor_coverage(
    binary: np.ndarray,
    *,
    scale: float,
    box_bbox: tuple[float, float, float, float],
    marker_bbox: tuple[float, float, float, float],
    side: str,
    half_width_pt: float = 1.25,
) -> float:
    x0, y0, x1, y1 = box_bbox
    tx0, ty0, tx1, ty1 = marker_bbox
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)

    def _clip_x(value: float) -> int:
        return int(min(binary.shape[1], max(0, round(value * scale))))

    def _clip_y(value: float) -> int:
        return int(min(binary.shape[0], max(0, round(value * scale))))

    if side in {"bottom", "top"}:
        if side == "bottom":
            start, stop = y1, ty0
        else:
            start, stop = ty1, y0
        if stop < start:
            return 0.0
        x_start = _clip_x(cx - half_width_pt)
        x_stop = max(_clip_x(cx + half_width_pt), x_start + 1)
        y_start = _clip_y(start)
        y_stop = max(_clip_y(stop), y_start + 1)
        region = binary[y_start:y_stop, x_start:x_stop]
        if region.size == 0:
            return 0.0
        return float((region.max(axis=1) > 0).mean())

    if side in {"right", "left"}:
        if side == "right":
            start, stop = x1, tx0
        else:
            start, stop = tx1, x0
        if stop < start:
            return 0.0
        x_start = _clip_x(start)
        x_stop = max(_clip_x(stop), x_start + 1)
        y_start = _clip_y(cy - half_width_pt)
        y_stop = max(_clip_y(cy + half_width_pt), y_start + 1)
        region = binary[y_start:y_stop, x_start:x_stop]
        if region.size == 0:
            return 0.0
        return float((region.max(axis=0) > 0).mean())

    return 0.0


def _candidate_marker_pairs(
    box_bbox: tuple[float, float, float, float],
    markers: Iterable[_MarkerCandidate],
    *,
    max_gap_pt: float,
) -> list[tuple[str, float, _MarkerCandidate]]:
    x0, y0, x1, y1 = box_bbox
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    width = x1 - x0
    height = y1 - y0
    output: list[tuple[str, float, _MarkerCandidate]] = []

    for marker in markers:
        tx0, ty0, tx1, ty1 = marker.bbox
        tcx = 0.5 * (tx0 + tx1)
        tcy = 0.5 * (ty0 + ty1)

        if abs(tcx - cx) <= max(2.5, 0.35 * width):
            bottom_gap = ty0 - y1
            if 0.0 <= bottom_gap <= max_gap_pt:
                output.append(("bottom", bottom_gap, marker))
            top_gap = y0 - ty1
            if 0.0 <= top_gap <= max_gap_pt:
                output.append(("top", top_gap, marker))

        if abs(tcy - cy) <= max(2.5, 0.35 * height):
            right_gap = tx0 - x1
            if 0.0 <= right_gap <= max_gap_pt:
                output.append(("right", right_gap, marker))
            left_gap = x0 - tx1
            if 0.0 <= left_gap <= max_gap_pt:
                output.append(("left", left_gap, marker))

    return output


def detect_datum_feature_indicators(
    pdf_bytes: bytes,
    *,
    page_index: int = 0,
    raster_dpi: int = 200,
    min_box_size_pt: float = 7.0,
    max_box_size_pt: float = 24.0,
    min_box_aspect: float = 0.65,
    max_box_aspect: float = 1.50,
    min_rectangularity: float = 0.45,
    max_marker_size_pt: float = 16.0,
    max_marker_gap_pt: float = 30.0,
    min_stem_coverage: float = 0.55,
) -> list[DatumFeatureIndicatorCandidate]:
    """Detect datum feature indicators without treating isolated letters as datums."""

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        if page_index < 0 or page_index >= len(doc):
            raise IndexError(f"page_index {page_index} outside PDF with {len(doc)} page(s)")
        page = doc[page_index]
        binary, scale = _render_binary(page, raster_dpi)
        boxes = _detect_small_boxes(
            binary,
            scale=scale,
            min_size_pt=min_box_size_pt,
            max_size_pt=max_box_size_pt,
            min_aspect=min_box_aspect,
            max_aspect=max_box_aspect,
            min_rectangularity=min_rectangularity,
        )
        markers = _detect_filled_triangle_markers(
            page,
            max_marker_size_pt=max_marker_size_pt,
            max_fill_channel=0.25,
        )

        words: list[tuple[str, tuple[float, float, float, float], float, float]] = []
        for word in page.get_text("words"):
            label = str(word[4]).strip().upper()
            if not _SINGLE_DATUM_RE.fullmatch(label):
                continue
            bbox = (float(word[0]), float(word[1]), float(word[2]), float(word[3]))
            words.append((label, bbox, 0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3])))

        output: list[DatumFeatureIndicatorCandidate] = []
        for label, text_bbox, text_cx, text_cy in words:
            containing_boxes = [
                box
                for box in boxes
                if box.bbox[0] <= text_cx <= box.bbox[2]
                and box.bbox[1] <= text_cy <= box.bbox[3]
            ]
            if not containing_boxes:
                continue

            best: Optional[DatumFeatureIndicatorCandidate] = None
            best_key: Optional[tuple[float, float]] = None
            for box in containing_boxes:
                pairs = _candidate_marker_pairs(box.bbox, markers, max_gap_pt=max_marker_gap_pt)
                for side, gap, marker in pairs:
                    coverage = _corridor_coverage(
                        binary,
                        scale=scale,
                        box_bbox=box.bbox,
                        marker_bbox=marker.bbox,
                        side=side,
                    )
                    if coverage < min_stem_coverage:
                        continue
                    candidate = DatumFeatureIndicatorCandidate(
                        label=label,
                        page=page_index + 1,
                        text_bbox=text_bbox,
                        box_bbox=box.bbox,
                        marker_bbox=marker.bbox,
                        marker_side=side,
                        stem_coverage=coverage,
                        box_rectangularity=box.rectangularity,
                    )
                    key = (-coverage, gap)
                    if best is None or (best_key is not None and key < best_key):
                        best = candidate
                        best_key = key
                    elif best is None:
                        best = candidate
                        best_key = key
            if best is not None:
                output.append(best)

        output.sort(key=lambda row: (row.page, row.box_bbox[1], row.box_bbox[0], row.label))
        return output
    finally:
        doc.close()


__all__ = ["DatumFeatureIndicatorCandidate", "detect_datum_feature_indicators"]
