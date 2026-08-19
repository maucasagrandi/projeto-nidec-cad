"""Datum definition finder.

A datum definition is a single letter inside a small standalone rectangular
box, placed on the drawing to identify a datum feature. It is the SAME size as
an FCF datum reference cell (~12-13pt wide, ~14pt tall), but stands alone
outside any feature control frame.

Strategy (geometric, not image-template):
1. Extract horizontal/vertical line segments from the PDF page.
2. Find small rectangles (2 horizontal + 2 vertical lines) at datum-cell size.
3. Exclude boxes inside FCF frames (those are datum references, not definitions).
4. Exclude boxes at the page border (drawing grid zone labels).
5. Keep boxes whose interior contains ink (a letter glyph).

This never matches bare triangles or arrowheads because a datum box is, by
construction, a closed four-sided rectangle. Triangles are ignored entirely.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import fitz
import numpy as np

from src.gdt.fcf_expander import FcfFrame, extract_page_lines


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class DatumDefinition:
    """A datum feature definition (letter in a standalone box)."""

    x: float  # center x in PDF points
    y: float  # center y in PDF points
    width: float  # box width in PDF points
    height: float  # box height in PDF points
    ink_ratio: float  # interior content density (letter presence)

    def to_dict(self) -> dict:
        return {
            "center_pt": [round(self.x, 1), round(self.y, 1)],
            "bbox_pt": [
                round(self.x - self.width / 2, 1),
                round(self.y - self.height / 2, 1),
                round(self.x + self.width / 2, 1),
                round(self.y + self.height / 2, 1),
            ],
            "size_pt": [round(self.width, 1), round(self.height, 1)],
            "ink_ratio": round(self.ink_ratio, 3),
        }


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Datum indicator boxes match FCF datum cell dimensions.
MIN_BOX_WIDTH = 10.0
MAX_BOX_WIDTH = 16.0
MIN_BOX_HEIGHT = 12.0
MAX_BOX_HEIGHT = 17.0

# Line-alignment tolerance when assembling a rectangle from segments (points).
BOX_TOLERANCE = 2.5

# Minimum interior ink to confirm a letter is present.
MIN_INK_RATIO = 0.04

# Page border margin — grid zone labels (A-H / 1-8) live within this band.
BORDER_MARGIN_PT = 90.0


# ---------------------------------------------------------------------------
# Box detection
# ---------------------------------------------------------------------------


def _find_boxes(
    h_lines: List[Tuple[float, float, float]],
    v_lines: List[Tuple[float, float, float]],
) -> List[Tuple[float, float, float, float]]:
    """Find small rectangles formed by two horizontal + two vertical lines.

    Returns list of (x0, y0, x1, y1) boxes at datum-cell size.
    """
    boxes: List[Tuple[float, float, float, float]] = []
    hs = sorted(h_lines, key=lambda l: l[1])  # by y

    for i in range(len(hs)):
        x0a, ya, x1a = hs[i]
        for j in range(i + 1, len(hs)):
            x0b, yb, x1b = hs[j]
            box_h = yb - ya
            if box_h < MIN_BOX_HEIGHT:
                continue
            if box_h > MAX_BOX_HEIGHT:
                break  # hs sorted by y, no taller box will qualify

            left = max(x0a, x0b)
            right = min(x1a, x1b)
            box_w = right - left
            if not (MIN_BOX_WIDTH <= box_w <= MAX_BOX_WIDTH):
                continue

            # Require matching vertical lines on left and right edges
            has_left = any(
                abs(vx - left) < BOX_TOLERANCE
                and vy0 <= ya + BOX_TOLERANCE and vy1 >= yb - BOX_TOLERANCE
                for vx, vy0, vy1 in v_lines
            )
            has_right = any(
                abs(vx - right) < BOX_TOLERANCE
                and vy0 <= ya + BOX_TOLERANCE and vy1 >= yb - BOX_TOLERANCE
                for vx, vy0, vy1 in v_lines
            )
            if has_left and has_right:
                boxes.append((left, ya, right, yb))

    # Collapse heavily overlapping boxes (keep the first occurrence)
    boxes = _dedupe_overlapping(boxes)

    # Reject table cells: a datum box is isolated, whereas table cells align in
    # columns/rows. Drop any box whose left edge is shared (within tolerance) by
    # 2+ other boxes, or whose top edge is shared by 2+ others.
    boxes = _reject_grid_cells(boxes)

    return boxes


def _dedupe_overlapping(
    boxes: List[Tuple[float, float, float, float]],
) -> List[Tuple[float, float, float, float]]:
    """Remove boxes that overlap an already-kept box (IoU > 0.3)."""
    kept: List[Tuple[float, float, float, float]] = []
    for b in boxes:
        if not any(_iou(b, k) > 0.3 for k in kept):
            kept.append(b)
    return kept


def _iou(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    ix0 = max(a[0], b[0]); iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2]); iy1 = min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _reject_grid_cells(
    boxes: List[Tuple[float, float, float, float]],
    align_tol: float = 3.0,
    min_run: int = 3,
) -> List[Tuple[float, float, float, float]]:
    """Reject boxes that belong to a table (aligned column or row of >=3).

    A datum indicator box stands alone. Table cells share a common left edge
    (a column) or top edge (a row). Any box sharing its left-x with >=min_run
    boxes, or its top-y with >=min_run boxes, is treated as a table cell.
    """
    keep = []
    for b in boxes:
        col = sum(1 for o in boxes if abs(o[0] - b[0]) < align_tol)
        row = sum(1 for o in boxes if abs(o[1] - b[1]) < align_tol)
        if col >= min_run or row >= min_run:
            continue  # part of a table grid
        keep.append(b)
    return keep


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------


def find_datum_definitions(
    pdf_bytes: bytes,
    page_gray: np.ndarray,
    zoom: float,
    frames: Sequence[FcfFrame],
    *,
    page_index: int = 0,
) -> List[DatumDefinition]:
    """Find datum definition boxes on the page.

    Parameters
    ----------
    pdf_bytes : bytes
        PDF content (for vector line extraction).
    page_gray : np.ndarray
        Page rendered as grayscale (for ink verification).
    zoom : float
        Rendering zoom factor (dpi / 72).
    frames : sequence of FcfFrame
        Known FCF frames — boxes inside these are references, not definitions.
    page_index : int
        Page to analyze.

    Returns
    -------
    List of DatumDefinition objects.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        page = doc[page_index]
        h_lines, v_lines = extract_page_lines(page)
        page_w = page.rect.width
        page_h = page.rect.height
    finally:
        doc.close()

    boxes = _find_boxes(h_lines, v_lines)

    fcf_zones = [(f.x0, f.y0, f.x1, f.y1) for f in frames]
    ph, pw = page_gray.shape[:2]

    definitions: List[DatumDefinition] = []
    for bx0, by0, bx1, by1 in boxes:
        cx = (bx0 + bx1) / 2
        cy = (by0 + by1) / 2

        # Exclude page border zone labels
        if (cx < BORDER_MARGIN_PT or cx > page_w - BORDER_MARGIN_PT or
                cy < BORDER_MARGIN_PT or cy > page_h - BORDER_MARGIN_PT):
            continue

        # Exclude boxes inside FCF frames (datum references)
        if _inside_fcf(cx, cy, fcf_zones):
            continue

        # Verify interior ink (a letter glyph must be present)
        ink = _interior_ink(page_gray, bx0, by0, bx1, by1, zoom, pw, ph)
        if ink < MIN_INK_RATIO:
            continue

        definitions.append(DatumDefinition(
            x=cx,
            y=cy,
            width=bx1 - bx0,
            height=by1 - by0,
            ink_ratio=ink,
        ))

    return sorted(definitions, key=lambda d: (d.y, d.x))


def _inside_fcf(
    cx: float, cy: float,
    zones: Sequence[Tuple[float, float, float, float]],
    margin: float = 4.0,
) -> bool:
    for x0, y0, x1, y1 in zones:
        if x0 - margin <= cx <= x1 + margin and y0 - margin <= cy <= y1 + margin:
            return True
    return False


def _interior_ink(
    page_gray: np.ndarray,
    bx0: float, by0: float, bx1: float, by1: float,
    zoom: float, pw: int, ph: int,
) -> float:
    """Compute ink ratio inside a box interior (stripping the border)."""
    px0 = max(0, int(bx0 * zoom) + 2)
    py0 = max(0, int(by0 * zoom) + 2)
    px1 = min(pw, int(bx1 * zoom) - 2)
    py1 = min(ph, int(by1 * zoom) - 2)
    if px1 <= px0 or py1 <= py0:
        return 0.0
    interior = page_gray[py0:py1, px0:px1]
    _, bw = cv2.threshold(interior, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return float(bw.sum()) / (255.0 * bw.size) if bw.size > 0 else 0.0


__all__ = [
    "DatumDefinition",
    "find_datum_definitions",
]
