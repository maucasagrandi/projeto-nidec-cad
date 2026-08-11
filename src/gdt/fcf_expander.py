"""FCF (Feature Control Frame) box expander.

Given a detected GD&T symbol location, this module expands the detection to
find the full constraint box by tracing vector line primitives in the PDF.

Strategy:
1. Extract all horizontal and vertical line segments from the PDF page.
2. Starting from the detected symbol bbox, find the horizontal lines that
   form the top and bottom boundaries of the FCF row.
3. Find vertical lines that span the full height between top/bottom — these
   are cell dividers.
4. Return the full FCF frame and its cells.

The FCF structure is: [symbol] | [tolerance] | [modifier?] | [datum ref(s)]
- Height is consistently ~14pt.
- Datum reference cells are ~10-12pt wide and appear at the right end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import fitz


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class FcfCell:
    """A single cell in the feature control frame."""

    x0: float  # left edge in PDF points
    y0: float  # top edge
    x1: float  # right edge
    y1: float  # bottom edge
    index: int = 0  # cell position (0 = symbol cell)
    role: str = ""  # "symbol", "tolerance", "modifier", "datum"

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "role": self.role,
            "bbox_pt": [round(self.x0, 1), round(self.y0, 1),
                        round(self.x1, 1), round(self.y1, 1)],
            "width_pt": round(self.width, 1),
        }


@dataclass
class FcfFrame:
    """Complete feature control frame extracted from a detection."""

    x0: float
    y0: float
    x1: float
    y1: float
    cells: List[FcfCell] = field(default_factory=list)
    class_name: str = ""
    detection_score: float = 0.0

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def cell_count(self) -> int:
        return len(self.cells)

    @property
    def datum_cells(self) -> List[FcfCell]:
        return [c for c in self.cells if c.role == "datum"]

    def to_dict(self) -> dict:
        return {
            "class_name": self.class_name,
            "detection_score": round(self.detection_score, 4),
            "frame_bbox_pt": [round(self.x0, 1), round(self.y0, 1),
                              round(self.x1, 1), round(self.y1, 1)],
            "width_pt": round(self.width, 1),
            "height_pt": round(self.height, 1),
            "cell_count": self.cell_count,
            "cells": [c.to_dict() for c in self.cells],
        }


# ---------------------------------------------------------------------------
# Line extraction from PDF
# ---------------------------------------------------------------------------


def extract_page_lines(
    page: fitz.Page,
) -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float, float]]]:
    """Extract horizontal and vertical line segments from a PDF page.

    Returns:
        (h_lines, v_lines) where:
        - h_lines: list of (x0, y, x1) for horizontal segments
        - v_lines: list of (x, y0, y1) for vertical segments
    """
    paths = page.get_drawings()

    h_lines: List[Tuple[float, float, float]] = []
    v_lines: List[Tuple[float, float, float]] = []

    for path in paths:
        for item in path.get("items", []):
            if item[0] == "l":
                p1, p2 = item[1], item[2]
                dx = abs(p2.x - p1.x)
                dy = abs(p2.y - p1.y)
                if dy < 0.5 and dx > 3.0:  # horizontal
                    h_lines.append((min(p1.x, p2.x), round(p1.y, 1), max(p1.x, p2.x)))
                elif dx < 0.5 and dy > 3.0:  # vertical
                    v_lines.append((round(p1.x, 1), min(p1.y, p2.y), max(p1.y, p2.y)))
            elif item[0] == "re":
                rect = item[1]
                # A rectangle contributes 4 lines
                x0, y0, x1, y1 = rect.x0, rect.y0, rect.x1, rect.y1
                if x1 - x0 > 3.0:
                    h_lines.append((x0, round(y0, 1), x1))
                    h_lines.append((x0, round(y1, 1), x1))
                if y1 - y0 > 3.0:
                    v_lines.append((round(x0, 1), y0, y1))
                    v_lines.append((round(x1, 1), y0, y1))

    return h_lines, v_lines


# ---------------------------------------------------------------------------
# FCF expansion logic
# ---------------------------------------------------------------------------


# Expected FCF cell height range (in points)
MIN_FCF_HEIGHT = 8.0
MAX_FCF_HEIGHT = 22.0
# Minimum cell width to be considered a real cell
MIN_CELL_WIDTH = 5.0
# Maximum total FCF width (to prevent runaway expansion)
MAX_FCF_WIDTH = 250.0
# Datum cells are narrow (10-14pt typically)
MAX_DATUM_CELL_WIDTH = 18.0
# Tolerance for line alignment
LINE_TOLERANCE = 2.0


def expand_fcf_from_detection(
    symbol_x: float,
    symbol_y: float,
    symbol_w: float,
    symbol_h: float,
    h_lines: List[Tuple[float, float, float]],
    v_lines: List[Tuple[float, float, float]],
    *,
    class_name: str = "",
    detection_score: float = 0.0,
) -> Optional[FcfFrame]:
    """Expand a detected symbol into a full FCF using vector primitives.

    Parameters
    ----------
    symbol_x, symbol_y : float
        Top-left corner of the detected symbol in PDF points.
    symbol_w, symbol_h : float
        Width and height of the detected symbol bbox.
    h_lines : list
        Horizontal lines as (x0, y, x1).
    v_lines : list
        Vertical lines as (x, y0, y1).
    class_name : str
        GD&T class name from detection.
    detection_score : float
        Template matching score.

    Returns
    -------
    FcfFrame or None if expansion fails.
    """
    sym_cx = symbol_x + symbol_w / 2
    sym_cy = symbol_y + symbol_h / 2
    top_y = symbol_y
    bottom_y = symbol_y + symbol_h

    # Step 1: Find the top horizontal line of the FCF
    # Must be near top_y and must start at or before the symbol left edge
    top_candidates = [
        (x0, y, x1) for x0, y, x1 in h_lines
        if abs(y - top_y) < LINE_TOLERANCE
        and x0 <= symbol_x + LINE_TOLERANCE
        and x1 >= symbol_x + symbol_w - LINE_TOLERANCE  # extends past symbol
        and (x1 - x0) < MAX_FCF_WIDTH
    ]

    if not top_candidates:
        # Try slightly wider tolerance
        top_candidates = [
            (x0, y, x1) for x0, y, x1 in h_lines
            if abs(y - top_y) < LINE_TOLERANCE * 2
            and x0 <= symbol_x + LINE_TOLERANCE * 2
            and x1 >= sym_cx  # at least past center
            and (x1 - x0) < MAX_FCF_WIDTH
        ]

    if not top_candidates:
        return None

    # Pick the line that starts closest to the symbol and extends furthest right
    top_line = max(top_candidates, key=lambda l: l[2] - l[0])
    frame_top = top_line[1]

    # Step 2: Find the bottom horizontal line
    expected_bottom = frame_top + symbol_h
    bottom_candidates = [
        (x0, y, x1) for x0, y, x1 in h_lines
        if abs(y - expected_bottom) < LINE_TOLERANCE * 2
        and x0 <= symbol_x + LINE_TOLERANCE
        and x1 >= sym_cx
        and (x1 - x0) < MAX_FCF_WIDTH
    ]

    if not bottom_candidates:
        # Try finding any horizontal line that forms a valid FCF height
        bottom_candidates = [
            (x0, y, x1) for x0, y, x1 in h_lines
            if MIN_FCF_HEIGHT <= (y - frame_top) <= MAX_FCF_HEIGHT
            and x0 <= symbol_x + LINE_TOLERANCE
            and x1 >= sym_cx
            and (x1 - x0) < MAX_FCF_WIDTH
        ]

    if not bottom_candidates:
        return None

    # Pick bottom line closest to expected height
    bottom_line = min(bottom_candidates, key=lambda l: abs(l[1] - expected_bottom))
    frame_bottom = bottom_line[1]
    frame_height = frame_bottom - frame_top

    if not (MIN_FCF_HEIGHT <= frame_height <= MAX_FCF_HEIGHT):
        return None

    # Step 3: Determine frame left and right edges
    frame_left = min(top_line[0], bottom_line[0])
    frame_right = max(top_line[2], bottom_line[2])

    # Constrain: frame shouldn't start way before the symbol
    if symbol_x - frame_left > symbol_w * 2:
        frame_left = symbol_x - LINE_TOLERANCE

    # Constrain max width
    if frame_right - frame_left > MAX_FCF_WIDTH:
        frame_right = frame_left + MAX_FCF_WIDTH

    # Step 4: Find vertical lines (cell dividers) within the frame
    dividers: List[float] = []
    for x, y0, y1 in v_lines:
        if frame_left - LINE_TOLERANCE <= x <= frame_right + LINE_TOLERANCE:
            # Must span most of the frame height
            span = min(y1, frame_bottom) - max(y0, frame_top)
            if span >= frame_height * 0.7:
                dividers.append(x)

    # Add frame edges as boundaries if not already there
    if not dividers or min(dividers) > frame_left + LINE_TOLERANCE:
        dividers.append(frame_left)
    if not dividers or max(dividers) < frame_right - LINE_TOLERANCE:
        dividers.append(frame_right)

    # Deduplicate close dividers
    dividers = sorted(set(dividers))
    merged: List[float] = []
    for x in dividers:
        if not merged or x - merged[-1] >= MIN_CELL_WIDTH:
            merged.append(x)
        else:
            # Keep the one closer to a round number or just the existing one
            pass
    dividers = merged

    if len(dividers) < 2:
        return None

    # Step 5: Build cells
    cells: List[FcfCell] = []
    for i in range(len(dividers) - 1):
        cell_width = dividers[i + 1] - dividers[i]
        if cell_width < MIN_CELL_WIDTH:
            continue
        cells.append(FcfCell(
            x0=dividers[i],
            y0=frame_top,
            x1=dividers[i + 1],
            y1=frame_bottom,
            index=len(cells),
        ))

    if not cells:
        return None

    # Step 6: Assign cell roles
    _assign_cell_roles(cells)

    # Update frame bounds to actual cell extent
    actual_left = cells[0].x0
    actual_right = cells[-1].x1

    return FcfFrame(
        x0=actual_left,
        y0=frame_top,
        x1=actual_right,
        y1=frame_bottom,
        cells=cells,
        class_name=class_name,
        detection_score=detection_score,
    )


def _assign_cell_roles(cells: List[FcfCell]) -> None:
    """Assign roles to cells based on position and width.

    Pattern: [symbol] [tolerance...] [datum refs...]
    - First cell is always the symbol.
    - Last cells that are narrow (≤18pt) are datum references.
    - Middle cells are tolerance/modifier content.
    """
    if not cells:
        return

    cells[0].role = "symbol"

    # Find datum cells from the right: narrow cells at the end
    datum_start = len(cells)
    for i in range(len(cells) - 1, 0, -1):
        if cells[i].width <= MAX_DATUM_CELL_WIDTH:
            datum_start = i
        else:
            break

    # Assign roles
    for i in range(1, len(cells)):
        if i >= datum_start:
            cells[i].role = "datum"
        else:
            cells[i].role = "tolerance"


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------


def expand_detections_to_fcf(
    pdf_bytes: bytes,
    detections: list,
    *,
    page_index: int = 0,
) -> List[FcfFrame]:
    """Expand a list of template detections into full FCF frames.

    Parameters
    ----------
    pdf_bytes : bytes
        PDF file content.
    detections : list
        List of Detection objects from template_detector.
    page_index : int
        Page to analyze.

    Returns
    -------
    List of FcfFrame objects.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        page = doc[page_index]
        h_lines, v_lines = extract_page_lines(page)
    finally:
        doc.close()

    frames: List[FcfFrame] = []
    for det in detections:
        frame = expand_fcf_from_detection(
            symbol_x=det.x,
            symbol_y=det.y,
            symbol_w=det.width,
            symbol_h=det.height,
            h_lines=h_lines,
            v_lines=v_lines,
            class_name=det.class_name,
            detection_score=det.score,
        )
        if frame is not None:
            frames.append(frame)

    return frames


__all__ = [
    "FcfCell",
    "FcfFrame",
    "expand_detections_to_fcf",
    "expand_fcf_from_detection",
    "extract_page_lines",
]
