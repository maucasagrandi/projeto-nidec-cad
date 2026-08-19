"""Datum cell content extraction from rendered FCF frames.

This module crops datum cells from the rendered page and identifies their
content (typically single uppercase letters A-Z used as datum references).

Strategy:
1. Render the page at high DPI (300).
2. Crop each datum cell region.
3. Strip the cell border (left/top/right/bottom lines).
4. Isolate the character glyph in the interior.
5. Use contour analysis and simple heuristics for character identification,
   or return the crop for downstream OCR.

For now, this module extracts the crops and provides basic analysis. Full OCR
integration can be added later with Tesseract or a trained model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.gdt.fcf_expander import FcfCell, FcfFrame
from src.gdt.datum_text import DatumTextCandidate, match_text_candidate_to_bbox


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class DatumRef:
    """A datum reference extracted from an FCF cell."""

    cell: FcfCell
    crop: Optional[np.ndarray] = None  # grayscale crop of the cell
    glyph: Optional[np.ndarray] = None  # isolated character glyph
    text: str = ""  # recognized character (empty if unknown)
    confidence: float = 0.0
    ink_ratio: float = 0.0  # fraction of dark pixels
    text_source: str = ""
    text_bbox: Optional[Tuple[float, float, float, float]] = None
    visual_match_margin: Optional[float] = None
    visual_template_source: str = ""

    @property
    def has_content(self) -> bool:
        return bool(self.text) or self.ink_ratio > 0.05

    def to_dict(self) -> dict:
        return {
            "cell_index": self.cell.index,
            "bbox_pt": [round(self.cell.x0, 1), round(self.cell.y0, 1),
                        round(self.cell.x1, 1), round(self.cell.y1, 1)],
            "text": self.text,
            "confidence": round(self.confidence, 3),
            "text_source": self.text_source or None,
            "text_bbox_pt": (
                [round(value, 4) for value in self.text_bbox]
                if self.text_bbox is not None else None
            ),
            "visual_match_margin": (
                round(self.visual_match_margin, 4)
                if self.visual_match_margin is not None else None
            ),
            "visual_template_source": self.visual_template_source or None,
            "ink_ratio": round(self.ink_ratio, 3),
            "has_content": self.has_content,
        }


@dataclass
class FcfExtraction:
    """Complete extraction result for one FCF frame."""

    frame: FcfFrame
    datum_refs: List[DatumRef] = field(default_factory=list)
    tolerance_crop: Optional[np.ndarray] = None

    @property
    def has_datums(self) -> bool:
        return any(d.has_content for d in self.datum_refs)

    @property
    def datum_texts(self) -> List[str]:
        return [d.text for d in self.datum_refs if d.text and d.has_content]

    def to_dict(self) -> dict:
        return {
            **self.frame.to_dict(),
            "datum_refs": [d.to_dict() for d in self.datum_refs],
            "datum_texts": self.datum_texts,
            "has_datums": self.has_datums,
        }


# ---------------------------------------------------------------------------
# Cell content extraction
# ---------------------------------------------------------------------------


def _strip_border(crop: np.ndarray, border_px: int = 3) -> np.ndarray:
    """Remove the cell border lines from a crop.

    The border is typically 1-2px wide on the left (and sometimes top/bottom).
    We strip a fixed margin to get just the interior content.
    """
    h, w = crop.shape[:2]
    if h <= border_px * 2 or w <= border_px * 2:
        return crop

    interior = crop[border_px:h - border_px, border_px:w - border_px]
    return interior


def _isolate_glyph(interior: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
    """Isolate the character glyph from the cell interior.

    Returns (glyph_crop, ink_ratio).
    """
    if interior.size == 0:
        return None, 0.0

    # Binarize
    _, bw = cv2.threshold(interior, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink_ratio = float(bw.sum()) / (255.0 * bw.size) if bw.size > 0 else 0.0

    if ink_ratio < 0.02:
        return None, ink_ratio

    # Find bounding box of the content
    coords = cv2.findNonZero(bw)
    if coords is None:
        return None, ink_ratio

    x, y, w, h = cv2.boundingRect(coords)
    glyph = interior[y:y + h, x:x + w]
    return glyph, ink_ratio


def _identify_character(glyph: np.ndarray) -> Tuple[str, float]:
    """Report whether a glyph is present.

    NOTE: reliable OCR of the datum letter is out of scope here. Fabricating a
    guess (as an earlier version did) produced misleading data in the report,
    so we intentionally do NOT invent a letter. Presence of a glyph is already
    captured by ``ink_ratio`` on the DatumRef. Integrate a real OCR engine to
    populate the actual letter.
    """
    return "", 0.0


# ---------------------------------------------------------------------------
# Main extraction API
# ---------------------------------------------------------------------------


def extract_datum_cells(
    page_gray: np.ndarray,
    zoom: float,
    frames: List[FcfFrame],
    *,
    border_px: int = 3,
    text_candidates: Optional[List[DatumTextCandidate]] = None,
) -> List[FcfExtraction]:
    """Extract datum cell content from rendered page.

    Parameters
    ----------
    page_gray : np.ndarray
        Page rendered as grayscale at the given DPI.
    zoom : float
        Zoom factor (dpi / 72).
    frames : list of FcfFrame
        Expanded FCF frames with cells.
    border_px : int
        Pixels to strip from cell borders.
    text_candidates : list of DatumTextCandidate, optional
        Vector/invisible single-letter PDF text in page coordinates.  When a
        candidate belongs to a datum cell it provides the actual A/B/C label;
        raster ink remains a fallback for presence-only detection.

    Returns
    -------
    List of FcfExtraction objects with datum crops and identification.
    """
    ph, pw = page_gray.shape[:2]
    results: List[FcfExtraction] = []

    for frame in frames:
        extraction = FcfExtraction(frame=frame)

        for cell in frame.datum_cells:
            # Convert PDF points to pixels
            px0 = max(0, int(cell.x0 * zoom))
            py0 = max(0, int(cell.y0 * zoom))
            px1 = min(pw, int(cell.x1 * zoom))
            py1 = min(ph, int(cell.y1 * zoom))

            if px1 <= px0 or py1 <= py0:
                continue

            crop = page_gray[py0:py1, px0:px1].copy()
            interior = _strip_border(crop, border_px=border_px)
            glyph, ink_ratio = _isolate_glyph(interior)

            text = ""
            confidence = 0.0
            text_source = ""
            text_bbox = None
            if text_candidates:
                matched = match_text_candidate_to_bbox(
                    text_candidates,
                    (cell.x0, cell.y0, cell.x1, cell.y1),
                )
                if matched is not None:
                    text = matched.label
                    confidence = matched.confidence
                    text_source = matched.source
                    text_bbox = matched.bbox

            if not text and glyph is not None and ink_ratio > 0.05:
                text, confidence = _identify_character(glyph)

            datum_ref = DatumRef(
                cell=cell,
                crop=crop,
                glyph=glyph,
                text=text,
                confidence=confidence,
                ink_ratio=ink_ratio,
                text_source=text_source,
                text_bbox=text_bbox,
            )
            extraction.datum_refs.append(datum_ref)

        # Also extract tolerance cells as a single crop for reference
        tol_cells = [c for c in frame.cells if c.role == "tolerance"]
        if tol_cells:
            tol_x0 = int(tol_cells[0].x0 * zoom)
            tol_y0 = int(tol_cells[0].y0 * zoom)
            tol_x1 = int(tol_cells[-1].x1 * zoom)
            tol_y1 = int(tol_cells[-1].y1 * zoom)
            tol_x0 = max(0, tol_x0)
            tol_y0 = max(0, tol_y0)
            tol_x1 = min(pw, tol_x1)
            tol_y1 = min(ph, tol_y1)
            if tol_x1 > tol_x0 and tol_y1 > tol_y0:
                extraction.tolerance_crop = page_gray[tol_y0:tol_y1, tol_x0:tol_x1].copy()

        results.append(extraction)

    return results


def render_datum_contact_sheet(
    extractions: List[FcfExtraction],
    *,
    cell_height: int = 48,
) -> Optional[np.ndarray]:
    """Render a contact sheet of all datum cell crops for visual inspection.

    Returns a grayscale image or None if no datums found.
    """
    cells_with_content = []
    for ext in extractions:
        for datum in ext.datum_refs:
            if datum.crop is not None and datum.ink_ratio > 0.03:
                cells_with_content.append((ext.frame.class_name, datum))

    if not cells_with_content:
        return None

    # Normalize all to same height
    resized: List[np.ndarray] = []
    for class_name, datum in cells_with_content:
        crop = datum.crop
        scale = cell_height / crop.shape[0]
        new_w = max(1, int(crop.shape[1] * scale))
        r = cv2.resize(crop, (new_w, cell_height), interpolation=cv2.INTER_AREA)
        # Add a 1px separator
        sep = np.full((cell_height, 2), 200, dtype=np.uint8)
        resized.append(np.hstack([r, sep]))

    if not resized:
        return None

    # Single row
    sheet = np.hstack(resized)
    return sheet


__all__ = [
    "DatumRef",
    "FcfExtraction",
    "extract_datum_cells",
    "render_datum_contact_sheet",
]
