"""Deterministic datum-letter extraction in PDF page coordinates.

The geometry pipeline decides whether a region is an FCF datum cell or a
datum-feature indicator.  This module supplies the missing identity: the
single uppercase letter contained by that region.

Two PDF text sources are combined:

* PyMuPDF ``page.get_text("words")`` for normal/selectable text;
* the low-level content-stream parser for invisible CAD text (``3 Tr``).

Candidates are normalized to PyMuPDF page coordinates and deduplicated, so
callers can match them to geometric boxes without OCR or an LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Optional, Sequence

import fitz

from src.gdt.pdf_hidden_text_geometry import extract_page_text_geometry_events


_SINGLE_DATUM_RE = re.compile(r"^[A-Z]$")
BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class DatumTextCandidate:
    label: str
    page: int
    bbox: BBox
    source: str
    confidence: float
    invisible: bool = False
    bbox_quality: str = "native"

    @property
    def center(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.bbox
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "page": self.page,
            "bbox_pt": [round(value, 4) for value in self.bbox],
            "source": self.source,
            "confidence": round(self.confidence, 3),
            "invisible": self.invisible,
            "bbox_quality": self.bbox_quality,
        }


def _normalize_label(value: object) -> Optional[str]:
    label = str(value or "").strip().upper()
    return label if _SINGLE_DATUM_RE.fullmatch(label) else None


def _intersection_ratio(inner: BBox, outer: BBox) -> float:
    ix0 = max(inner[0], outer[0])
    iy0 = max(inner[1], outer[1])
    ix1 = min(inner[2], outer[2])
    iy1 = min(inner[3], outer[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    intersection = (ix1 - ix0) * (iy1 - iy0)
    area = max(0.0, inner[2] - inner[0]) * max(0.0, inner[3] - inner[1])
    return intersection / area if area > 0 else 0.0


def _same_candidate(a: DatumTextCandidate, b: DatumTextCandidate) -> bool:
    if a.label != b.label or a.page != b.page:
        return False
    acx, acy = a.center
    bcx, bcy = b.center
    if abs(acx - bcx) <= 2.0 and abs(acy - bcy) <= 2.0:
        return True
    return max(_intersection_ratio(a.bbox, b.bbox), _intersection_ratio(b.bbox, a.bbox)) >= 0.60


def _deduplicate(candidates: Iterable[DatumTextCandidate]) -> list[DatumTextCandidate]:
    kept: list[DatumTextCandidate] = []
    # Native word boxes are preferred over estimated hidden-text geometry.
    for candidate in sorted(candidates, key=lambda row: -row.confidence):
        if any(_same_candidate(candidate, existing) for existing in kept):
            continue
        kept.append(candidate)
    return sorted(kept, key=lambda row: (row.page, row.bbox[1], row.bbox[0], row.label))


def extract_datum_text_candidates(
    pdf_bytes: bytes,
    *,
    page_index: int = 0,
) -> list[DatumTextCandidate]:
    """Extract single uppercase PDF text with page-coordinate bounding boxes."""

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        if page_index < 0 or page_index >= len(doc):
            raise IndexError(f"page_index {page_index} outside PDF with {len(doc)} page(s)")
        page = doc[page_index]
        candidates: list[DatumTextCandidate] = []

        for word in page.get_text("words"):
            label = _normalize_label(word[4])
            if label is None:
                continue
            candidates.append(
                DatumTextCandidate(
                    label=label,
                    page=page_index + 1,
                    bbox=(float(word[0]), float(word[1]), float(word[2]), float(word[3])),
                    source="pymupdf_word",
                    confidence=1.0,
                    invisible=False,
                    bbox_quality="native_word_bbox",
                )
            )

        # CAD exports can store selectable text in an invisible 3 Tr layer that
        # is not reliably associated with words by the high-level extractor.
        geometry_events, _resolver = extract_page_text_geometry_events(doc, page_index=page_index)
        for event in geometry_events:
            label = _normalize_label(event.text)
            if label is None:
                continue
            confidence = 0.90 if event.bbox_quality == "font_metrics" else 0.75
            candidates.append(
                DatumTextCandidate(
                    label=label,
                    page=page_index + 1,
                    bbox=tuple(float(value) for value in event.page_bbox),
                    source="pdf_content_stream",
                    confidence=confidence,
                    invisible=event.invisible,
                    bbox_quality=event.bbox_quality,
                )
            )

        return _deduplicate(candidates)
    finally:
        doc.close()


def match_text_candidate_to_bbox(
    candidates: Sequence[DatumTextCandidate],
    bbox: BBox,
    *,
    padding_pt: float = 1.5,
    min_text_overlap: float = 0.50,
) -> Optional[DatumTextCandidate]:
    """Return the strongest single-letter candidate contained by ``bbox``.

    Center containment is the primary rule.  Text-overlap is a fallback for
    estimated baselines / bounding boxes near a cell edge.
    """

    x0, y0, x1, y1 = bbox
    expanded = (x0 - padding_pt, y0 - padding_pt, x1 + padding_pt, y1 + padding_pt)
    box_cx = (x0 + x1) / 2.0
    box_cy = (y0 + y1) / 2.0
    matches: list[tuple[int, float, float, DatumTextCandidate]] = []

    for candidate in candidates:
        cx, cy = candidate.center
        center_inside = expanded[0] <= cx <= expanded[2] and expanded[1] <= cy <= expanded[3]
        overlap = _intersection_ratio(candidate.bbox, expanded)
        if not center_inside and overlap < min_text_overlap:
            continue
        distance = (cx - box_cx) ** 2 + (cy - box_cy) ** 2
        matches.append((0 if center_inside else 1, -candidate.confidence, distance, candidate))

    if not matches:
        return None
    matches.sort(key=lambda row: (row[0], row[1], row[2], row[3].label))
    return matches[0][3]


__all__ = [
    "BBox",
    "DatumTextCandidate",
    "extract_datum_text_candidates",
    "match_text_candidate_to_bbox",
]
