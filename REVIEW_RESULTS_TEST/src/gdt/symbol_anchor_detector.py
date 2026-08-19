"""Symbol-first GD&T detector for urgent multi-CAD hardening.

This detector inverts the V1/V2 search order:

1. Find small closed cell-like rectangles in a rasterized page.
2. Treat each cell only as an *anchor proposal*.
3. Rank the content of that cell against the existing GD&T symbol templates.
4. Keep only cells with credible symbol evidence.
5. Grow a feature-control-frame to the right from the accepted symbol cell by
   following adjacent cells with approximately the same vertical band.

The intent is to avoid the failure mode where every small rectangle/table cell
becomes a GD&T proposal.  Geometry alone no longer opens the downstream path;
a GD&T-like first cell must anchor the frame.

All thresholds here are engineering heuristics for validation. They are not ISO
requirements and must not be described as calibrated production thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import cv2
import fitz
import numpy as np

from src.gdt.detector import BBox, GdtCell, GdtFrameCandidate
from src.gdt.symbol_classifier import load_template_catalog, render_page_gray, score_candidates


@dataclass
class SymbolAnchorEvidence:
    anchor_id: str
    page: int
    bbox: BBox
    best_class: str | None
    best_score: float | None
    second_best_class: str | None
    second_best_score: float | None
    margin: float | None
    negative_control_score: float | None
    accepted: bool
    rejection_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "anchor_id": self.anchor_id,
            "page": self.page,
            "bbox": self.bbox.to_list(),
            "best_class": self.best_class,
            "best_score": self.best_score,
            "second_best_class": self.second_best_class,
            "second_best_score": self.second_best_score,
            "margin": self.margin,
            "negative_control_score": self.negative_control_score,
            "accepted": self.accepted,
            "rejection_reasons": list(self.rejection_reasons),
        }


def _iou(a: BBox, b: BBox) -> float:
    x0, y0 = max(a.x0, b.x0), max(a.y0, b.y0)
    x1, y1 = min(a.x1, b.x1), min(a.y1, b.y1)
    iw, ih = max(0.0, x1 - x0), max(0.0, y1 - y0)
    inter = iw * ih
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def _dedup_boxes(boxes: Iterable[BBox], iou_threshold: float = 0.75) -> list[BBox]:
    ranked = sorted(boxes, key=lambda b: b.area)
    kept: list[BBox] = []
    for box in ranked:
        if any(_iou(box, existing) >= iou_threshold for existing in kept):
            continue
        kept.append(box)
    return kept


def _cell_rectangles_from_raster(
    pdf_bytes: bytes,
    *,
    page_index: int,
    dpi: int = 260,
    min_height_pt: float = 4.0,
    max_height_pt: float = 24.0,
    min_width_pt: float = 4.0,
    max_width_pt: float = 34.0,
) -> tuple[list[BBox], dict]:
    """Find small closed rectangular cells, not whole FCF rows.

    Unlike the old raster proposal generator, this does not build arbitrary
    2..6-cell windows across the page.  It only finds individual compact cells
    that can later be tested for GD&T symbol evidence.
    """

    scale = dpi / 72.0
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        page = doc[page_index]
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=fitz.csGRAY, alpha=False)
        gray = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)

    binary = cv2.threshold(gray, 205, 255, cv2.THRESH_BINARY_INV)[1]
    # Close tiny export gaps but do not join neighboring table cells globally.
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    boxes: list[BBox] = []
    hmin, hmax = min_height_pt * scale, max_height_pt * scale
    wmin, wmax = min_width_pt * scale, max_width_pt * scale
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if not (hmin <= h <= hmax and wmin <= w <= wmax):
            continue
        if h <= 0 or w <= 0:
            continue
        aspect = w / h
        if not (0.30 <= aspect <= 3.5):
            continue

        # A symbol cell should be a genuinely enclosed small box. Estimate border
        # coverage on the four sides instead of accepting every rectangular crop.
        pad = max(1, int(round(scale * 0.5)))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(closed.shape[1], x + w + pad), min(closed.shape[0], y + h + pad)
        roi = closed[y0:y1, x0:x1]
        if roi.size == 0:
            continue
        border = max(1, int(round(scale * 0.7)))
        top = float((roi[:border, :] > 0).mean())
        bottom = float((roi[-border:, :] > 0).mean())
        left = float((roi[:, :border] > 0).mean())
        right = float((roi[:, -border:] > 0).mean())
        if min(top, bottom, left, right) < 0.18:
            continue

        boxes.append(BBox(x / scale, y / scale, (x + w) / scale, (y + h) / scale))

    boxes = _dedup_boxes(boxes)
    return boxes, {
        "raster_dpi": dpi,
        "raw_contours": len(contours),
        "cell_rectangles": len(boxes),
        "semantics": "cell anchors only; not FCF candidates",
    }


def _candidate_for_single_cell(box: BBox, *, page: int, idx: int) -> GdtFrameCandidate:
    cell = GdtCell(bbox=box)
    return GdtFrameCandidate(
        candidate_id=f"GDT-ANCHOR-P{page:02d}-{idx:04d}",
        page=page,
        frame_bbox=box,
        symbol_bbox=box,
        cells=[cell],
        confidence_score=0.0,
    )


def _score_anchor_cells(
    pdf_bytes: bytes,
    *,
    page_index: int,
    boxes: Sequence[BBox],
    templates: Sequence[Any],
    symbol_dpi: int = 300,
    min_score: float = 0.46,
    min_margin: float = 0.025,
    negative_margin: float = 0.035,
) -> tuple[list[SymbolAnchorEvidence], list[GdtFrameCandidate]]:
    """Rank potential first cells and keep credible GD&T symbol anchors.

    Thresholds are intentionally validation heuristics.  Acceptance also checks
    the negative-control class when available, which is critical for suppressing
    letters/table fragments that happen to correlate with a simple line symbol.
    """

    page_number = page_index + 1
    candidates = [_candidate_for_single_cell(box, page=page_number, idx=i + 1) for i, box in enumerate(boxes)]
    if not candidates or not templates:
        return [], []

    page_gray, zoom = render_page_gray(pdf_bytes, page_index=page_index, dpi=symbol_dpi)
    scored = score_candidates(candidates, page_gray, zoom, templates)

    evidence: list[SymbolAnchorEvidence] = []
    accepted: list[GdtFrameCandidate] = []
    for candidate, (score, _crop) in zip(candidates, scored):
        payload = score.to_dict()
        classes = payload.get("class_scores") or {}
        best_class = payload.get("best_class")
        best_score = float(payload.get("best_score", -1.0))
        second_class = payload.get("second_best_class")
        second_score = float(payload.get("second_best_score", -1.0))
        margin = float(payload.get("margin", best_score - second_score))
        negative = classes.get("negative_controls")
        negative_score = float(negative) if negative is not None else None

        reasons: list[str] = []
        if best_class in {None, "negative_controls"}:
            reasons.append("NEGATIVE_CONTROL_OR_NO_CLASS")
        if best_score < min_score:
            reasons.append("SYMBOL_SCORE_BELOW_VALIDATION_HEURISTIC")
        if margin < min_margin:
            reasons.append("SYMBOL_MARGIN_TOO_SMALL")
        if negative_score is not None and best_score - negative_score < negative_margin:
            reasons.append("NOT_SEPARATED_FROM_NEGATIVE_CONTROL")

        ok = not reasons
        evidence.append(
            SymbolAnchorEvidence(
                anchor_id=candidate.candidate_id,
                page=page_number,
                bbox=candidate.symbol_bbox,
                best_class=best_class,
                best_score=best_score,
                second_best_class=second_class,
                second_best_score=second_score,
                margin=margin,
                negative_control_score=negative_score,
                accepted=ok,
                rejection_reasons=reasons,
            )
        )
        if ok:
            candidate.confidence_score = best_score
            accepted.append(candidate)

    return evidence, accepted


def _all_small_cell_boxes(
    pdf_bytes: bytes,
    *,
    page_index: int,
    dpi: int,
) -> list[BBox]:
    boxes, _ = _cell_rectangles_from_raster(pdf_bytes, page_index=page_index, dpi=dpi)
    return boxes


def _same_row(anchor: BBox, other: BBox) -> bool:
    ah = anchor.height
    oh = other.height
    if ah <= 0 or oh <= 0:
        return False
    cy_a = (anchor.y0 + anchor.y1) / 2.0
    cy_o = (other.y0 + other.y1) / 2.0
    return (
        abs(cy_a - cy_o) <= 0.28 * max(ah, oh)
        and abs(ah - oh) <= 0.35 * max(ah, oh)
    )


def _grow_frame_from_anchor(anchor: GdtFrameCandidate, cells: Sequence[BBox], *, max_cells: int = 6) -> list[BBox]:
    """Grow rightward from a symbol anchor through genuinely adjacent cells."""

    row = [b for b in cells if _same_row(anchor.symbol_bbox, b) and b.x0 >= anchor.symbol_bbox.x0 - 0.8]
    row.sort(key=lambda b: b.x0)

    # Start from the box that best overlaps the accepted anchor.
    start_idx = None
    for idx, box in enumerate(row):
        if _iou(box, anchor.symbol_bbox) >= 0.45:
            start_idx = idx
            break
    if start_idx is None:
        return [anchor.symbol_bbox]

    chain = [row[start_idx]]
    height = anchor.symbol_bbox.height
    max_gap = max(1.5, 0.22 * height)
    for box in row[start_idx + 1 :]:
        gap = box.x0 - chain[-1].x1
        if gap < -0.35 * height:
            continue
        if gap > max_gap:
            break
        # Reject nested/overlapping contour artifacts instead of treating them as
        # extra FCF cells.
        if _iou(box, chain[-1]) > 0.25:
            continue
        chain.append(box)
        if len(chain) >= max_cells:
            break
    return chain


def detect_symbol_anchored_frames(
    pdf_bytes: bytes,
    *,
    page_index: int = 0,
    template_root: str,
    raster_dpi: int = 260,
    symbol_dpi: int = 300,
) -> tuple[list[GdtFrameCandidate], dict]:
    """Detect FCFs by locating the GD&T symbol cell first.

    Returns downstream candidates plus fully auditable anchor evidence.
    """

    templates = load_template_catalog(template_root)
    boxes, raster_audit = _cell_rectangles_from_raster(
        pdf_bytes,
        page_index=page_index,
        dpi=raster_dpi,
    )
    anchor_evidence, accepted_anchors = _score_anchor_cells(
        pdf_bytes,
        page_index=page_index,
        boxes=boxes,
        templates=templates,
        symbol_dpi=symbol_dpi,
    )

    all_cells = boxes
    page_number = page_index + 1
    frames: list[GdtFrameCandidate] = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        page = doc[page_index]
        words = page.get_text("words")

    for idx, anchor in enumerate(accepted_anchors, start=1):
        chain = _grow_frame_from_anchor(anchor, all_cells)
        if len(chain) < 2:
            # A real FCF requires at least the symbol + one downstream cell.
            continue
        cells: list[GdtCell] = []
        for box in chain:
            texts: list[str] = []
            for word in words:
                wx = (word[0] + word[2]) / 2.0
                wy = (word[1] + word[3]) / 2.0
                if box.contains_point(wx, wy) and str(word[4]).strip():
                    texts.append(str(word[4]).strip())
            cells.append(GdtCell(bbox=box, texts=texts))

        frame_bbox = BBox(
            min(b.x0 for b in chain),
            min(b.y0 for b in chain),
            max(b.x1 for b in chain),
            max(b.y1 for b in chain),
        )
        frames.append(
            GdtFrameCandidate(
                candidate_id=f"GDT-SYMBOL-P{page_number:02d}-{idx:03d}",
                page=page_number,
                frame_bbox=frame_bbox,
                symbol_bbox=chain[0],
                cells=cells,
                confidence_score=anchor.confidence_score,
            )
        )

    # Deduplicate multiple anchor contours around the same real FCF.
    dedup: list[GdtFrameCandidate] = []
    for candidate in sorted(frames, key=lambda c: c.confidence_score, reverse=True):
        if any(_iou(candidate.frame_bbox, existing.frame_bbox) >= 0.55 for existing in dedup):
            continue
        dedup.append(candidate)
    dedup.sort(key=lambda c: (c.page, c.frame_bbox.y0, c.frame_bbox.x0))

    return dedup, {
        "strategy": "symbol_first_anchor_then_frame_growth",
        "validation_status": "DIAGNOSTIC_ONLY",
        "thresholds_calibrated": False,
        "raster_audit": raster_audit,
        "anchor_count": len(anchor_evidence),
        "accepted_anchor_count": sum(1 for row in anchor_evidence if row.accepted),
        "frame_count": len(dedup),
        "anchors": [row.to_dict() for row in anchor_evidence],
    }


__all__ = ["detect_symbol_anchored_frames", "SymbolAnchorEvidence"]
