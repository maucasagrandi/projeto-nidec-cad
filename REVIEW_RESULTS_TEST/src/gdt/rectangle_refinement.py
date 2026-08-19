"""Geometry-only refinement for GD&T frame proposals.

This module addresses one specific failure mode observed in multi-CAD validation:
raw proposal generation may combine unrelated horizontal/vertical strokes into a
rectangular window, especially inside title blocks, tables, notes, and section
geometry. The goal here is not to classify the GD&T symbol. It is to decide
whether the proposed box is a coherent *row of rectangular cells* before symbol
classification runs.

Important semantics:
- this is implementation geometry, not ISO tolerance;
- proposals are never called TP/FP without independent ground truth;
- the classifier is intentionally not used here;
- thresholds are validation heuristics and are recorded in the evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import cv2
import fitz
import numpy as np

from src.gdt.candidate_detector_v2 import RawProposal


@dataclass(frozen=True)
class RectangleGeometryEvidence:
    top_coverage: float
    bottom_coverage: float
    left_coverage: float
    right_coverage: float
    outer_mean: float
    outer_min: float
    divider_mean: float
    row_aspect: float
    row_aspect_score: float
    horizontal_extension_penalty: float
    vertical_continuation_fraction: float
    nearby_parallel_rows: int
    grid_penalty: float
    geometry_score: float

    def to_dict(self) -> dict:
        return {
            "top_coverage": round(self.top_coverage, 6),
            "bottom_coverage": round(self.bottom_coverage, 6),
            "left_coverage": round(self.left_coverage, 6),
            "right_coverage": round(self.right_coverage, 6),
            "outer_mean": round(self.outer_mean, 6),
            "outer_min": round(self.outer_min, 6),
            "divider_mean": round(self.divider_mean, 6),
            "row_aspect": round(self.row_aspect, 6),
            "row_aspect_score": round(self.row_aspect_score, 6),
            "horizontal_extension_penalty": round(self.horizontal_extension_penalty, 6),
            "vertical_continuation_fraction": round(self.vertical_continuation_fraction, 6),
            "nearby_parallel_rows": int(self.nearby_parallel_rows),
            "grid_penalty": round(self.grid_penalty, 6),
            "geometry_score": round(self.geometry_score, 6),
        }


def _render_line_masks(pdf_bytes: bytes, page_index: int, dpi: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Render one page and isolate horizontal/vertical ruling strokes.

    Morphological opening suppresses most text glyphs and keeps the line evidence
    needed to evaluate whether a proposed frame really has four borders and
    internal dividers. This works independently of how the CAD exporter encoded
    the vector paths.
    """

    scale = dpi / 72.0
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        page = doc[page_index]
        pix = page.get_pixmap(
            matrix=fitz.Matrix(scale, scale),
            colorspace=fitz.csGRAY,
            alpha=False,
        )
        gray = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)

    _threshold, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    h_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(7, int(round(scale * 2.2))), 1),
    )
    v_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, max(7, int(round(scale * 2.2)))),
    )
    h_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    v_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
    return h_mask, v_mask, scale


def _h_coverage(mask: np.ndarray, y: float, x0: float, x1: float, scale: float, *, tol_pt: float = 1.2) -> float:
    if x1 <= x0:
        return 0.0
    yy = int(round(y * scale))
    left = max(0, int(round(x0 * scale)))
    right = min(mask.shape[1], int(round(x1 * scale)))
    tol = max(1, int(round(tol_pt * scale)))
    top = max(0, yy - tol)
    bottom = min(mask.shape[0], yy + tol + 1)
    if right <= left or bottom <= top:
        return 0.0
    strip = mask[top:bottom, left:right]
    return float(np.mean(np.any(strip > 0, axis=0))) if strip.size else 0.0


def _v_coverage(mask: np.ndarray, x: float, y0: float, y1: float, scale: float, *, tol_pt: float = 1.2) -> float:
    if y1 <= y0:
        return 0.0
    xx = int(round(x * scale))
    top = max(0, int(round(y0 * scale)))
    bottom = min(mask.shape[0], int(round(y1 * scale)))
    tol = max(1, int(round(tol_pt * scale)))
    left = max(0, xx - tol)
    right = min(mask.shape[1], xx + tol + 1)
    if bottom <= top or right <= left:
        return 0.0
    strip = mask[top:bottom, left:right]
    return float(np.mean(np.any(strip > 0, axis=1))) if strip.size else 0.0


def _row_aspect_score(aspect: float) -> float:
    # FCFs are horizontal rows. Keep this deliberately permissive: it is a
    # structural prior, not a calibrated acceptance probability.
    if aspect < 1.15:
        return 0.0
    if aspect < 1.70:
        return (aspect - 1.15) / (1.70 - 1.15)
    if aspect <= 12.0:
        return 1.0
    if aspect < 20.0:
        return (20.0 - aspect) / 8.0
    return 0.0


def _horizontal_extension_penalty(
    h_mask: np.ndarray,
    proposal: RawProposal,
    scale: float,
) -> float:
    """Penalize partial windows cut out of longer horizontal grid lines.

    A true outer frame should normally stop near its left/right bounds. A table
    sub-window often has both top and bottom lines continuing well beyond the
    proposed frame. One isolated continuation is tolerated because a real FCF can
    be connected to a leader/adjacent annotation.
    """

    box = proposal.frame_bbox
    extension = min(max(box.height * 2.0, 4.0), max(box.width * 0.50, 4.0))
    values: list[float] = []
    for y in (box.y0, box.y1):
        values.append(_h_coverage(h_mask, y, box.x0 - extension, box.x0 - 0.5, scale))
        values.append(_h_coverage(h_mask, y, box.x1 + 0.5, box.x1 + extension, scale))
    # Ignore the single strongest continuation so one legitimate attachment does
    # not kill recall. Multiple continuations still produce a strong penalty.
    ordered = sorted(values, reverse=True)
    retained = ordered[1:] if len(ordered) > 1 else ordered
    return float(np.mean(retained)) if retained else 0.0


def _vertical_continuation_fraction(
    v_mask: np.ndarray,
    proposal: RawProposal,
    scale: float,
) -> float:
    """Fraction of cell boundaries that continue vertically outside the frame.

    This is a strong signal for a row embedded in a larger table/grid: the same
    column lines keep going into neighboring rows. Real FCF dividers generally
    terminate at the frame's top/bottom.
    """

    box = proposal.frame_bbox
    boundaries = [box.x0]
    boundaries.extend(cell.x1 for cell in proposal.cell_bboxes[:-1])
    boundaries.append(box.x1)
    span = max(box.height * 1.75, 5.0)
    extended = 0
    for x in boundaries:
        above = _v_coverage(v_mask, x, box.y0 - span, box.y0 - 0.5, scale)
        below = _v_coverage(v_mask, x, box.y1 + 0.5, box.y1 + span, scale)
        if max(above, below) >= 0.25:
            extended += 1
    return extended / max(len(boundaries), 1)


def _count_nearby_parallel_rows(
    h_mask: np.ndarray,
    proposal: RawProposal,
    scale: float,
) -> int:
    """Count strong horizontal rows immediately above/below the candidate.

    Table/title-block rows repeat at similar x coverage. Only rows *outside* the
    proposed frame are counted; the frame's own top and bottom are excluded.
    """

    box = proposal.frame_bbox
    pad = max(box.height * 3.0, 8.0)
    x0 = max(0, int(round(box.x0 * scale)))
    x1 = min(h_mask.shape[1], int(round(box.x1 * scale)))
    y0 = max(0, int(round((box.y0 - pad) * scale)))
    y1 = min(h_mask.shape[0], int(round((box.y1 + pad) * scale)))
    if x1 <= x0 or y1 <= y0:
        return 0

    sub = h_mask[y0:y1, x0:x1]
    row_coverage = np.mean(sub > 0, axis=1)
    strong = np.where(row_coverage >= 0.60)[0]
    if len(strong) == 0:
        return 0

    gap_px = max(2, int(round(scale * 0.8)))
    groups: list[list[int]] = [[int(strong[0])]]
    for pos in strong[1:]:
        pos = int(pos)
        if pos - groups[-1][-1] <= gap_px:
            groups[-1].append(pos)
        else:
            groups.append([pos])

    count = 0
    exclusion = 1.8  # pt around candidate top/bottom
    for group in groups:
        center_px = sum(group) / len(group) + y0
        y_pt = center_px / scale
        if box.y0 - exclusion <= y_pt <= box.y1 + exclusion:
            continue
        count += 1
    return count


def evaluate_rectangle_geometry(
    proposal: RawProposal,
    h_mask: np.ndarray,
    v_mask: np.ndarray,
    scale: float,
) -> RectangleGeometryEvidence:
    box = proposal.frame_bbox
    top = _h_coverage(h_mask, box.y0, box.x0, box.x1, scale)
    bottom = _h_coverage(h_mask, box.y1, box.x0, box.x1, scale)
    left = _v_coverage(v_mask, box.x0, box.y0, box.y1, scale)
    right = _v_coverage(v_mask, box.x1, box.y0, box.y1, scale)
    outer_values = [top, bottom, left, right]
    outer_mean = float(np.mean(outer_values))
    outer_min = float(min(outer_values))

    divider_x = [cell.x1 for cell in proposal.cell_bboxes[:-1]]
    divider_values = [
        _v_coverage(v_mask, x, box.y0, box.y1, scale)
        for x in divider_x
    ]
    divider_mean = float(np.mean(divider_values)) if divider_values else 1.0

    aspect = box.width / max(box.height, 1e-6)
    aspect_score = _row_aspect_score(aspect)
    horizontal_extension = _horizontal_extension_penalty(h_mask, proposal, scale)
    vertical_continuation = _vertical_continuation_fraction(v_mask, proposal, scale)
    nearby_rows = _count_nearby_parallel_rows(h_mask, proposal, scale)

    # Large-grid penalty needs evidence in both axes. This deliberately avoids a
    # crude positional rule such as "bottom-right == title block".
    rows_component = min(1.0, nearby_rows / 3.0)
    grid_penalty = min(
        1.0,
        0.62 * vertical_continuation
        + 0.28 * rows_component
        + 0.10 * horizontal_extension,
    )

    endpoint_score = 1.0 - horizontal_extension
    geometry_score = (
        0.36 * outer_mean
        + 0.20 * divider_mean
        + 0.12 * outer_min
        + 0.14 * aspect_score
        + 0.18 * endpoint_score
        - 0.38 * grid_penalty
    )
    geometry_score = float(max(0.0, min(1.0, geometry_score)))

    return RectangleGeometryEvidence(
        top_coverage=top,
        bottom_coverage=bottom,
        left_coverage=left,
        right_coverage=right,
        outer_mean=outer_mean,
        outer_min=outer_min,
        divider_mean=divider_mean,
        row_aspect=aspect,
        row_aspect_score=aspect_score,
        horizontal_extension_penalty=horizontal_extension,
        vertical_continuation_fraction=vertical_continuation,
        nearby_parallel_rows=nearby_rows,
        grid_penalty=grid_penalty,
        geometry_score=geometry_score,
    )


def _same_row_band(a: RawProposal, b: RawProposal, tol: float = 1.8) -> bool:
    return abs(a.frame_bbox.y0 - b.frame_bbox.y0) <= tol and abs(a.frame_bbox.y1 - b.frame_bbox.y1) <= tol


def _horizontal_overlap_fraction(a: RawProposal, b: RawProposal) -> float:
    left = max(a.frame_bbox.x0, b.frame_bbox.x0)
    right = min(a.frame_bbox.x1, b.frame_bbox.x1)
    overlap = max(0.0, right - left)
    smaller = min(a.frame_bbox.width, b.frame_bbox.width)
    return overlap / smaller if smaller > 0 else 0.0


def _suppress_competing_windows(proposals: Sequence[RawProposal]) -> tuple[list[RawProposal], list[RawProposal]]:
    """Keep the best rectangle when several windows describe the same local row."""

    ranked = sorted(
        proposals,
        key=lambda proposal: (
            float(proposal.primitive_evidence.get("rectangle_geometry", {}).get("geometry_score", 0.0)),
            len(proposal.cell_bboxes),
            proposal.frame_bbox.width,
        ),
        reverse=True,
    )
    kept: list[RawProposal] = []
    suppressed: list[RawProposal] = []
    for proposal in ranked:
        competes = any(
            _same_row_band(proposal, existing)
            and _horizontal_overlap_fraction(proposal, existing) >= 0.72
            for existing in kept
        )
        if competes:
            proposal.validation_status = "rejected_by_rectangle_geometry"
            proposal.rejection_reasons.append("COMPETING_WINDOW_LOWER_GEOMETRY_SCORE")
            suppressed.append(proposal)
        else:
            kept.append(proposal)
    kept.sort(key=lambda p: (p.page, p.frame_bbox.y0, p.frame_bbox.x0))
    return kept, suppressed


def refine_rectangle_proposals(
    pdf_bytes: bytes,
    page_index: int,
    proposals: Iterable[RawProposal],
    *,
    dpi: int = 220,
    min_geometry_score: float = 0.52,
) -> tuple[list[RawProposal], list[RawProposal], dict]:
    """Refine raw proposals using rectangle geometry only.

    The gate is intentionally permissive because recall is still the primary
    objective. Strong large-grid evidence is a separate hard rejection because a
    perfect table cell can otherwise score as a perfect rectangle.
    """

    proposals = list(proposals)
    if not proposals:
        return [], [], {
            "rectangle_refinement_input": 0,
            "rectangle_refinement_kept": 0,
            "rectangle_refinement_rejected": 0,
            "rectangle_refinement_dpi": dpi,
        }

    h_mask, v_mask, scale = _render_line_masks(pdf_bytes, page_index, dpi)
    kept_pre_suppression: list[RawProposal] = []
    rejected: list[RawProposal] = []

    for proposal in proposals:
        evidence = evaluate_rectangle_geometry(proposal, h_mask, v_mask, scale)
        evidence_dict = evidence.to_dict()
        proposal.primitive_evidence["rectangle_geometry"] = evidence_dict
        proposal.primitive_evidence["rectangle_geometry_thresholds"] = {
            "min_geometry_score": min_geometry_score,
            "thresholds_are_validation_heuristics_not_iso": True,
        }

        reasons: list[str] = []
        if evidence.outer_mean < 0.58 or evidence.outer_min < 0.34:
            reasons.append("OUTER_BORDER_SUPPORT_LOW")
        if evidence.divider_mean < 0.45:
            reasons.append("DIVIDER_SUPPORT_LOW")
        if evidence.row_aspect_score <= 0.0:
            reasons.append("FRAME_NOT_ROW_LIKE")
        if evidence.grid_penalty >= 0.74 and evidence.nearby_parallel_rows >= 2:
            reasons.append("EMBEDDED_IN_LARGE_GRID")
        if evidence.geometry_score < min_geometry_score:
            reasons.append("RECTANGLE_GEOMETRY_SCORE_LOW")

        if reasons:
            proposal.validation_status = "rejected_by_rectangle_geometry"
            proposal.rejection_reasons.extend(reason for reason in reasons if reason not in proposal.rejection_reasons)
            rejected.append(proposal)
        else:
            proposal.validation_status = "accepted_rectangle_geometry"
            kept_pre_suppression.append(proposal)

    kept, suppressed = _suppress_competing_windows(kept_pre_suppression)
    rejected.extend(suppressed)

    audit = {
        "rectangle_refinement_input": len(proposals),
        "rectangle_refinement_kept_before_competing_window_suppression": len(kept_pre_suppression),
        "rectangle_refinement_kept": len(kept),
        "rectangle_refinement_rejected": len(rejected),
        "rectangle_refinement_dpi": dpi,
        "rectangle_geometry_min_score": min_geometry_score,
        "rectangle_geometry_thresholds_are_validation_heuristics_not_iso": True,
    }
    return kept, rejected, audit


__all__ = [
    "RectangleGeometryEvidence",
    "evaluate_rectangle_geometry",
    "refine_rectangle_proposals",
]
