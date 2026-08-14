"""High-recall GD&T candidate detector V2.

Design goals
------------
1. Preserve the validated V1 detector as one proposal source.
2. Normalize vector drawing primitives into a common H/V edge model so export
   differences (line segments vs rectangle primitives) do not decide recall.
3. Add a raster proposal source as fallback for drawings whose vector topology
   is not recoverable by the V1 assumptions.
4. Keep *raw proposals* separate from *accepted candidates*. Geometry proposes;
   structural/content evidence validates.
5. Never call a proposal TP/FP without independent ground truth.

The validator is deliberately conservative about semantics: it does not claim
that a proposal is a true FCF. It only labels whether the proposal is
``accepted_for_downstream`` or ``rejected_by_validator`` and records evidence.
Symbol ranking remains diagnostic and is not a calibrated probability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import fitz
import numpy as np
from PIL import Image

from src.gdt.detector import BBox, GdtCell, GdtFrameCandidate, GdtFrameDetector


@dataclass
class EdgeH:
    y: float
    x0: float
    x1: float
    source: str


@dataclass
class EdgeV:
    x: float
    y0: float
    y1: float
    source: str


@dataclass
class RawProposal:
    proposal_id: str
    page: int
    frame_bbox: BBox
    cell_bboxes: List[BBox]
    symbol_bbox: BBox
    sources: List[str] = field(default_factory=list)
    primitive_evidence: dict = field(default_factory=dict)
    validation_status: str = "raw_proposal"
    rejection_reasons: List[str] = field(default_factory=list)
    validator_evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "proposal_id": self.proposal_id,
            "page": self.page,
            "frame_bbox": self.frame_bbox.to_list(),
            "cell_bboxes": [b.to_list() for b in self.cell_bboxes],
            "symbol_bbox": self.symbol_bbox.to_list(),
            "sources": list(self.sources),
            "primitive_evidence": dict(self.primitive_evidence),
            "validation_status": self.validation_status,
            "rejection_reasons": list(self.rejection_reasons),
            "validator_evidence": dict(self.validator_evidence),
        }


@dataclass
class CandidateDetectionV2Result:
    page: int
    primitive_audit: dict
    raw_proposals: List[RawProposal]
    accepted_candidates: List[GdtFrameCandidate]
    rejected_proposals: List[RawProposal]

    def to_dict(self) -> dict:
        return {
            "page": self.page,
            "primitive_audit": dict(self.primitive_audit),
            "raw_proposals": [p.to_dict() for p in self.raw_proposals],
            "accepted_candidate_ids": [c.candidate_id for c in self.accepted_candidates],
            "rejected_proposal_ids": [p.proposal_id for p in self.rejected_proposals],
        }


def _bbox_iou(a: BBox, b: BBox) -> float:
    ix0, iy0 = max(a.x0, b.x0), max(a.y0, b.y0)
    ix1, iy1 = min(a.x1, b.x1), min(a.y1, b.y1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def _bbox_ios(a: BBox, b: BBox) -> float:
    ix0, iy0 = max(a.x0, b.x0), max(a.y0, b.y0)
    ix1, iy1 = min(a.x1, b.x1), min(a.y1, b.y1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    small = min(a.area, b.area)
    return inter / small if small > 0 else 0.0


def _rect_edges(rect: fitz.Rect, source: str) -> tuple[list[EdgeH], list[EdgeV]]:
    return (
        [EdgeH(rect.y0, rect.x0, rect.x1, source), EdgeH(rect.y1, rect.x0, rect.x1, source)],
        [EdgeV(rect.x0, rect.y0, rect.y1, source), EdgeV(rect.x1, rect.y0, rect.y1, source)],
    )


def audit_and_normalize_vector_primitives(page: fitz.Page, *, axis_tolerance: float = 1.0) -> tuple[list[EdgeH], list[EdgeV], dict]:
    """Normalize PyMuPDF drawing items into horizontal/vertical edges.

    Supported explicitly:
    - ``l`` line segments;
    - ``re`` rectangle primitives;
    - ``qu`` quadrilaterals, when their sides are axis-aligned within tolerance.

    Curves and unknown primitives are audited but not approximated into straight
    edges, avoiding silent geometry invention.
    """

    h_edges: list[EdgeH] = []
    v_edges: list[EdgeV] = []
    counts: dict[str, int] = {}

    for drawing in page.get_drawings():
        for item in drawing.get("items", []):
            kind = str(item[0])
            counts[kind] = counts.get(kind, 0) + 1

            if kind == "l":
                p1, p2 = item[1], item[2]
                dx, dy = abs(p2.x - p1.x), abs(p2.y - p1.y)
                if dy <= axis_tolerance and dx > 0:
                    h_edges.append(EdgeH((p1.y + p2.y) / 2.0, min(p1.x, p2.x), max(p1.x, p2.x), "vector:l"))
                elif dx <= axis_tolerance and dy > 0:
                    v_edges.append(EdgeV((p1.x + p2.x) / 2.0, min(p1.y, p2.y), max(p1.y, p2.y), "vector:l"))
                continue

            if kind == "re":
                rect = fitz.Rect(item[1])
                hs, vs = _rect_edges(rect, "vector:re")
                h_edges.extend(hs)
                v_edges.extend(vs)
                continue

            if kind == "qu":
                quad = item[1]
                pts = [quad.ul, quad.ur, quad.lr, quad.ll, quad.ul]
                for p1, p2 in zip(pts, pts[1:]):
                    dx, dy = abs(p2.x - p1.x), abs(p2.y - p1.y)
                    if dy <= axis_tolerance and dx > 0:
                        h_edges.append(EdgeH((p1.y + p2.y) / 2.0, min(p1.x, p2.x), max(p1.x, p2.x), "vector:qu"))
                    elif dx <= axis_tolerance and dy > 0:
                        v_edges.append(EdgeV((p1.x + p2.x) / 2.0, min(p1.y, p2.y), max(p1.y, p2.y), "vector:qu"))

    audit = {
        "primitive_counts": counts,
        "normalized_horizontal_edges": len(h_edges),
        "normalized_vertical_edges": len(v_edges),
        "supported_primitives": ["l", "re", "qu"],
        "unsupported_primitives_observed": sorted(k for k in counts if k not in {"l", "re", "qu"}),
    }
    return h_edges, v_edges, audit


def _merge_h(edges: Sequence[EdgeH], *, y_tol: float = 1.2, gap: float = 3.5) -> list[EdgeH]:
    if not edges:
        return []
    rows = sorted(edges, key=lambda e: (e.y, e.x0))
    groups: list[list[EdgeH]] = [[rows[0]]]
    for edge in rows[1:]:
        if abs(edge.y - groups[-1][-1].y) <= y_tol:
            groups[-1].append(edge)
        else:
            groups.append([edge])
    out: list[EdgeH] = []
    for group in groups:
        group.sort(key=lambda e: e.x0)
        y = sum(e.y for e in group) / len(group)
        cur_x0, cur_x1 = group[0].x0, group[0].x1
        sources = {group[0].source}
        for edge in group[1:]:
            if edge.x0 <= cur_x1 + gap:
                cur_x1 = max(cur_x1, edge.x1)
                sources.add(edge.source)
            else:
                out.append(EdgeH(y, cur_x0, cur_x1, "+".join(sorted(sources))))
                cur_x0, cur_x1, sources = edge.x0, edge.x1, {edge.source}
        out.append(EdgeH(y, cur_x0, cur_x1, "+".join(sorted(sources))))
    return out


def _merge_v(edges: Sequence[EdgeV], *, x_tol: float = 1.2, gap: float = 3.5) -> list[EdgeV]:
    if not edges:
        return []
    cols = sorted(edges, key=lambda e: (e.x, e.y0))
    groups: list[list[EdgeV]] = [[cols[0]]]
    for edge in cols[1:]:
        if abs(edge.x - groups[-1][-1].x) <= x_tol:
            groups[-1].append(edge)
        else:
            groups.append([edge])
    out: list[EdgeV] = []
    for group in groups:
        group.sort(key=lambda e: e.y0)
        x = sum(e.x for e in group) / len(group)
        cur_y0, cur_y1 = group[0].y0, group[0].y1
        sources = {group[0].source}
        for edge in group[1:]:
            if edge.y0 <= cur_y1 + gap:
                cur_y1 = max(cur_y1, edge.y1)
                sources.add(edge.source)
            else:
                out.append(EdgeV(x, cur_y0, cur_y1, "+".join(sorted(sources))))
                cur_y0, cur_y1, sources = edge.y0, edge.y1, {edge.source}
        out.append(EdgeV(x, cur_y0, cur_y1, "+".join(sorted(sources))))
    return out


def vector_high_recall_proposals(
    page: fitz.Page,
    *,
    page_number: int,
    min_height: float = 4.0,
    max_height: float = 42.0,
    min_width: float = 12.0,
    max_width: float = 320.0,
    min_cell_width: float = 3.0,
    endpoint_tolerance: float = 5.5,
) -> tuple[list[RawProposal], dict]:
    h_raw, v_raw, audit = audit_and_normalize_vector_primitives(page)
    hs, vs = _merge_h(h_raw), _merge_v(v_raw)
    audit["merged_horizontal_edges"] = len(hs)
    audit["merged_vertical_edges"] = len(vs)

    proposals: list[RawProposal] = []
    for i, top in enumerate(hs):
        for bottom in hs[i + 1 :]:
            height = bottom.y - top.y
            if height < min_height:
                continue
            if height > max_height:
                break
            left_overlap = max(top.x0, bottom.x0)
            right_overlap = min(top.x1, bottom.x1)
            if right_overlap - left_overlap < min_width:
                continue

            connectors = [
                v for v in vs
                if left_overlap - endpoint_tolerance <= v.x <= right_overlap + endpoint_tolerance
                and v.y0 <= top.y + endpoint_tolerance
                and v.y1 >= bottom.y - endpoint_tolerance
            ]
            connectors.sort(key=lambda e: e.x)
            dedup: list[EdgeV] = []
            for edge in connectors:
                if not dedup or edge.x - dedup[-1].x >= min_cell_width:
                    dedup.append(edge)
            if len(dedup) < 3:
                continue

            # High-recall proposal generation: try coherent contiguous windows of
            # 2..6 cells instead of rejecting a local FCF because a nearby table
            # line created too many connectors globally.
            max_edges = min(7, len(dedup))
            for n_edges in range(3, max_edges + 1):
                for start in range(0, len(dedup) - n_edges + 1):
                    window = dedup[start : start + n_edges]
                    width = window[-1].x - window[0].x
                    if not (min_width <= width <= max_width):
                        continue
                    cells = [BBox(a.x, top.y, b.x, bottom.y) for a, b in zip(window, window[1:])]
                    first_aspect = cells[0].width / max(cells[0].height, 1e-6)
                    if not (0.30 <= first_aspect <= 3.50):
                        continue
                    frame = BBox(window[0].x, top.y, window[-1].x, bottom.y)
                    sources = sorted({top.source, bottom.source, *(v.source for v in window)})
                    proposals.append(
                        RawProposal(
                            proposal_id="",
                            page=page_number,
                            frame_bbox=frame,
                            cell_bboxes=cells,
                            symbol_bbox=cells[0],
                            sources=["vector_normalized_v2", *sources],
                            primitive_evidence={"connector_count": len(window), "height": height, "width": width},
                        )
                    )

    audit["vector_raw_proposals_before_dedup"] = len(proposals)
    return proposals, audit


def raster_high_recall_proposals(
    page: fitz.Page,
    *,
    page_number: int,
    dpi: int = 220,
    min_cells: int = 2,
    max_cells: int = 6,
) -> list[RawProposal]:
    """Generate raster fallback proposals from small row-like rectangle groups.

    The raster branch is intentionally a proposal generator, not a decision
    maker. It uses morphology to expose horizontal/vertical ruling and connected
    contours, then looks for several adjacent small rectangular cells sharing a
    common vertical band.
    """

    scale = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=fitz.csGRAY, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    binary = cv2.threshold(img, 210, 255, cv2.THRESH_BINARY_INV)[1]

    # Extract long-ish H/V strokes while suppressing most glyph details.
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(8, int(scale * 3.0)), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(8, int(scale * 3.0))))
    hmask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, hk)
    vmask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vk)
    grid = cv2.bitwise_or(hmask, vmask)
    grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))

    contours, _ = cv2.findContours(grid, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    rects: list[tuple[int, int, int, int]] = []
    page_area = float(pix.width * pix.height)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < page_area * 0.000003 or area > page_area * 0.01:
            continue
        if h < max(5, int(scale * 3.5)) or h > int(scale * 45):
            continue
        if w < max(5, int(scale * 3.0)) or w > int(scale * 320):
            continue
        rects.append((x, y, x + w, y + h))

    # Build row groups by y-overlap/height similarity.
    rects.sort(key=lambda r: (r[1], r[0]))
    groups: list[list[tuple[int, int, int, int]]] = []
    for rect in rects:
        placed = False
        cy = (rect[1] + rect[3]) / 2.0
        rh = rect[3] - rect[1]
        for group in groups:
            g0 = group[0]
            gcy = (g0[1] + g0[3]) / 2.0
            gh = g0[3] - g0[1]
            if abs(cy - gcy) <= max(4, 0.30 * max(rh, gh)) and abs(rh - gh) <= max(4, 0.35 * max(rh, gh)):
                group.append(rect)
                placed = True
                break
        if not placed:
            groups.append([rect])

    proposals: list[RawProposal] = []
    for group in groups:
        group.sort(key=lambda r: r[0])
        # Remove nested/near-duplicate cell boxes.
        clean: list[tuple[int, int, int, int]] = []
        for rect in group:
            if any(
                abs(rect[0] - c[0]) <= 3 and abs(rect[1] - c[1]) <= 3
                and abs(rect[2] - c[2]) <= 3 and abs(rect[3] - c[3]) <= 3
                for c in clean
            ):
                continue
            clean.append(rect)
        if len(clean) < min_cells:
            continue

        for n in range(min_cells, min(max_cells, len(clean)) + 1):
            for start in range(len(clean) - n + 1):
                cells_px = clean[start : start + n]
                # Adjacent cells in an FCF should touch or have only a tiny gap.
                gaps = [b[0] - a[2] for a, b in zip(cells_px, cells_px[1:])]
                if any(g > max(5, int(scale * 2.5)) for g in gaps):
                    continue
                y0 = min(r[1] for r in cells_px)
                y1 = max(r[3] for r in cells_px)
                x0, x1 = cells_px[0][0], cells_px[-1][2]
                frame = BBox(x0 / scale, y0 / scale, x1 / scale, y1 / scale)
                cells = [BBox(r[0] / scale, r[1] / scale, r[2] / scale, r[3] / scale) for r in cells_px]
                proposals.append(
                    RawProposal(
                        proposal_id="",
                        page=page_number,
                        frame_bbox=frame,
                        cell_bboxes=cells,
                        symbol_bbox=cells[0],
                        sources=["raster_morphology_v2"],
                        primitive_evidence={"dpi": dpi, "cell_count": len(cells)},
                    )
                )
    return proposals


def _candidate_from_raw(proposal: RawProposal, page: fitz.Page, page_image: Image.Image, scale: float) -> GdtFrameCandidate:
    words = page.get_text("words")
    cells: list[GdtCell] = []
    for box in proposal.cell_bboxes:
        texts: list[str] = []
        for word in words:
            wx, wy = (word[0] + word[2]) / 2.0, (word[1] + word[3]) / 2.0
            if box.contains_point(wx, wy) and str(word[4]).strip():
                texts.append(str(word[4]).strip())
        cells.append(GdtCell(bbox=box, texts=texts))

    def crop(box: BBox, pad_pt: float = 4.0) -> Image.Image:
        p = pad_pt * scale
        px = (
            max(0, int(round(box.x0 * scale - p))),
            max(0, int(round(box.y0 * scale - p))),
            min(page_image.width, int(round(box.x1 * scale + p))),
            min(page_image.height, int(round(box.y1 * scale + p))),
        )
        return page_image.crop(px)

    return GdtFrameCandidate(
        candidate_id=proposal.proposal_id,
        page=proposal.page,
        frame_bbox=proposal.frame_bbox,
        symbol_bbox=proposal.symbol_bbox,
        cells=cells,
        frame_crop=crop(proposal.frame_bbox),
        symbol_crop=crop(proposal.symbol_bbox),
        confidence_score=0.0,
    )


def _deduplicate_raw(proposals: Iterable[RawProposal], *, iou: float = 0.55, ios: float = 0.88) -> list[RawProposal]:
    ranked = sorted(
        list(proposals),
        key=lambda p: ("v1_legacy" in p.sources, len(p.cell_bboxes), p.frame_bbox.area),
        reverse=True,
    )
    kept: list[RawProposal] = []
    for proposal in ranked:
        match: Optional[RawProposal] = None
        for existing in kept:
            if _bbox_iou(proposal.frame_bbox, existing.frame_bbox) >= iou or _bbox_ios(proposal.frame_bbox, existing.frame_bbox) >= ios:
                match = existing
                break
        if match is None:
            kept.append(proposal)
        else:
            match.sources = sorted(set(match.sources + proposal.sources))
            match.primitive_evidence.setdefault("merged_sources", []).extend(proposal.sources)
    kept.sort(key=lambda p: (p.page, p.frame_bbox.y0, p.frame_bbox.x0))
    for idx, proposal in enumerate(kept, start=1):
        proposal.proposal_id = f"GDT-V2-P{proposal.page:02d}-{idx:03d}"
    return kept


def _score_map(score: Any) -> dict[str, float]:
    """Best-effort extraction of class ranking from current score objects."""
    if score is None:
        return {}
    payload = score.to_dict() if hasattr(score, "to_dict") else dict(score)
    result: dict[str, float] = {}
    for key in ("class_scores", "scores", "ranking"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            for name, val in value.items():
                try:
                    result[str(name)] = float(val)
                except (TypeError, ValueError):
                    pass
        elif isinstance(value, list):
            for row in value:
                if isinstance(row, Mapping):
                    name = row.get("class_name") or row.get("class") or row.get("name")
                    val = row.get("score")
                    if name is not None and val is not None:
                        try:
                            result[str(name)] = float(val)
                        except (TypeError, ValueError):
                            pass
    best = payload.get("best_class")
    best_score = payload.get("best_score")
    if best is not None and best_score is not None:
        try:
            result.setdefault(str(best), float(best_score))
        except (TypeError, ValueError):
            pass
    return result


def validate_proposals(
    proposals: Sequence[RawProposal],
    candidates: Sequence[GdtFrameCandidate],
    scores: Sequence[Any],
    *,
    symbol_min_score: float = 0.30,
    symbol_min_margin: float = 0.015,
) -> tuple[list[GdtFrameCandidate], list[RawProposal]]:
    """Validate proposals using structure + first-cell symbol evidence.

    The numeric defaults are *engineering heuristics for validation only*, not ISO
    tolerances and not production-calibrated thresholds. They are intentionally
    permissive to protect recall. A proposal can pass with ambiguous symbol
    ranking when its downstream cells contain tolerance/datum-like evidence.
    """

    accepted: list[GdtFrameCandidate] = []
    rejected: list[RawProposal] = []
    for proposal, candidate, score in zip(proposals, candidates, scores):
        score_payload = score.to_dict() if hasattr(score, "to_dict") else (dict(score) if score is not None else {})
        score_values = sorted(_score_map(score).values(), reverse=True)
        best_score = score_values[0] if score_values else None
        second_score = score_values[1] if len(score_values) > 1 else None
        margin = (best_score - second_score) if best_score is not None and second_score is not None else None

        downstream_text = [t for cell in candidate.cells[1:] for t in cell.texts]
        has_numeric = any(any(ch.isdigit() for ch in text) for text in downstream_text)
        has_single_letter = any(len(text.strip()) == 1 and text.strip().isalpha() for text in downstream_text)
        num_cells = len(candidate.cells)
        first_aspect = candidate.symbol_bbox.width / max(candidate.symbol_bbox.height, 1e-6)

        symbol_like = best_score is not None and best_score >= symbol_min_score
        symbol_separated = margin is None or margin >= symbol_min_margin
        structural_support = 2 <= num_cells <= 6 and 0.30 <= first_aspect <= 3.50
        content_support = has_numeric or has_single_letter

        reasons: list[str] = []
        if not structural_support:
            reasons.append("STRUCTURE_NOT_FCF_LIKE")
        if not symbol_like:
            reasons.append("FIRST_CELL_NOT_GDT_LIKE")
        # An ambiguous first-cell rank is tolerated only when downstream content
        # looks like a tolerance/datum cell. This avoids rejecting real line-like
        # symbols such as parallelism while still suppressing table crops.
        if symbol_like and not symbol_separated and not content_support:
            reasons.append("SYMBOL_AMBIGUOUS_WITHOUT_DOWNSTREAM_SUPPORT")
        if not content_support and num_cells >= 3:
            reasons.append("NO_TOLERANCE_OR_DATUM_LIKE_CONTENT")

        proposal.validator_evidence = {
            "best_class": score_payload.get("best_class"),
            "best_score": best_score,
            "margin": margin,
            "num_cells": num_cells,
            "first_cell_aspect": first_aspect,
            "has_numeric_downstream": has_numeric,
            "has_single_letter_downstream": has_single_letter,
            "thresholds_are_validation_heuristics_not_iso": True,
        }

        if reasons:
            proposal.validation_status = "rejected_by_validator"
            proposal.rejection_reasons = reasons
            rejected.append(proposal)
        else:
            proposal.validation_status = "accepted_for_downstream"
            candidate.confidence_score = min(
                1.0,
                0.35
                + (0.20 if content_support else 0.0)
                + (0.20 if symbol_like else 0.0)
                + (0.10 if symbol_separated else 0.0)
                + (0.15 if 3 <= num_cells <= 4 else 0.0),
            )
            accepted.append(candidate)

    return accepted, rejected


class GdtCandidateDetectorV2:
    """Hybrid proposal detector: V1 + normalized vector + raster fallback."""

    def __init__(self, *, raster_dpi: int = 220):
        self.raster_dpi = int(raster_dpi)
        self.v1 = GdtFrameDetector()

    def propose(self, pdf_bytes: bytes, *, page_index: int = 0) -> tuple[list[RawProposal], dict]:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            page = doc[page_index]
            vector_props, audit = vector_high_recall_proposals(page, page_number=page_index + 1)
            raster_props = raster_high_recall_proposals(page, page_number=page_index + 1, dpi=self.raster_dpi)

        v1_candidates = self.v1.detect_frames(pdf_bytes, page_index=page_index)
        v1_props = [
            RawProposal(
                proposal_id="",
                page=c.page,
                frame_bbox=c.frame_bbox,
                cell_bboxes=[cell.bbox for cell in c.cells],
                symbol_bbox=c.symbol_bbox,
                sources=["v1_legacy"],
                primitive_evidence={"v1_confidence": c.confidence_score},
            )
            for c in v1_candidates
        ]

        combined = _deduplicate_raw([*v1_props, *vector_props, *raster_props])
        audit.update(
            {
                "v1_proposals": len(v1_props),
                "normalized_vector_proposals": len(vector_props),
                "raster_proposals": len(raster_props),
                "combined_after_dedup": len(combined),
            }
        )
        return combined, audit

    def materialize(self, pdf_bytes: bytes, proposals: Sequence[RawProposal], *, page_index: int = 0, dpi: int = 300) -> list[GdtFrameCandidate]:
        scale = dpi / 72.0
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            page = doc[page_index]
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=fitz.csRGB, alpha=False)
            image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            return [_candidate_from_raw(p, page, image, scale) for p in proposals]


__all__ = [
    "RawProposal",
    "CandidateDetectionV2Result",
    "GdtCandidateDetectorV2",
    "audit_and_normalize_vector_primitives",
    "vector_high_recall_proposals",
    "raster_high_recall_proposals",
    "validate_proposals",
]
