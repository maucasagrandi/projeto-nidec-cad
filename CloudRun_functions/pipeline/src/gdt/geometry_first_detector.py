"""Geometry-first, scale-measured GD&T detector.

This module replaces the full-page multi-scale/multi-rotation template scan
(``GdtTemplateDetector``) used by ``analyze_page`` Step 1 + Step 2. It is built
on two domain facts:

1. **Boxes first.** GD&T symbols only ever appear inside the *first cell* of a
   feature-control-frame (FCF) box. So we find candidate boxes from PDF vector
   geometry first (resolution-independent), then classify *only inside* the
   first cell of each candidate. This turns ~672 full-page correlations per page
   into a few dozen tiny fixed-size crop classifications, and structurally
   rejects anything that is not a box-shaped, FCF-proportioned region.

2. **Measured scale.** The symbol template prints include the surrounding box,
   so the box is a *scale ruler*. Each template's own box height (auto-measured
   from its pixels) versus a detected cell's height yields the render scale.
   Because a CAD export renders the whole page at one zoom, we lock a single
   page-level scale from the first confident box and use it as a *consistency
   filter* (the classifier itself is already scale-invariant via 48x48
   canonicalization), demoting boxes whose implied scale is off-band.

The public entry point is :func:`detect_geometry_first_frames`, which returns
``(detections, frames, audit)`` with the SAME ``Detection`` / ``FcfFrame`` shapes
that ``analyze_page`` Step 1 + Step 2 produced, so everything downstream (datum
steps, report assembly, annotated rendering) is unchanged.

Thresholds (``min_score`` / ``min_margin`` / ``negative_margin``) are inherited
from the anchor detector and remain engineering heuristics to be calibrated on
the example folders; they are not ISO-calibrated production thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import fitz
import numpy as np

from src.gdt.candidate_detector_v2 import vector_high_recall_proposals
from src.gdt.fcf_expander import (
    _assign_cell_roles,
    expand_fcf_from_detection,
    extract_page_lines,
)
from src.gdt.symbol_anchor_detector import (
    _cell_rectangles_from_raster,
    _dedup_boxes,
    _grow_frame_from_anchor,
)
from src.gdt.template_detector import (
    Detection,
    TemplateDef,
    _rotate_image,
    load_templates,
    render_page_gray,
)
from src.gdt.fcf_expander import FcfCell, FcfFrame
from src.utils.gdt_detector import BBox, GdtCell, GdtFrameCandidate

# FCF cell geometry (points). Mirrors fcf_expander constants; the symbol (first)
# cell is compact, so its width band is tighter than a whole FCF row.
MIN_FCF_HEIGHT_PT = 8.0
MAX_FCF_HEIGHT_PT = 30.0  # angularity symbol cells reach ~26pt
MIN_CELL_WIDTH_PT = 5.0
MAX_SYMBOL_CELL_WIDTH_PT = 34.0
# Symbol cells range from narrow/portrait (symmetry ~11x18) to wide (~24x14),
# so allow a broad aspect band; NCC scoring rejects non-symbols precisely.
CELL_ASPECT_RANGE = (0.30, 3.50)
LINE_TOLERANCE_PT = 2.0

# NCC acceptance for a box crop to count as a GD&T symbol. Comparable to the old
# full-page detector's 0.74 floor; slightly lower since we already constrained
# to box-shaped crops. The margin (best vs second-best *class*) rejects the
# false-positive mode where a geometrically simple template (e.g. symmetry's
# stacked lines) correlates weakly-but-similarly with table rulings.
DEFAULT_NCC_MIN_SCORE = 0.66
# Margin gate (best vs second-best class). The per-class score floors are now the
# primary precision mechanism; the margin only needs to break near-ties. A low
# value (0.05) recovers real symbols that resemble a runner-up class (e.g. a real
# perpendicularity scoring 0.85 with a 0.08 margin) while a higher gate rejected
# them. Calibrated on the example set: 0.05 recovers ~35 real detections vs 0.12
# with no increase in false positives.
DEFAULT_NCC_MIN_MARGIN = 0.05

# Score above a class's floor at which the margin requirement is waived entirely
# (a confident detection is accepted even against a close runner-up).
DEFAULT_MARGIN_WAIVER_MARGIN = 0.12

# Per-class NCC acceptance floors, calibrated from the score distribution across
# the example set (16/50/84 percentiles). Distinctive, box-shaped symbols score
# tightly high (p16 >= 0.82), so a 0.80 floor is safe. The geometrically simple
# single-line symbols (angularity, straightness) are false-positive magnets:
# their real detections — when they exist — score >= 0.83, while hatching/table
# false positives top out at ~0.82, so a 0.83 floor separates them cleanly.
# Classes not listed fall back to DEFAULT_NCC_MIN_SCORE.
PER_CLASS_MIN_SCORE: dict[str, float] = {
    # FP-prone single-line symbols: real detections (when present) score >= 0.83,
    # while hatching/table false positives top out at ~0.82.
    "angularity": 0.83,
    "straightness": 0.83,
    # Remaining floors are set near each class's 10th-percentile true-positive
    # score (measured from detections that agree with the legacy detector on the
    # example set), so ~90% of real symbols pass while staying above the FP band.
    "circular_runout": 0.82,
    "circularity": 0.80,
    "concentricity_coaxiality": 0.78,
    "cylindricity": 0.72,
    "flatness": 0.82,
    "parallelism": 0.68,
    "perpendicularity": 0.80,
    "position": 0.80,
    "profile": 0.65,
    "symmetry": 0.66,
    "total_runout": 0.74,
}


def _class_min_score(class_name: str | None, default: float) -> float:
    """Acceptance floor for a class (per-class calibrated, else the default)."""
    if class_name is None:
        return default
    return PER_CLASS_MIN_SCORE.get(class_name, default)


# Rotations to try per template crop. FCF symbols are usually upright, but some
# (notably straightness — a plain line) appear rotated; matching only upright
# under-scores a real rotated symbol and makes it look like a false positive.
# The crops are tiny, so trying a few rotations is cheap (unlike a full-page
# rotated scan).
DEFAULT_MATCH_ROTATIONS = (0, 90, -90)

# Scale multipliers to try around the measured box scale. The measured box
# height fixes the nominal scale, but auto-measure and cell geometry can be a
# little off, and symbols vary in how tightly they fill their cell. A small
# sweep recovers those without reverting to the old blind 8-scale full-page
# search (this sweep is over a tiny crop, and centered on a known scale).
DEFAULT_MATCH_SCALES = (0.8, 0.9, 1.0, 1.1, 1.2)

# Default classifier render DPI (matches symbol_classifier default).
DEFAULT_SYMBOL_DPI = 300


# ==============================================================================
# Component 3: Template auto-measure
# ==============================================================================


@dataclass(frozen=True)
class TemplateBoxMetric:
    """Auto-measured reference geometry of one template image.

    ``box_height_px`` is the height of the template's own enclosing rectangle
    (the printed FCF cell border) in template pixels; ``cell_height_px`` is the
    inner height after subtracting the border stroke. ``ok`` is False when no
    rectangle could be measured.
    """

    class_name: str
    template_name: str
    box_height_px: float
    cell_height_px: float
    ok: bool


def _border_coverage(binary: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    """Fraction of the 4 sides of a rect that are covered by ink (0..1)."""
    if w < 2 or h < 2:
        return 0.0
    b = max(1, int(round(min(w, h) * 0.12)))
    roi = binary[y : y + h, x : x + w]
    if roi.size == 0:
        return 0.0
    top = float((roi[:b, :] > 0).mean())
    bottom = float((roi[-b:, :] > 0).mean())
    left = float((roi[:, :b] > 0).mean())
    right = float((roi[:, -b:] > 0).mean())
    return float(np.mean([top, bottom, left, right]))


def _estimate_border_thickness(binary: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    """Estimate the box border stroke thickness (px) from the top edge run."""
    strip = binary[y : y + max(1, h // 3), x : x + w]
    if strip.size == 0:
        return 1.0
    # Ink pixels per row near the top; the border row is (near-)fully inked.
    row_ink = (strip > 0).mean(axis=1)
    thick = int(np.count_nonzero(row_ink >= 0.6))
    return float(max(1, thick))


def measure_template_box(image_gray: np.ndarray) -> TemplateBoxMetric:
    """Detect a template's own enclosing box and return its measured heights.

    The template print includes the surrounding rectangle. We threshold to ink,
    find the largest enclosing rectangle with credible border coverage, and take
    its height. If no bordered rectangle is found we fall back to the bounding
    box of all ink (the box may itself be the outermost contour).
    """
    empty = TemplateBoxMetric("", "", 0.0, 0.0, False)
    if image_gray is None or image_gray.size == 0:
        return empty

    gray = image_gray
    if float(gray.mean()) < 127.0:
        gray = 255 - gray  # normalize to dark ink on light background
    binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)[1]
    if not binary.any():
        return empty

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best: tuple[int, int, int, int] | None = None
    best_area = 0.0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 4 or h < 4:
            continue
        coverage = _border_coverage(binary, x, y, w, h)
        area = float(w * h)
        if coverage >= 0.5 and area > best_area:
            best, best_area = (x, y, w, h), area

    if best is None:
        # Fallback: box is the outermost ink bounding box.
        ys, xs = np.nonzero(binary)
        if xs.size == 0:
            return empty
        x, y = int(xs.min()), int(ys.min())
        w, h = int(xs.max() - x + 1), int(ys.max() - y + 1)
        best = (x, y, w, h)

    bx, by, bw, bh = best
    border = _estimate_border_thickness(binary, bx, by, bw, bh)
    box_h = float(bh)
    cell_h = float(max(1.0, bh - 2.0 * border))
    return TemplateBoxMetric("", "", box_h, cell_h, True)


@lru_cache(maxsize=8)
def build_template_box_metrics(template_root: str) -> dict[tuple[str, str], TemplateBoxMetric]:
    """Auto-measure every template's box height once, cached per template_root.

    Keyed by ``(class_name, template_name)`` to match ``TemplateImage`` /
    ``TemplateScore`` identifiers.
    """
    root = Path(template_root)
    if not root.exists():
        return {}
    allowed = {".png", ".jpg", ".jpeg", ".webp"}
    metrics: dict[tuple[str, str], TemplateBoxMetric] = {}
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        class_name = class_dir.name.strip().lower()
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() not in allowed:
                continue
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            m = measure_template_box(gray)
            metrics[(class_name, path.stem)] = TemplateBoxMetric(
                class_name=class_name,
                template_name=path.stem,
                box_height_px=m.box_height_px,
                cell_height_px=m.cell_height_px,
                ok=m.ok,
            )
    return metrics


def _class_box_height_px(
    metrics: Mapping[tuple[str, str], TemplateBoxMetric], class_name: str | None
) -> float | None:
    """Median measured box height (px) across a class's templates, or None."""
    if class_name is None:
        return None
    heights = [
        m.box_height_px
        for (cls, _tpl), m in metrics.items()
        if cls == class_name and m.ok and m.box_height_px > 0
    ]
    if not heights:
        return None
    return float(np.median(heights))


# ==============================================================================
# Component 4: Page-scale estimator
# ==============================================================================


@dataclass
class PageScaleLock:
    """One page-level render scale locked from the first confident box.

    ``scale`` is normalized (detected cell height in template-pixel units divided
    by the template's own box height in pixels), so it is ~1.0 when a detected
    cell matches the template's proportions. ``tolerance`` is the +/- band used to
    demote off-scale boxes.
    """

    scale: float
    locked: bool
    tolerance: float = 0.05
    source_anchor_id: str | None = None

    def in_band(self, other_scale: float) -> bool:
        if not self.locked or self.scale <= 0:
            return True  # no lock -> do not reject anything
        return abs(other_scale - self.scale) <= self.tolerance * self.scale


def measure_scale(cell_height_px: float, template_box_height_px: float) -> float:
    """Render scale = detected cell height / template box height (both px)."""
    if template_box_height_px <= 0:
        return 0.0
    return float(cell_height_px) / float(template_box_height_px)


def lock_page_scale(
    evidences: Sequence[SymbolAnchorEvidence],
    metrics: Mapping[tuple[str, str], TemplateBoxMetric],
    *,
    zoom: float,
    tolerance: float = 0.05,
) -> PageScaleLock:
    """Lock one page scale from the most confident accepted anchor.

    ``evidences[i].bbox`` is the cell in POINTS; multiplied by ``zoom`` it becomes
    the detected cell height in page pixels at the classifier's render DPI, which
    is directly comparable to the template's measured box height in pixels.
    """
    ordered = sorted(
        (e for e in evidences if e.accepted and e.best_class is not None),
        key=lambda e: (e.best_score if e.best_score is not None else -1.0),
        reverse=True,
    )
    for e in ordered:
        box_h_px = _class_box_height_px(metrics, e.best_class)
        if box_h_px is None:
            continue
        detected_cell_px = e.bbox.height * zoom
        s = measure_scale(detected_cell_px, box_h_px)
        if s > 0:
            return PageScaleLock(
                scale=s, locked=True, tolerance=tolerance, source_anchor_id=e.anchor_id
            )
    return PageScaleLock(scale=1.0, locked=False, tolerance=tolerance)


# ==============================================================================
# Component 5b: NCC-inside-box scorer (accurate per-symbol matching, cheap)
# ==============================================================================


@dataclass
class BoxMatch:
    """Result of matching one found box crop against the template catalog."""

    box: BBox
    best_class: str | None
    best_template: str | None
    best_score: float
    second_best_score: float

    @property
    def margin(self) -> float:
        return self.best_score - self.second_best_score


def _resize_template_to_box(
    template: np.ndarray,
    template_box_height_px: float,
    target_box_height_px: float,
    *,
    max_width_px: float | None = None,
    max_height_px: float | None = None,
) -> np.ndarray | None:
    """Scale a template so its measured box height matches the crop's box height.

    This is the payoff of the measured-scale insight: instead of trying 8 blind
    scales, we resize each template to exactly the drawing's scale so a single
    NCC pass is scale-correct.

    ``max_width_px`` / ``max_height_px`` cap the result to the crop so the
    template always fits inside the matched region. This matters for tall
    (portrait) FCF cells where scaling purely by box height would make a
    landscape symbol template wider than the crop.
    """
    if template_box_height_px <= 0 or target_box_height_px <= 0:
        return None
    h, w = template.shape[:2]
    factor = target_box_height_px / template_box_height_px
    if max_width_px is not None and w * factor > max_width_px:
        factor = min(factor, max_width_px / w)
    if max_height_px is not None and h * factor > max_height_px:
        factor = min(factor, max_height_px / h)
    new_h = max(4, int(round(h * factor)))
    new_w = max(4, int(round(w * factor)))
    interp = cv2.INTER_AREA if factor < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(template, (new_w, new_h), interpolation=interp)


def _match_box_against_templates(
    crop: np.ndarray,
    templates: Sequence[TemplateDef],
    box_metrics: Mapping[tuple[str, str], TemplateBoxMetric],
    target_box_height_px: float,
    rotations: Sequence[int] = DEFAULT_MATCH_ROTATIONS,
    scales: Sequence[float] = DEFAULT_MATCH_SCALES,
) -> tuple[float, str | None, str | None, float]:
    """Return (best_score, best_class, best_template, second_best_score).

    Each template is resized to the crop's measured box height (times a small
    scale sweep), rotated, then matched with normalized cross-correlation (the
    same TM_CCOEFF_NORMED signal the old full-page detector used) over the small
    crop only. Because the crop is larger than the template, matchTemplate slides
    to find the symbol's true position; the scale sweep absorbs box-measurement
    error and symbol-to-cell fill variation.
    """
    ch, cw = crop.shape[:2]
    # Best NCC per class (so the margin is best-vs-second-best *class*, not
    # second-best template — otherwise two near-identical templates of the same
    # class collapse the margin to ~0 for a genuine detection).
    class_best: dict[str, float] = {}
    class_best_template: dict[str, str] = {}

    for tpl in templates:
        metric = box_metrics.get((tpl.class_name, tpl.template_name))
        tpl_box_h = metric.box_height_px if (metric and metric.ok) else float(tpl.image.shape[0])
        best_for_tpl = -1.0
        for angle in rotations:
            rotated = _rotate_image(tpl.image, angle) if angle else tpl.image
            # For 90/-90, the box height maps to the original width; scaling by
            # the rotated image's own height to the target keeps the box sized
            # correctly regardless of orientation.
            ref_h = tpl_box_h if angle == 0 else float(min(tpl.image.shape[:2]))
            for s in scales:
                scaled = _resize_template_to_box(
                    rotated,
                    ref_h,
                    target_box_height_px * s,
                    max_width_px=cw,
                    max_height_px=ch,
                )
                if scaled is None:
                    continue
                th, tw = scaled.shape[:2]
                if th > ch or tw > cw or th < 4 or tw < 4:
                    continue
                result = cv2.matchTemplate(crop, scaled, cv2.TM_CCOEFF_NORMED)
                score = float(result.max()) if result.size else -1.0
                if score > best_for_tpl:
                    best_for_tpl = score
        if best_for_tpl > class_best.get(tpl.class_name, -1.0):
            class_best[tpl.class_name] = best_for_tpl
            class_best_template[tpl.class_name] = tpl.template_name

    if not class_best:
        return -1.0, None, None, -1.0

    ranked = sorted(class_best.items(), key=lambda kv: kv[1], reverse=True)
    best_class, best_score = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else -1.0
    best_template = class_best_template.get(best_class)
    return best_score, best_class, best_template, second


def score_boxes_with_ncc(
    pdf_bytes: bytes,
    boxes: Sequence[BBox],
    templates: Sequence[TemplateDef],
    box_metrics: Mapping[tuple[str, str], TemplateBoxMetric],
    *,
    page_index: int = 0,
    match_dpi: int = DEFAULT_SYMBOL_DPI,
    crop_pad_pt: float = 5.0,
) -> list[BoxMatch]:
    """Score every candidate box crop against the templates via NCC.

    Renders the page once, crops each box (with a small pad), and matches every
    template resized to that box's height. The template is sized to the cell's
    measured box height (not the crop), so the crop is larger than the template
    and ``cv2.matchTemplate`` can SLIDE to find the symbol's true position even
    when the cell bbox is off by a few pixels. ``crop_pad_pt`` sets that slide
    margin (a few points is enough; larger just wastes time). Returns one
    BoxMatch per input box.

    Precision against borderless false positives (hatching / table diagonals
    matching a bare glyph) is handled at the template level: every template
    includes its FCF box border, so a bordered NCC template only scores highly
    inside a genuine bordered cell, not on open hatching.
    """
    page_gray, zoom = render_page_gray(pdf_bytes, page_index=page_index, dpi=match_dpi)
    H, W = page_gray.shape[:2]
    pad = int(round(crop_pad_pt * zoom))

    matches: list[BoxMatch] = []
    for box in boxes:
        x0 = max(0, int(round(box.x0 * zoom)) - pad)
        y0 = max(0, int(round(box.y0 * zoom)) - pad)
        x1 = min(W, int(round(box.x1 * zoom)) + pad)
        y1 = min(H, int(round(box.y1 * zoom)) + pad)
        if x1 - x0 < 4 or y1 - y0 < 4:
            matches.append(BoxMatch(box, None, None, -1.0, -1.0))
            continue
        crop = page_gray[y0:y1, x0:x1]
        target_box_h = box.height * zoom
        best, cls, tpl, second = _match_box_against_templates(
            crop, templates, box_metrics, target_box_h
        )
        matches.append(BoxMatch(box, cls, tpl, best, second))
    return matches


# ==============================================================================
# Component 1: Vector box finder (PRIMARY)
# ==============================================================================


def _is_symbol_cell(cell: BBox) -> bool:
    """A symbol (first) cell is compact and roughly square/portrait."""
    if not (MIN_FCF_HEIGHT_PT <= cell.height <= MAX_FCF_HEIGHT_PT):
        return False
    if not (MIN_CELL_WIDTH_PT <= cell.width <= MAX_SYMBOL_CELL_WIDTH_PT):
        return False
    aspect = cell.width / max(cell.height, 1e-6)
    return CELL_ASPECT_RANGE[0] <= aspect <= CELL_ASPECT_RANGE[1]


def vector_fcf_cells(
    page: "fitz.Page",
    *,
    page_number: int = 1,
) -> tuple[list[BBox], list[BBox], dict]:
    """Enumerate candidate FCF cells from vector geometry.

    Delegates FCF-row enumeration to the battle-tested
    ``vector_high_recall_proposals`` (which handles fragmented rules, endpoint
    tolerance, and multi-cell divider windows), then returns two pools:

    - ``symbol_cells``: compact, FCF-proportioned first-cell candidates that get
      classified against the templates (the anchors);
    - ``all_cells``: every FCF-height cell (including wide tolerance/datum cells),
      used as the neighbour pool so ``_grow_frame_from_anchor`` can grow the
      frame rightward.
    """
    proposals, audit = vector_high_recall_proposals(page, page_number=page_number)

    all_cells: list[BBox] = []
    for proposal in proposals:
        for cell in proposal.cell_bboxes:
            if MIN_FCF_HEIGHT_PT <= cell.height <= MAX_FCF_HEIGHT_PT:
                all_cells.append(cell)
    all_cells = _dedup_boxes(all_cells)

    symbol_cells = _dedup_boxes([c for c in all_cells if _is_symbol_cell(c)])

    audit.update(
        {
            "strategy": "vector_fcf_cells",
            "proposals": len(proposals),
            "all_cells": len(all_cells),
            "symbol_cells": len(symbol_cells),
        }
    )
    return symbol_cells, all_cells, audit


# ==============================================================================
# Component 5: Main detector
# ==============================================================================


def _to_pixel_bbox(box: BBox, zoom: float) -> tuple[int, int, int, int]:
    return (
        int(round(box.x0 * zoom)),
        int(round(box.y0 * zoom)),
        int(round(box.width * zoom)),
        int(round(box.height * zoom)),
    )


def _frame_from_chain(chain: Sequence[BBox], class_name: str, score: float) -> FcfFrame:
    """Build an FcfFrame directly from a grown cell chain (expansion fallback)."""
    cells = [
        FcfCell(x0=b.x0, y0=b.y0, x1=b.x1, y1=b.y1, index=i)
        for i, b in enumerate(chain)
    ]
    _assign_cell_roles(cells)
    return FcfFrame(
        x0=min(b.x0 for b in chain),
        y0=min(b.y0 for b in chain),
        x1=max(b.x1 for b in chain),
        y1=max(b.y1 for b in chain),
        cells=cells,
        class_name=class_name,
        detection_score=score,
    )


def _frame_iou(a: FcfFrame, b: FcfFrame) -> float:
    ix0, iy0 = max(a.x0, b.x0), max(a.y0, b.y0)
    ix1, iy1 = min(a.x1, b.x1), min(a.y1, b.y1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union > 0 else 0.0


def detect_geometry_first_frames(
    pdf_bytes: bytes,
    *,
    page_index: int = 0,
    template_root: str = "assets/gdt/templates",
    symbol_dpi: int = DEFAULT_SYMBOL_DPI,
    ncc_min_score: float = DEFAULT_NCC_MIN_SCORE,
    ncc_min_margin: float = DEFAULT_NCC_MIN_MARGIN,
    raster_fallback_dpi: int = 260,
) -> tuple[list[Detection], list[FcfFrame], dict]:
    """Detect FCFs geometry-first and return the SAME shape as analyze_page Step 1+2.

    Pipeline: find candidate FCF cells from vector + raster geometry, score each
    box crop against the templates with normalized cross-correlation (resized to
    the box's measured height, so a single scale-correct NCC pass replaces the
    old 8-scale full-page scan), accept boxes above ``ncc_min_score``, then grow
    each accepted symbol cell into a full FCF.

    Returns ``(detections, frames, audit)`` where each ``frame.detection_score``
    equals its paired ``detection.score`` (so ``_align_results`` joins them).
    """
    ncc_templates = load_templates(template_root)  # grayscale TemplateDef for NCC
    box_metrics = build_template_box_metrics(template_root)

    # --- Cell source: HYBRID (vector + raster), unioned and deduped ---
    # Vector geometry is exact but fragments thin FCF borders on some exports;
    # raster contours are robust to how the border was drawn. Taking the union
    # closes the recall gap while keeping classification cheap (a few dozen tiny
    # crops vs. the old 672 full-page correlations).
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        page = doc[page_index]
        vec_symbols, vec_pool, vec_audit = vector_fcf_cells(
            page, page_number=page_index + 1
        )

    raster_pool, raster_audit = _cell_rectangles_from_raster(
        pdf_bytes, page_index=page_index, dpi=raster_fallback_dpi
    )
    raster_symbols = [c for c in raster_pool if _is_symbol_cell(c)]

    # Union + dedup (dedup keeps the smaller box on overlap, favouring tight
    # symbol cells over lumped multi-cell contours).
    symbol_cells = _dedup_boxes([*vec_symbols, *raster_symbols])
    growth_pool = _dedup_boxes([*vec_pool, *raster_pool])

    audit: dict = {
        "strategy": "geometry_first_hybrid",
        "cell_source": "vector+raster",
        "vector_symbol_cells": len(vec_symbols),
        "raster_symbol_cells": len(raster_symbols),
        "symbol_candidate_cells": len(symbol_cells),
        "growth_pool_cells": len(growth_pool),
        "vector_audit": vec_audit,
        "raster_audit": raster_audit,
    }

    if not symbol_cells:
        audit["result"] = "no_cells"
        return [], [], audit

    # --- Score each candidate box crop with NCC at its measured scale ---
    matches = score_boxes_with_ncc(
        pdf_bytes,
        symbol_cells,
        ncc_templates,
        box_metrics,
        page_index=page_index,
        match_dpi=symbol_dpi,
    )
    accepted = []
    for m in matches:
        if not m.best_class:
            continue
        floor = _class_min_score(m.best_class, ncc_min_score)
        if m.best_score < floor:
            continue
        # Margin gate near the floor only; waive it for confident detections so a
        # real symbol is not rejected merely because a runner-up class scores
        # close.
        confident = m.best_score >= floor + DEFAULT_MARGIN_WAIVER_MARGIN
        if not confident and m.margin < ncc_min_margin:
            continue
        accepted.append(m)
    audit["candidate_matches"] = len(matches)
    audit["accepted_anchor_count"] = len(accepted)
    audit["ncc_min_score"] = ncc_min_score
    audit["ncc_min_margin"] = ncc_min_margin

    if not accepted:
        audit["result"] = "no_accepted_anchors"
        return [], [], audit

    zoom = symbol_dpi / 72.0

    # --- Grow frames + build Detection/FcfFrame pairs with a SHARED score ---
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        page = doc[page_index]
        h_lines, v_lines = extract_page_lines(page)

    page_number = page_index + 1
    detections: list[Detection] = []
    frames: list[FcfFrame] = []

    for idx, m in enumerate(sorted(accepted, key=lambda mm: mm.best_score, reverse=True), start=1):
        best_class = m.best_class
        best_score = float(m.best_score)
        symbol_box = m.box  # first-cell bbox in points

        # Grow the FCF rightward from this symbol cell through adjacent cells.
        anchor_candidate = GdtFrameCandidate(
            candidate_id=f"GDT-GEOM-P{page_number:02d}-{idx:03d}",
            page=page_number,
            frame_bbox=symbol_box,
            symbol_bbox=symbol_box,
            cells=[GdtCell(bbox=symbol_box)],
            confidence_score=best_score,
        )
        chain = _grow_frame_from_anchor(anchor_candidate, growth_pool)
        # A confidently classified symbol is a valid detection even if the frame
        # cannot be grown to multiple cells (some exports fragment the tolerance
        # cells, or the symbol stands alone). Fall back to a single-cell chain so
        # we never drop a genuine symbol at the growth stage.
        if len(chain) < 1:
            chain = [symbol_box]

        first = chain[0]
        # Measured scale from this box vs its matched template's box height.
        class_box_h = _class_box_height_px(box_metrics, best_class)
        measured = (
            measure_scale(first.height * zoom, class_box_h)
            if class_box_h
            else 1.0
        )
        det = Detection(
            class_name=best_class or "",
            template_name=m.best_template or best_class or "",
            score=best_score,
            x=first.x0,
            y=first.y0,
            width=first.width,
            height=first.height,
            scale=measured,
            rotation=0,
            pixel_bbox=_to_pixel_bbox(first, zoom),
        )

        frame = expand_fcf_from_detection(
            first.x0,
            first.y0,
            first.width,
            first.height,
            h_lines,
            v_lines,
            class_name=best_class or "",
            detection_score=best_score,
        )
        if frame is None:
            frame = _frame_from_chain(chain, best_class or "", best_score)
        else:
            _assign_cell_roles(frame.cells)

        detections.append(det)
        frames.append(frame)

    # --- Deduplicate paired detections/frames by frame overlap ---
    order = sorted(range(len(frames)), key=lambda i: detections[i].score, reverse=True)
    kept_det: list[Detection] = []
    kept_frame: list[FcfFrame] = []
    for i in order:
        if any(_frame_iou(frames[i], kf) >= 0.55 for kf in kept_frame):
            continue
        kept_det.append(detections[i])
        kept_frame.append(frames[i])

    # Stable, human-friendly ordering (top-to-bottom, left-to-right).
    paired = sorted(zip(kept_det, kept_frame), key=lambda p: (p[1].y0, p[1].x0))
    detections = [d for d, _ in paired]
    frames = [f for _, f in paired]

    audit["frame_count"] = len(frames)
    audit["result"] = "ok"
    return detections, frames, audit


__all__ = [
    "TemplateBoxMetric",
    "PageScaleLock",
    "measure_template_box",
    "build_template_box_metrics",
    "measure_scale",
    "lock_page_scale",
    "vector_fcf_cells",
    "detect_geometry_first_frames",
]
