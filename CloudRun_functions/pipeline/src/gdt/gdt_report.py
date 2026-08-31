"""GD&T constraint report generation and visualization.

Ties together the full pipeline:
1. Template detection (symbol identification)
2. FCF expansion (full box extraction with cell segmentation)
3. Datum cell extraction (datum reference identification in FCF)
4. Datum definition finding (standalone letters on the drawing)

Produces:
- A structured JSON report listing all constraints with type, location,
  cell structure, and datum references
- An annotated image showing FCF frames, symbol labels, and datum definitions
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.gdt.datum_extractor import FcfExtraction, extract_datum_cells
from src.gdt.datum_consistency import assess_referenced_datum_definitions
from src.gdt.datum_feature import detect_document_datum_feature_indicators
from src.gdt.datum_finder import DatumDefinition, find_datum_definitions
from src.gdt.datum_text import extract_datum_text_candidates
from src.gdt.datum_visual_resolver import resolve_outlined_datum_references
from src.gdt.fcf_expander import FcfFrame, expand_detections_to_fcf
from src.gdt.geometry_first_detector import detect_geometry_first_frames
from src.gdt.template_detector import (
    Detection,
    GdtTemplateDetector,
    render_page_gray,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class GdtConstraint:
    """A single GD&T constraint found in the drawing."""

    class_name: str
    detection_score: float
    symbol_pt: Tuple[float, float, float, float]  # (x0, y0, x1, y1)
    frame_pt: Optional[Tuple[float, float, float, float]] = None
    cell_count: int = 0
    datum_count: int = 0
    datum_refs: List[dict] = field(default_factory=list)
    referenced_datums: List[str] = field(default_factory=list)
    unresolved_datum_ref_count: int = 0
    datum_definition_findings: List[dict] = field(default_factory=list)
    scale: float = 0.0
    rotation: int = 0

    def to_dict(self) -> dict:
        return {
            "class_name": self.class_name,
            "detection_score": round(self.detection_score, 4),
            "symbol_bbox_pt": [round(v, 1) for v in self.symbol_pt],
            "frame_bbox_pt": [round(v, 1) for v in self.frame_pt] if self.frame_pt else None,
            "cell_count": self.cell_count,
            "datum_count": self.datum_count,
            "datum_refs": self.datum_refs,
            "referenced_datums": self.referenced_datums,
            "unresolved_datum_ref_count": self.unresolved_datum_ref_count,
            "datum_definition_findings": self.datum_definition_findings,
            "has_datums": self.datum_count > 0,
            "scale": round(self.scale, 3),
            "rotation": self.rotation,
        }


@dataclass
class GdtPageReport:
    """Complete GD&T analysis report for one page."""

    pdf_name: str
    page_index: int
    constraints: List[GdtConstraint] = field(default_factory=list)
    datum_definitions: List[dict] = field(default_factory=list)
    datum_box_candidates: List[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "pdf_name": self.pdf_name,
            "page": self.page_index + 1,
            "summary": self.summary,
            "constraints": [c.to_dict() for c in self.constraints],
            "datum_definitions": self.datum_definitions,
            "datum_box_candidates": self.datum_box_candidates,
        }


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------


def analyze_page(
    pdf_bytes: bytes,
    *,
    page_index: int = 0,
    template_root: str = "assets/gdt/templates",
    dpi: int = 150,
    score_threshold: float = 0.74,
    scales: Optional[List[float]] = None,
    rotations: Optional[List[int]] = None,
    pdf_name: str = "",
    max_workers: int = 8,
) -> Tuple[GdtPageReport, List[Detection], List[FcfFrame], List[FcfExtraction], List[DatumDefinition]]:
    """Run the full GD&T analysis pipeline on a single page.

    Returns the report plus intermediate results for visualization.
    """
    if scales is None:
        scales = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0, 1.1]
    if rotations is None:
        rotations = [0, 90, -90]

    # Steps 1+2: Geometry-first detection.
    #
    # Replaces the old full-page multi-scale/multi-rotation template scan
    # (~672 correlations/page) with a geometry-first detector: find candidate
    # FCF boxes from vector + raster geometry, then score each box crop with
    # normalized cross-correlation resized to the box's measured height (a single
    # scale-correct pass), and grow each accepted symbol cell into a full FCF.
    # Returns the same Detection + FcfFrame shapes the old Step 1 + Step 2 did,
    # with frame.detection_score == det.score so _align_results still joins them.
    #
    # NOTE: `scales`, `rotations`, `score_threshold`, and `max_workers` are
    # retained for signature compatibility but no longer drive the primary path
    # (there is no full-page multi-scale scan). `dpi` still sets the render DPI
    # used for NCC scoring (clamped to a sensible minimum inside the detector).
    detections, frames, _gdt_audit = detect_geometry_first_frames(
        pdf_bytes,
        page_index=page_index,
        template_root=template_root,
        symbol_dpi=max(dpi, 300),
    )

    # Step 3: Render at 300 DPI for extraction and datum finding
    extraction_dpi = max(dpi, 300)
    page_gray, zoom = render_page_gray(pdf_bytes, page_index=page_index, dpi=extraction_dpi)

    # Step 4: Resolve vector/invisible datum letters inside FCF datum cells.
    text_candidates = extract_datum_text_candidates(pdf_bytes, page_index=page_index)
    extractions = extract_datum_cells(
        page_gray,
        zoom,
        frames,
        text_candidates=text_candidates,
    )

    # Step 5: Find datum definitions (geometric box detection + ink check)
    datum_defs = find_datum_definitions(
        pdf_bytes, page_gray, zoom, frames, page_index=page_index
    )

    # Step 5b: Find high-confidence datum feature indicators across the entire
    # PDF.  A reference on page 1 may be defined on a different drawing page.
    verified_datum_defs = detect_document_datum_feature_indicators(
        pdf_bytes,
        raster_dpi=200,
    )

    # Step 5c: Some CAD generators convert the letters inside FCFs to vector
    # outlines. Learn labelled glyph shapes from PDF text elsewhere in the
    # drawing and resolve those cells before checking definition consistency.
    visual_resolution = resolve_outlined_datum_references(
        pdf_bytes,
        extractions,
        verified_datum_defs,
        dpi=extraction_dpi,
    )

    # Step 6: Build constraints
    constraints: List[GdtConstraint] = []
    for det, frame, extraction in _align_results(detections, frames, extractions):
        datum_refs_dicts = []
        datum_count = 0
        referenced_datums: List[str] = []
        frame_bbox = None
        cell_count = 0

        if frame is not None:
            frame_bbox = (frame.x0, frame.y0, frame.x1, frame.y1)
            cell_count = frame.cell_count

        if extraction is not None:
            for dref in extraction.datum_refs:
                if dref.has_content:
                    datum_count += 1
                    datum_refs_dicts.append(dref.to_dict())
                    if dref.text:
                        referenced_datums.append(dref.text)

        definition_findings = assess_referenced_datum_definitions(
            referenced_datums=referenced_datums,
            defined_indicators=verified_datum_defs,
        )

        constraints.append(GdtConstraint(
            class_name=det.class_name,
            detection_score=det.score,
            symbol_pt=(det.x, det.y, det.x + det.width, det.y + det.height),
            frame_pt=frame_bbox,
            cell_count=cell_count,
            datum_count=datum_count,
            datum_refs=datum_refs_dicts,
            referenced_datums=referenced_datums,
            unresolved_datum_ref_count=max(0, datum_count - len(referenced_datums)),
            datum_definition_findings=[row.to_dict() for row in definition_findings],
            scale=det.scale,
            rotation=det.rotation,
        ))

    # Build report
    report = GdtPageReport(
        pdf_name=pdf_name,
        page_index=page_index,
        constraints=constraints,
        datum_definitions=[d.to_dict() for d in verified_datum_defs],
        datum_box_candidates=[d.to_dict() for d in datum_defs],
        summary={
            "total_detections": len(detections),
            "fcf_frames_expanded": len(frames),
            "constraints_with_datums": sum(1 for c in constraints if c.datum_count > 0),
            "total_datum_refs": sum(c.datum_count for c in constraints),
            "resolved_datum_refs": sum(len(c.referenced_datums) for c in constraints),
            "unresolved_datum_refs": sum(c.unresolved_datum_ref_count for c in constraints),
            "datum_definitions_found": len(verified_datum_defs),
            "datum_box_candidates_found": len(datum_defs),
            "undefined_referenced_datums": sum(
                1
                for constraint in constraints
                for finding in constraint.datum_definition_findings
                if finding["code"] == "ISO5459_REFERENCED_DATUM_NOT_DEFINED"
            ),
            "datum_visual_templates": visual_resolution.template_count,
            "datum_visual_template_labels": list(visual_resolution.template_labels),
            "datum_refs_resolved_by_visual_template": visual_resolution.resolved_count,
            "datum_visual_matches_rejected": visual_resolution.rejected_count,
            "datum_cells_reclassified_empty": visual_resolution.empty_cell_count,
            "constraint_types": _count_types(constraints),
        },
    )

    return report, detections, frames, extractions, datum_defs


def _align_results(
    detections: List[Detection],
    frames: List[FcfFrame],
    extractions: List[FcfExtraction],
) -> List[Tuple[Detection, Optional[FcfFrame], Optional[FcfExtraction]]]:
    """Align detections with their corresponding frames and extractions."""
    frame_by_det = {}
    ext_by_det = {}

    for frame, ext in zip(frames, extractions):
        key = (frame.class_name, round(frame.detection_score, 4))
        frame_by_det[key] = frame
        ext_by_det[key] = ext

    results = []
    for det in detections:
        key = (det.class_name, round(det.score, 4))
        frame = frame_by_det.get(key)
        ext = ext_by_det.get(key)
        results.append((det, frame, ext))

    return results


def _count_types(constraints: List[GdtConstraint]) -> dict:
    counts: dict[str, int] = {}
    for c in constraints:
        counts[c.class_name] = counts.get(c.class_name, 0) + 1
    return dict(sorted(counts.items()))


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

TYPE_COLORS = {
    "position": (0, 180, 0),
    "profile": (180, 0, 180),
    "perpendicularity": (255, 0, 0),
    "parallelism": (0, 200, 200),
    "angularity": (0, 128, 255),
    "circularity": (255, 255, 0),
    "cylindricity": (200, 100, 0),
    "flatness": (0, 255, 255),
    "straightness": (128, 128, 0),
    "circular_runout": (100, 0, 200),
    "total_runout": (50, 0, 150),
    "concentricity_coaxiality": (0, 100, 200),
    "symmetry": (200, 200, 0),
}

FCF_FRAME_COLOR = (0, 200, 0)
DATUM_DEF_COLOR = (0, 0, 220)  # red for datum definitions


def render_annotated_page(
    pdf_bytes: bytes,
    detections: List[Detection],
    frames: List[FcfFrame],
    extractions: List[FcfExtraction],
    datum_defs: List[DatumDefinition],
    *,
    page_index: int = 0,
    dpi: int = 150,
    page_gray: np.ndarray | None = None,
    zoom: float | None = None,
    verified_datum_defs: Optional[List[dict]] = None,
) -> np.ndarray:
    """Render the page with GD&T annotations overlaid.

    If page_gray/zoom are provided, uses them directly (avoids re-rendering).
    """
    if page_gray is None or zoom is None:
        page_gray, zoom = render_page_gray(pdf_bytes, page_index=page_index, dpi=dpi)
    page_bgr = cv2.cvtColor(page_gray, cv2.COLOR_GRAY2BGR)

    # Map each frame to its extraction so we only highlight datum cells that
    # actually contain a letter (empty cells must not be highlighted).
    ext_by_frame = {id(ext.frame): ext for ext in extractions}

    # Draw FCF frames (green outlines)
    for frame in frames:
        px0 = int(frame.x0 * zoom)
        py0 = int(frame.y0 * zoom)
        px1 = int(frame.x1 * zoom)
        py1 = int(frame.y1 * zoom)
        cv2.rectangle(page_bgr, (px0, py0), (px1, py1), FCF_FRAME_COLOR, 2)

        # Highlight only datum cells with real content (a datum letter)
        ext = ext_by_frame.get(id(frame))
        if ext is None:
            continue
        for dref in ext.datum_refs:
            if not dref.has_content:
                continue  # empty cell — not a datum reference, skip
            cell = dref.cell
            cx0 = int(cell.x0 * zoom)
            cy0 = int(cell.y0 * zoom)
            cx1 = int(cell.x1 * zoom)
            cy1 = int(cell.y1 * zoom)
            if cy1 > cy0 and cx1 > cx0:
                overlay = page_bgr[cy0:cy1, cx0:cx1].copy()
                cv2.rectangle(overlay, (0, 0), (cx1 - cx0, cy1 - cy0), (0, 165, 255), -1)
                page_bgr[cy0:cy1, cx0:cx1] = cv2.addWeighted(
                    page_bgr[cy0:cy1, cx0:cx1], 0.7, overlay, 0.3, 0
                )
                if dref.text:
                    cv2.putText(
                        page_bgr,
                        f"REF {dref.text}",
                        (cx0, max(cy0 - 3, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        (0, 120, 220),
                        1,
                        cv2.LINE_AA,
                    )

    # Draw constraint type labels
    for det in detections:
        color = TYPE_COLORS.get(det.class_name, (128, 128, 128))
        px0 = int(det.x * zoom)
        py0 = int(det.y * zoom)
        label = f"{det.class_name} {det.score:.2f}"
        label_y = max(py0 - 4, 10)
        cv2.putText(
            page_bgr, label, (px0, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA,
        )

    # Draw geometry-only candidates retained for diagnostics.
    for datum_def in datum_defs:
        cx = int(datum_def.x * zoom)
        cy = int(datum_def.y * zoom)
        hw = int(datum_def.width * zoom / 2) + 3
        hh = int(datum_def.height * zoom / 2) + 3
        cv2.rectangle(page_bgr, (cx - hw, cy - hh), (cx + hw, cy + hh), DATUM_DEF_COLOR, 2)
        cv2.putText(
            page_bgr, "DATUM?",
            (cx + hw + 3, cy + 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, DATUM_DEF_COLOR, 1, cv2.LINE_AA,
        )

    # Draw verified datum-feature indicators (letter + box + stem + marker).
    for datum_def in verified_datum_defs or []:
        if int(datum_def.get("page", 0)) != page_index + 1:
            continue
        box = datum_def.get("box_bbox")
        if not isinstance(box, list) or len(box) != 4:
            continue
        x0, y0, x1, y1 = (int(float(value) * zoom) for value in box)
        cv2.rectangle(page_bgr, (x0, y0), (x1, y1), DATUM_DEF_COLOR, 2)
        cv2.putText(
            page_bgr,
            f"DATUM {datum_def.get('label', '?')}",
            (x1 + 3, y0 + max(10, (y1 - y0) // 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            DATUM_DEF_COLOR,
            1,
            cv2.LINE_AA,
        )

    return page_bgr


# ---------------------------------------------------------------------------
# File output
# ---------------------------------------------------------------------------


def save_report(report: GdtPageReport, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "gdt_report.json"
    path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def save_visualization(image: np.ndarray, output_dir: Path, *, page_index: int = 0) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"page_{page_index + 1:03d}_gdt_analysis.png"
    cv2.imwrite(str(path), image)
    return path


__all__ = [
    "GdtConstraint",
    "GdtPageReport",
    "analyze_page",
    "render_annotated_page",
    "save_report",
    "save_visualization",
]
