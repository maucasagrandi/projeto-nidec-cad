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
from src.gdt.datum_finder import DatumDefinition, find_datum_definitions
from src.gdt.fcf_expander import FcfFrame, expand_detections_to_fcf
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
    summary: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "pdf_name": self.pdf_name,
            "page": self.page_index + 1,
            "summary": self.summary,
            "constraints": [c.to_dict() for c in self.constraints],
            "datum_definitions": self.datum_definitions,
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

    # Step 1: Detect GD&T symbols (parallel template matching)
    # Detection uses its own DPI (lower = faster). The extraction render is separate.
    detector = GdtTemplateDetector(
        template_root=template_root,
        dpi=dpi,
        scales=scales,
        rotations=rotations,
        score_threshold=score_threshold,
        max_workers=max_workers,
    )
    detections = detector.detect(pdf_bytes, page_index=page_index)

    # Step 2: Expand detections to full FCF frames (vector lines from PDF)
    frames = expand_detections_to_fcf(pdf_bytes, detections, page_index=page_index)

    # Step 3: Render at 300 DPI for extraction and datum finding
    extraction_dpi = max(dpi, 300)
    page_gray, zoom = render_page_gray(pdf_bytes, page_index=page_index, dpi=extraction_dpi)

    # Step 4: Extract datum cell content
    extractions = extract_datum_cells(page_gray, zoom, frames)

    # Step 5: Find datum definitions (geometric box detection + ink check)
    datum_defs = find_datum_definitions(
        pdf_bytes, page_gray, zoom, frames, page_index=page_index
    )

    # Step 6: Build constraints
    constraints: List[GdtConstraint] = []
    for det, frame, extraction in _align_results(detections, frames, extractions):
        datum_refs_dicts = []
        datum_count = 0
        frame_bbox = None
        cell_count = 0

        if frame is not None:
            frame_bbox = (frame.x0, frame.y0, frame.x1, frame.y1)
            cell_count = frame.cell_count

        if extraction is not None:
            for dref in extraction.datum_refs:
                if dref.ink_ratio > 0.05:
                    datum_count += 1
                    datum_refs_dicts.append(dref.to_dict())

        constraints.append(GdtConstraint(
            class_name=det.class_name,
            detection_score=det.score,
            symbol_pt=(det.x, det.y, det.x + det.width, det.y + det.height),
            frame_pt=frame_bbox,
            cell_count=cell_count,
            datum_count=datum_count,
            datum_refs=datum_refs_dicts,
            scale=det.scale,
            rotation=det.rotation,
        ))

    # Build report
    report = GdtPageReport(
        pdf_name=pdf_name,
        page_index=page_index,
        constraints=constraints,
        datum_definitions=[d.to_dict() for d in datum_defs],
        summary={
            "total_detections": len(detections),
            "fcf_frames_expanded": len(frames),
            "constraints_with_datums": sum(1 for c in constraints if c.datum_count > 0),
            "total_datum_refs": sum(c.datum_count for c in constraints),
            "datum_definitions_found": len(datum_defs),
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
            if dref.ink_ratio <= 0.05:
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

    # Draw datum definitions (red circles marking standalone datum boxes)
    for datum_def in datum_defs:
        cx = int(datum_def.x * zoom)
        cy = int(datum_def.y * zoom)
        hw = int(datum_def.width * zoom / 2) + 3
        hh = int(datum_def.height * zoom / 2) + 3
        cv2.rectangle(page_bgr, (cx - hw, cy - hh), (cx + hw, cy + hh), DATUM_DEF_COLOR, 2)
        cv2.putText(
            page_bgr, "DATUM",
            (cx + hw + 3, cy + 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, DATUM_DEF_COLOR, 1, cv2.LINE_AA,
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
