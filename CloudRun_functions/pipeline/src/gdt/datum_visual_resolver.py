"""Resolve outlined datum letters with templates learned from the PDF itself.

CAD exports frequently keep ordinary text as PDF text while converting the
letters inside feature-control frames to paths.  This module bridges those two
representations without OCR: labelled single-letter text elsewhere in the
drawing becomes a raster glyph template, and an unresolved datum-cell glyph is
matched against those templates deterministically.

Matches are deliberately rejected unless both the absolute score and the gap
to the runner-up are strong.  An unknown letter therefore remains unresolved
instead of being forced to the nearest known datum.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import cv2
import fitz
import numpy as np

from src.gdt.datum_extractor import FcfExtraction
from src.gdt.datum_feature import DatumFeatureIndicatorCandidate
from src.gdt.datum_glyph import DatumGlyphTemplateClassifier, normalize_glyph_mask
from src.gdt.datum_text import DatumTextCandidate, extract_datum_text_candidates


@dataclass(frozen=True)
class DatumVisualResolutionStats:
    template_count: int
    template_labels: tuple[str, ...]
    resolved_count: int
    rejected_count: int
    empty_cell_count: int


def _render_page_gray(page: fitz.Page, dpi: int) -> tuple[np.ndarray, float]:
    zoom = float(dpi) / 72.0
    pix = page.get_pixmap(
        matrix=fitz.Matrix(zoom, zoom),
        colorspace=fitz.csGRAY,
        alpha=False,
    )
    gray = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width).copy()
    return gray, zoom


def _clean_normalized_mask(
    gray: np.ndarray,
    *,
    edge_margin_px: int = 0,
    min_component_area: int = 6,
) -> tuple[np.ndarray, int]:
    """Remove cell borders/edge geometry and normalize the remaining glyph."""

    if gray.size == 0:
        return normalize_glyph_mask(np.zeros((1, 1), dtype=np.uint8)), 0

    margin = max(0, int(edge_margin_px))
    if margin and gray.shape[0] > 2 * margin and gray.shape[1] > 2 * margin:
        gray = gray[margin:-margin, margin:-margin]

    _threshold, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
    )
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, 8)
    cleaned = np.zeros_like(binary)
    height, width = binary.shape
    for label in range(1, count):
        x, y, component_width, component_height, area = [int(value) for value in stats[label]]
        touches_edge = (
            x <= 0
            or y <= 0
            or x + component_width >= width
            or y + component_height >= height
        )
        if area >= min_component_area and not touches_edge:
            cleaned[labels == label] = 255

    ink_pixels = int(np.count_nonzero(cleaned))
    return normalize_glyph_mask(cleaned), ink_pixels


def _crop_bbox(
    gray: np.ndarray,
    zoom: float,
    bbox: tuple[float, float, float, float],
    *,
    padding_pt: float = 0.0,
) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    x0 -= padding_pt
    y0 -= padding_pt
    x1 += padding_pt
    y1 += padding_pt
    height, width = gray.shape
    px0 = max(0, int(math.floor(x0 * zoom)))
    py0 = max(0, int(math.floor(y0 * zoom)))
    px1 = min(width, int(math.ceil(x1 * zoom)))
    py1 = min(height, int(math.ceil(y1 * zoom)))
    if px1 <= px0 or py1 <= py0:
        return np.empty((0, 0), dtype=np.uint8)
    return gray[py0:py1, px0:px1]


def _register_text_template(
    classifier: DatumGlyphTemplateClassifier,
    gray: np.ndarray,
    zoom: float,
    candidate: DatumTextCandidate,
    *,
    source_id: str,
) -> bool:
    # PDF font bboxes can touch the actual outline. A small pad keeps the
    # outline away from the crop edge before edge-connected noise is removed.
    crop = _crop_bbox(gray, zoom, candidate.bbox, padding_pt=0.6)
    normalized, ink_pixels = _clean_normalized_mask(crop)
    if ink_pixels < 6:
        return False
    classifier.register(candidate.label, normalized, source_id=source_id)
    return True


def resolve_outlined_datum_references(
    pdf_bytes: bytes,
    extractions: Iterable[FcfExtraction],
    verified_definitions: Iterable[DatumFeatureIndicatorCandidate],
    *,
    dpi: int = 300,
    min_score: float = 0.72,
    min_margin: float = 0.12,
    max_templates_per_label: int = 12,
) -> DatumVisualResolutionStats:
    """Populate unresolved datum refs by matching outlined letter geometry.

    Template labels come from the PDF text coordinate matrix. Verified datum
    definitions are registered first, followed by other isolated uppercase
    text tokens in the document. This allows a reference to be read before the
    consistency check determines whether its datum is actually defined.
    """

    classifier = DatumGlyphTemplateClassifier()
    definition_rows = list(verified_definitions)
    definitions_by_page: dict[int, list[DatumFeatureIndicatorCandidate]] = {}
    for definition in definition_rows:
        definitions_by_page.setdefault(definition.page - 1, []).append(definition)

    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    registered_per_label: dict[str, int] = {}
    try:
        for page_index in range(document.page_count):
            candidates = extract_datum_text_candidates(pdf_bytes, page_index=page_index)
            if not candidates and page_index not in definitions_by_page:
                continue
            gray, zoom = _render_page_gray(document[page_index], dpi)

            # Datum-definition letters are especially relevant templates.
            for definition in definitions_by_page.get(page_index, []):
                if registered_per_label.get(definition.label, 0) >= max_templates_per_label:
                    continue
                candidate = DatumTextCandidate(
                    label=definition.label,
                    page=definition.page,
                    bbox=definition.text_bbox,
                    source="verified_datum_definition",
                    confidence=1.0,
                    invisible=False,
                    bbox_quality="verified",
                )
                if _register_text_template(
                    classifier,
                    gray,
                    zoom,
                    candidate,
                    source_id=f"datum_definition:p{definition.page}:{definition.label}",
                ):
                    registered_per_label[definition.label] = (
                        registered_per_label.get(definition.label, 0) + 1
                    )

            for candidate_index, candidate in enumerate(candidates):
                if registered_per_label.get(candidate.label, 0) >= max_templates_per_label:
                    continue
                if _register_text_template(
                    classifier,
                    gray,
                    zoom,
                    candidate,
                    source_id=(
                        f"pdf_text:p{page_index + 1}:{candidate_index}:{candidate.source}"
                    ),
                ):
                    registered_per_label[candidate.label] = (
                        registered_per_label.get(candidate.label, 0) + 1
                    )
    finally:
        document.close()

    resolved_count = 0
    rejected_count = 0
    empty_cell_count = 0
    edge_margin_px = max(1, int(round(1.1 * float(dpi) / 72.0)))
    for extraction in extractions:
        for datum_ref in extraction.datum_refs:
            if datum_ref.text or not datum_ref.has_content or datum_ref.crop is None:
                continue
            normalized, ink_pixels = _clean_normalized_mask(
                datum_ref.crop,
                edge_margin_px=edge_margin_px,
            )
            if ink_pixels < 6:
                # The first-pass ink ratio can include a clipped frame border.
                # Once edge-connected geometry is removed, this is an empty
                # parser cell rather than an unresolved datum reference.
                datum_ref.ink_ratio = 0.0
                datum_ref.glyph = None
                empty_cell_count += 1
                continue
            ranking = classifier.rank(normalized)
            if not ranking:
                rejected_count += 1
                continue
            best = ranking[0]
            runner_up_score = ranking[1].score if len(ranking) > 1 else 0.0
            margin = best.score - runner_up_score
            if best.score < min_score or margin < min_margin:
                rejected_count += 1
                continue

            datum_ref.text = best.label
            datum_ref.confidence = best.score
            datum_ref.text_source = "visual_template_from_pdf_text"
            datum_ref.visual_match_margin = margin
            datum_ref.visual_template_source = best.source_id
            resolved_count += 1

    return DatumVisualResolutionStats(
        template_count=classifier.template_count,
        template_labels=classifier.labels,
        resolved_count=resolved_count,
        rejected_count=rejected_count,
        empty_cell_count=empty_cell_count,
    )


__all__ = [
    "DatumVisualResolutionStats",
    "resolve_outlined_datum_references",
]
