"""API estável para detecção geométrica de quadros GD&T.

O frame externo continua sendo detectado pela implementação legada já validada.
Depois disso, este módulo refaz SOMENTE a segmentação interna das células com
uma tolerância de endpoint mais rígida.

Motivo: a tolerância permissiva necessária para reconstruir o frame externo
(4.5 pt) pode interpretar traços do próprio símbolo — especialmente a linha
vertical do símbolo Position — como divisórias de célula. Isso não altera o
``frame_bbox``, mas pode cortar o ``symbol_bbox`` pela metade.
"""

from __future__ import annotations

from typing import List

import fitz

from src.utils.gdt_detector import (
    BBox,
    GdtCell,
    GdtFrameCandidate,
    GdtFrameDetector as _LegacyGdtFrameDetector,
    VSegment,
)


class GdtFrameDetector(_LegacyGdtFrameDetector):
    """Detector estável com refinamento estrito das divisórias internas."""

    def __init__(self, *args, cell_endpoint_tolerance: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.cell_endpoint_tolerance = float(cell_endpoint_tolerance)

    def _refine_cells_for_frame(
        self,
        frame_bbox: BBox,
        v_lines: List[VSegment],
        original_cells: List[GdtCell],
    ) -> List[GdtCell]:
        """Ressegmenta células sem permitir que traços internos virem divisórias.

        O ``frame_bbox`` é mantido exatamente como saiu da detecção aprovada na
        Fase 1. Somente verticais que praticamente atravessam toda a altura do
        frame são aceitas como divisórias internas.
        """

        tol = self.cell_endpoint_tolerance
        x0, x1 = frame_bbox.x0, frame_bbox.x1
        y0, y1 = frame_bbox.y0, frame_bbox.y1

        internal_x: List[float] = []
        for line in sorted(v_lines, key=lambda item: item.x):
            # Divisórias internas precisam deixar células minimamente plausíveis
            # nos dois lados. As bordas externas vêm do próprio frame_bbox.
            if line.x <= x0 + self.min_cell_width:
                continue
            if line.x >= x1 - self.min_cell_width:
                continue

            # Critério muito mais rígido que o usado para reconstruir o frame.
            if line.y0 > y0 + tol:
                continue
            if line.y1 < y1 - tol:
                continue

            if internal_x and line.x - internal_x[-1] < self.min_cell_width:
                continue
            internal_x.append(line.x)

        boundaries = [x0] + internal_x + [x1]

        # Remove uma eventual divisória final que deixaria a última célula fina.
        while len(boundaries) > 2 and boundaries[-1] - boundaries[-2] < self.min_cell_width:
            boundaries.pop(-2)

        cells: List[GdtCell] = []
        for left, right in zip(boundaries, boundaries[1:]):
            if right - left < self.min_cell_width:
                continue
            cells.append(GdtCell(bbox=BBox(left, y0, right, y1)))

        # O refinamento não pode destruir um candidato já aprovado geometricamente.
        # Se a segmentação estrita não formar um frame plausível, preserva o legado.
        if not (self.min_cells <= len(cells) <= self.max_cells):
            return original_cells

        return cells

    def detect_frames(self, pdf_bytes: bytes, page_index: int = 0) -> List[GdtFrameCandidate]:
        candidates = super().detect_frames(pdf_bytes, page_index=page_index)
        if not candidates:
            return candidates

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            page = doc[page_index]
            _h_raw, v_raw = self._extract_line_segments(page)
            v_merged = self._merge_vertical(v_raw)
            words = page.get_text("words")

            for candidate in candidates:
                refined = self._refine_cells_for_frame(
                    candidate.frame_bbox,
                    v_merged,
                    candidate.cells,
                )

                for cell in refined:
                    cell.texts = self._words_in_cell(words, cell.bbox)

                candidate.cells = refined
                candidate.symbol_bbox = refined[0].bbox
                candidate.symbol_crop = self._crop_from_page_image(candidate.symbol_bbox)
                candidate.confidence_score = self._compute_confidence(refined)

            return candidates
        finally:
            doc.close()


def detect_gdt_frames(
    pdf_bytes: bytes,
    page_index: int = 0,
    **detector_kwargs,
):
    """Detecta candidatos usando a API estável e devolve também o debug image."""

    detector = GdtFrameDetector(**detector_kwargs)
    candidates = detector.detect_frames(pdf_bytes, page_index=page_index)
    return candidates, detector.render_debug_image(candidates)


__all__ = [
    "BBox",
    "GdtCell",
    "GdtFrameCandidate",
    "GdtFrameDetector",
    "detect_gdt_frames",
]
