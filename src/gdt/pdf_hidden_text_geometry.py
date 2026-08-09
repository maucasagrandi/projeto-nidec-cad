"""Geometria estimada da camada textual invisível de PDFs CAD.

Esta camada é deliberadamente separada de ``pdf_hidden_text.py``. O parser
estável continua responsável por reconstruir os operadores de texto; aqui
usamos o estado interno dele no instante de cada ``Tj/TJ`` para estimar a área
ocupada pelo texto.

A estimativa usa, quando possível, a própria fonte embutida no PDF:
- ``Font.text_length`` para largura;
- ``Font.ascender`` / ``Font.descender`` para altura em torno da baseline;
- text matrix + CTM para levar o retângulo local ao espaço PDF;
- ``page.transformation_matrix`` para produzir também a versão no espaço de
  página do PyMuPDF.

Isto ainda NÃO é uma bbox oficial fornecida pelo PDF. É uma estimativa baseada
em métricas de fonte, usada primeiro como diagnóstico da Fase 5.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import fitz

from src.gdt.pdf_hidden_text import PdfContentTextParser, PdfTextEvent


Point = tuple[float, float]
MatrixTuple = tuple[float, float, float, float, float, float]
QuadTuple = tuple[Point, Point, Point, Point]
BBoxTuple = tuple[float, float, float, float]


def _matrix_tuple(matrix: fitz.Matrix) -> MatrixTuple:
    return (
        float(matrix.a),
        float(matrix.b),
        float(matrix.c),
        float(matrix.d),
        float(matrix.e),
        float(matrix.f),
    )


def _point_tuple(point: fitz.Point) -> Point:
    return float(point.x), float(point.y)


def _quad_bbox(quad: QuadTuple) -> BBoxTuple:
    xs = [point[0] for point in quad]
    ys = [point[1] for point in quad]
    return min(xs), min(ys), max(xs), max(ys)


def _transform_quad(quad: QuadTuple, matrix: fitz.Matrix) -> QuadTuple:
    return tuple(  # type: ignore[return-value]
        _point_tuple(fitz.Point(point[0], point[1]) * matrix)
        for point in quad
    )


@dataclass(frozen=True)
class ResolvedFontMetrics:
    resource_name: str
    xref: Optional[int]
    basefont: Optional[str]
    encoding: Optional[str]
    source: str
    font: fitz.Font


class PdfFontMetricsResolver:
    """Resolve o resource name usado por ``Tf`` para métricas de fonte."""

    def __init__(self, doc: fitz.Document, page_index: int = 0) -> None:
        self.doc = doc
        self.page_index = int(page_index)
        self._by_resource: dict[str, ResolvedFontMetrics] = {}
        self._build()

    @staticmethod
    def _fallback_font(basefont: Optional[str]) -> fitz.Font:
        name = (basefont or "").lower()
        if "courier" in name:
            return fitz.Font("cour")
        if "times" in name:
            return fitz.Font("tiro")
        return fitz.Font("helv")

    def _build(self) -> None:
        page = self.doc[self.page_index]
        for row in page.get_fonts(full=True):
            if len(row) < 6:
                continue

            xref = int(row[0]) if row[0] else None
            basefont = str(row[3]) if row[3] else None
            resource_name = str(row[4]) if row[4] else ""
            encoding = str(row[5]) if row[5] else None
            if not resource_name:
                continue

            font: Optional[fitz.Font] = None
            source = "fallback_base14"

            if xref:
                try:
                    _basename, _ext, _font_type, content = self.doc.extract_font(xref)
                    if content:
                        font = fitz.Font(fontbuffer=content)
                        source = "embedded_font_buffer"
                except Exception:
                    font = None

            if font is None:
                font = self._fallback_font(basefont)

            self._by_resource[resource_name] = ResolvedFontMetrics(
                resource_name=resource_name,
                xref=xref,
                basefont=basefont,
                encoding=encoding,
                source=source,
                font=font,
            )

    def resolve(self, resource_name: Optional[str]) -> Optional[ResolvedFontMetrics]:
        if not resource_name:
            return None
        return self._by_resource.get(resource_name)

    def to_dict(self) -> dict:
        return {
            name: {
                "xref": metrics.xref,
                "basefont": metrics.basefont,
                "encoding": metrics.encoding,
                "source": metrics.source,
                "font_name": metrics.font.name,
                "ascender": float(metrics.font.ascender),
                "descender": float(metrics.font.descender),
                "is_monospaced": bool(metrics.font.is_monospaced),
            }
            for name, metrics in sorted(self._by_resource.items())
        }


@dataclass(frozen=True)
class PdfTextGeometryEvent:
    """Evento textual acrescido de quad/bbox estimadas por métricas de fonte."""

    xref: int
    sequence: int
    operator: str
    text: str
    raw_hex: str
    font_resource: Optional[str]
    font_size: Optional[float]
    font_xref: Optional[int]
    font_basefont: Optional[str]
    font_metrics_source: str
    rendering_mode: int
    pdf_origin: Point
    page_origin: Point
    text_matrix: MatrixTuple
    ctm: MatrixTuple
    width_text_space: float
    ascender_text_space: float
    descender_text_space: float
    pdf_quad: QuadTuple
    pdf_bbox: BBoxTuple
    page_quad: QuadTuple
    page_bbox: BBoxTuple
    bbox_quality: str

    @property
    def invisible(self) -> bool:
        return self.rendering_mode == 3

    def to_dict(self) -> dict:
        def rounded_point(point: Point) -> list[float]:
            return [round(point[0], 4), round(point[1], 4)]

        def rounded_bbox(bbox: BBoxTuple) -> list[float]:
            return [round(value, 4) for value in bbox]

        return {
            "xref": self.xref,
            "sequence": self.sequence,
            "operator": self.operator,
            "text": self.text,
            "raw_hex": self.raw_hex,
            "font_resource": self.font_resource,
            "font_size": self.font_size,
            "font_xref": self.font_xref,
            "font_basefont": self.font_basefont,
            "font_metrics_source": self.font_metrics_source,
            "rendering_mode": self.rendering_mode,
            "invisible": self.invisible,
            "pdf_origin": rounded_point(self.pdf_origin),
            "page_origin": rounded_point(self.page_origin),
            "text_matrix": [round(value, 6) for value in self.text_matrix],
            "ctm": [round(value, 6) for value in self.ctm],
            "width_text_space": round(self.width_text_space, 4),
            "ascender_text_space": round(self.ascender_text_space, 4),
            "descender_text_space": round(self.descender_text_space, 4),
            "pdf_quad": [rounded_point(point) for point in self.pdf_quad],
            "pdf_bbox": rounded_bbox(self.pdf_bbox),
            "page_quad": [rounded_point(point) for point in self.page_quad],
            "page_bbox": rounded_bbox(self.page_bbox),
            "bbox_quality": self.bbox_quality,
        }


class PdfContentTextGeometryParser(PdfContentTextParser):
    """Parser de diagnóstico que estima geometria e avanço de cada ``Tj/TJ``."""

    def __init__(
        self,
        *,
        font_resolver: PdfFontMetricsResolver,
        pdf_to_page: fitz.Matrix,
        text_encoding: str = "cp1252",
    ) -> None:
        super().__init__(pdf_to_page=pdf_to_page, text_encoding=text_encoding)
        self.font_resolver = font_resolver

    @staticmethod
    def _safe_vertical_metrics(font: fitz.Font, fontsize: float) -> tuple[float, float]:
        ascender = float(font.ascender) * fontsize
        descender = float(font.descender) * fontsize

        # Alguns arquivos/fontes podem expor métricas inválidas. O fallback é
        # conservador e explícito, nunca tratado como ground truth.
        if not (ascender > 0.0):
            ascender = 0.8 * fontsize
        if not (descender < 0.0):
            descender = -0.2 * fontsize
        return ascender, descender

    def _estimate_geometry(
        self,
        base: PdfTextEvent,
        raw_parts: list[bytes],
    ) -> PdfTextGeometryEvent:
        fontsize = float(base.font_size or 0.0)
        metrics = self.font_resolver.resolve(base.font_resource)

        if metrics is not None and fontsize > 0.0:
            try:
                width = float(metrics.font.text_length(base.text, fontsize=fontsize))
                ascender, descender = self._safe_vertical_metrics(metrics.font, fontsize)
                quality = "font_metrics"
            except Exception:
                width = max(0.0, len(base.text) * fontsize * 0.6)
                ascender = 0.8 * fontsize
                descender = -0.2 * fontsize
                quality = "fallback_heuristic"
        else:
            width = max(0.0, len(base.text) * fontsize * 0.6)
            ascender = 0.8 * fontsize
            descender = -0.2 * fontsize
            quality = "fallback_heuristic"

        local_points = (
            fitz.Point(0.0, descender),
            fitz.Point(width, descender),
            fitz.Point(width, ascender),
            fitz.Point(0.0, ascender),
        )
        pdf_points = tuple(  # type: ignore[assignment]
            _point_tuple(point * self.text_matrix * self.ctm)
            for point in local_points
        )
        pdf_quad: QuadTuple = pdf_points  # type: ignore[assignment]
        page_quad = _transform_quad(pdf_quad, self.pdf_to_page)

        return PdfTextGeometryEvent(
            xref=base.xref,
            sequence=base.sequence,
            operator=base.operator,
            text=base.text,
            raw_hex=base.raw_hex,
            font_resource=base.font_resource,
            font_size=base.font_size,
            font_xref=metrics.xref if metrics else None,
            font_basefont=metrics.basefont if metrics else None,
            font_metrics_source=metrics.source if metrics else "unresolved",
            rendering_mode=base.rendering_mode,
            pdf_origin=base.pdf_origin,
            page_origin=base.page_origin,
            text_matrix=_matrix_tuple(self.text_matrix),
            ctm=_matrix_tuple(self.ctm),
            width_text_space=width,
            ascender_text_space=ascender,
            descender_text_space=descender,
            pdf_quad=pdf_quad,
            pdf_bbox=_quad_bbox(pdf_quad),
            page_quad=page_quad,
            page_bbox=_quad_bbox(page_quad),
            bbox_quality=quality,
        )

    def _emit(
        self,
        xref: int,
        operator: str,
        raw_parts: Iterable[bytes],
    ) -> Optional[PdfTextGeometryEvent]:
        parts = list(raw_parts)
        base = super()._emit(xref, operator, parts)
        if base is None:
            return None

        event = self._estimate_geometry(base, parts)

        # Diferentemente do parser de origem, este diagnóstico atualiza a text
        # matrix após mostrar texto. Isto importa quando há Tj consecutivos sem
        # um Td/Tm intermediário. Para o CAD atual, TJ com ajustes numéricos não
        # foi observado; a largura visual continua sendo uma aproximação.
        if event.width_text_space:
            advance = fitz.Matrix(1, 0, 0, 1, event.width_text_space, 0)
            self.text_matrix = advance * self.text_matrix

        return event


def extract_page_text_geometry_events(
    doc: fitz.Document,
    page_index: int = 0,
    *,
    text_encoding: str = "cp1252",
) -> tuple[list[PdfTextGeometryEvent], PdfFontMetricsResolver]:
    """Extrai eventos com bboxes estimadas preservando estado entre streams."""

    page = doc[page_index]
    resolver = PdfFontMetricsResolver(doc, page_index=page_index)
    parser = PdfContentTextGeometryParser(
        font_resolver=resolver,
        pdf_to_page=page.transformation_matrix,
        text_encoding=text_encoding,
    )

    contents = page.get_contents()
    if isinstance(contents, int):
        xrefs = [contents]
    else:
        xrefs = list(contents or [])

    events: list[PdfTextGeometryEvent] = []
    for xref in xrefs:
        stream = doc.xref_stream(int(xref))
        parsed = parser.parse_stream(stream, xref=int(xref))
        events.extend(parsed)  # type: ignore[arg-type]

    return events, resolver


__all__ = [
    "PdfContentTextGeometryParser",
    "PdfFontMetricsResolver",
    "PdfTextGeometryEvent",
    "ResolvedFontMetrics",
    "extract_page_text_geometry_events",
]
