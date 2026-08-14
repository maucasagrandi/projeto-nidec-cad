"""Generate the customer-facing unified CAD Review PDF."""

from __future__ import annotations

from io import BytesIO
from typing import Any

import cv2
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def _escape(value: Any) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _display(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "Sim" if value else "Não"
    if isinstance(value, (list, tuple, set)):
        return "; ".join(_display(item) for item in value) or "—"
    if isinstance(value, dict):
        return "; ".join(f"{key}: {_display(item)}" for key, item in value.items()) or "—"
    return str(value)


def _classification_rows(classification: dict[str, Any]) -> list[list[Any]]:
    labels = {
        "classificacao": "Classificação da peça",
        "justificativa_classificacao": "Evidência da classificação",
        "lista_normas": "Normas citadas",
        "justificativas_normas": "Evidências das normas",
    }
    ordered = [key for key in labels if key in classification]
    ordered.extend(key for key in classification if key not in ordered)
    return [[labels.get(key, key.replace("_", " ").title()), _display(classification[key])] for key in ordered]


def _image_flowable(
    image_bgr: np.ndarray,
    *,
    max_width: float,
    max_height: float,
    raster_dpi: int = 144,
) -> Image:
    """Fit an image to the PDF and downsample it before ReportLab decodes it.

    OpenCV comparison panels can exceed 8,000 pixels in width. ReportLab would
    otherwise expand the full PNG to RGB even though the image is displayed at
    roughly 10 inches, causing a large temporary allocation on Windows.
    """

    if not isinstance(image_bgr, np.ndarray) or image_bgr.ndim not in (2, 3):
        raise TypeError("Report image must be a NumPy array")
    height, width = image_bgr.shape[:2]
    if width < 1 or height < 1:
        raise ValueError("Report image cannot be empty")

    display_scale = min(max_width / width, max_height / height)
    display_width = width * display_scale
    display_height = height * display_scale

    pixels_per_point = raster_dpi / 72.0
    max_raster_width = max(1, round(display_width * pixels_per_point))
    max_raster_height = max(1, round(display_height * pixels_per_point))
    raster_scale = min(1.0, max_raster_width / width, max_raster_height / height)

    report_image = image_bgr
    if raster_scale < 1.0:
        report_image = cv2.resize(
            image_bgr,
            (
                max(1, round(width * raster_scale)),
                max(1, round(height * raster_scale)),
            ),
            interpolation=cv2.INTER_AREA,
        )

    ok, encoded = cv2.imencode(".png", report_image)
    if not ok:
        raise ValueError("Could not encode report image")
    source = BytesIO(encoded.tobytes())
    return Image(source, width=display_width, height=display_height)


def _bullet(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(f"• {_escape(text)}", style)


def build_unified_report(result: Any) -> bytes:
    """Build the classification → dimensions → GD&T → comparison report."""

    buffer = BytesIO()
    page_size = landscape(A4)
    doc = SimpleDocTemplate(
        buffer,
        pagesize=page_size,
        leftMargin=1.4 * cm,
        rightMargin=1.4 * cm,
        topMargin=1.2 * cm,
        bottomMargin=1.2 * cm,
        title="Integrated CAD Review Report",
        author="CAD Review Platform",
    )
    styles = getSampleStyleSheet()
    title = ParagraphStyle("ReportTitle", parent=styles["Title"], alignment=TA_CENTER, fontSize=19)
    heading = ParagraphStyle("Section", parent=styles["Heading1"], fontSize=15, spaceAfter=10)
    subheading = ParagraphStyle("Subsection", parent=styles["Heading2"], fontSize=12, spaceAfter=7)
    body = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=9, leading=12, spaceAfter=5)
    bullet = ParagraphStyle("Bullet", parent=body, leftIndent=12, firstLineIndent=-8)
    cell = ParagraphStyle("Cell", parent=body, fontSize=8, leading=10)
    header = ParagraphStyle("HeaderCell", parent=cell, textColor=colors.white, alignment=TA_CENTER)

    story: list[Any] = [
        Paragraph("Relatório Integrado de Revisão CAD", title),
        Spacer(1, 0.25 * cm),
        Paragraph(f"Original: {_escape(result.original_name)}", body),
        Paragraph(f"Revisado: {_escape(result.revised_name)}", body),
        Spacer(1, 0.35 * cm),
        Paragraph("1. Part Classification", heading),
    ]

    rows = [[Paragraph("Campo", header), Paragraph("Valor extraído", header)]]
    rows.extend(
        [Paragraph(_escape(label), cell), Paragraph(_escape(value), cell)]
        for label, value in _classification_rows(result.part_classification)
    )
    table = Table(rows, colWidths=[5.2 * cm, 20.3 * cm], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E78")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#A0A0A0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F6F8")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.extend([table, Spacer(1, 0.45 * cm), Paragraph("2. Normas", heading)])

    cited = result.part_classification.get("lista_normas", []) or []
    evidence = result.part_classification.get("justificativas_normas", []) or []
    if cited:
        story.append(Paragraph("Normas explicitamente citadas no desenho revisado", subheading))
        for index, standard in enumerate(cited):
            suffix = f" — {evidence[index]}" if index < len(evidence) and evidence[index] else ""
            story.append(_bullet(f"{standard}{suffix}", bullet))
    else:
        story.append(_bullet("Nenhuma norma explícita foi extraída do desenho revisado.", bullet))

    suggested = result.inferred_standards.get("normas_sugeridas", []) or []
    if suggested:
        story.append(Spacer(1, 0.15 * cm))
        story.append(Paragraph("Normas sugeridas para validação humana", subheading))
        for standard in suggested:
            story.append(_bullet(str(standard), bullet))
        reasoning = result.inferred_standards.get("reasoning")
        if reasoning:
            story.append(Paragraph(f"Justificativa: {_escape(reasoning)}", body))

    story.extend([PageBreak(), Paragraph("3. Cotas do desenho revisado", heading)])
    dimension_total = sum(len(page.dimensions) for page in result.dimension_pages)
    story.append(_bullet(f"Total de cotas detectadas: {dimension_total}", bullet))
    for index, page in enumerate(result.dimension_pages):
        if index:
            story.append(PageBreak())
        story.append(Paragraph(f"Página {page.page_index + 1}", subheading))
        story.append(_bullet(f"Cotas detectadas nesta página: {len(page.dimensions)}", bullet))
        if page.dimensions:
            dimension_rows = [[
                Paragraph("ID", header),
                Paragraph("Cota extraída", header),
                Paragraph("Quadrante", header),
                Paragraph("BBox PDF (x0, y0, x1, y1)", header),
            ]]
            for dimension in page.dimensions:
                bbox = ", ".join(f"{coordinate:.1f}" for coordinate in dimension.bbox)
                dimension_rows.append([
                    Paragraph(_escape(dimension.dimension_id), cell),
                    Paragraph(_escape(dimension.value), cell),
                    Paragraph(_escape(dimension.quadrant), cell),
                    Paragraph(bbox, cell),
                ])
            dimensions_table = Table(
                dimension_rows,
                colWidths=[3.2 * cm, 7.0 * cm, 3.0 * cm, 12.4 * cm],
                repeatRows=1,
            )
            dimensions_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#B71C1C")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FFF5F5")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.extend([Spacer(1, 0.15 * cm), dimensions_table])
        else:
            story.append(_bullet("Nenhuma cota compatível foi encontrada na camada de texto.", bullet))
        if page.annotated_image is not None:
            story.append(Spacer(1, 0.25 * cm))
            story.append(KeepTogether([
                Paragraph("Desenho revisado com cotas marcadas", subheading),
                _image_flowable(page.annotated_image, max_width=25.5 * cm, max_height=12.5 * cm),
            ]))

    story.extend([PageBreak(), Paragraph("4. GD&T e datums no desenho revisado", heading)])
    for index, page in enumerate(result.gdt_pages):
        if index:
            story.append(PageBreak())
        summary = page.report.get("summary", {})
        story.append(Paragraph(f"Página {page.page_index + 1}", subheading))
        story.append(_bullet(f"GD&T detectados: {summary.get('total_detections', 0)}", bullet))
        story.append(_bullet(f"Referências de datum resolvidas: {summary.get('resolved_datum_refs', 0)}", bullet))
        story.append(_bullet(f"Definições de datum encontradas: {summary.get('datum_definitions_found', 0)}", bullet))
        if page.annotated_image is not None:
            story.append(Spacer(1, 0.15 * cm))
            story.append(_image_flowable(page.annotated_image, max_width=25.5 * cm, max_height=14.2 * cm))

    story.extend([PageBreak(), Paragraph("5. Part Comparison", heading)])
    if result.paper_format_changes:
        story.append(Paragraph("Alterações determinísticas de formato do desenho", subheading))
        for change in result.paper_format_changes:
            story.append(
                _bullet(
                    f"Página {change.get('page', '?')}: {change.get('description', 'formato alterado')}",
                    bullet,
                )
            )
        story.append(Spacer(1, 0.2 * cm))
    for index, page in enumerate(result.comparison_pages):
        if index:
            story.append(PageBreak())
        page_number = int(getattr(page, "page_index", index)) + 1
        story.append(Paragraph(f"Comparação — página {page_number}", subheading))
        true_changes = list(getattr(page, "true_changes", []) or [])
        false_positives = list(getattr(page, "false_positive_ids", []) or [])
        story.append(_bullet(f"Mudanças confirmadas: {len(true_changes)}", bullet))
        story.append(_bullet(f"Falsos positivos filtrados: {len(false_positives)}", bullet))
        if true_changes:
            change_rows = [[Paragraph("ID", header), Paragraph("Difference found", header), Paragraph("Recommended Action", header)]]
            for change in true_changes:
                description = str(getattr(change, "description", "Mudança identificada"))
                change_rows.append([
                    Paragraph(str(getattr(change, "index", "—")), cell),
                    Paragraph(_escape(description), cell),
                    Paragraph("Validar a alteração em relação ao requisito técnico aplicável.", cell),
                ])
            changes_table = Table(change_rows, colWidths=[1.4 * cm, 16.2 * cm, 8.0 * cm], repeatRows=1)
            changes_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2E7D32")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F8F4")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]))
            story.extend([Spacer(1, 0.2 * cm), changes_table])
        else:
            story.append(_bullet("Nenhuma mudança significativa confirmada.", bullet))

        highlighted = getattr(page, "image_highlighted", None)
        if highlighted is not None:
            story.extend([
                Spacer(1, 0.3 * cm),
                _image_flowable(highlighted, max_width=25.5 * cm, max_height=10.5 * cm),
            ])

    doc.build(story)
    return buffer.getvalue()


__all__ = ["build_unified_report"]
