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
        return "-"
    if isinstance(value, bool):
        return "Sim" if value else "Não"
    if isinstance(value, (list, tuple, set)):
        return "; ".join(_display(item) for item in value) or "-"
    if isinstance(value, dict):
        return "; ".join(f"{key}: {_display(item)}" for key, item in value.items()) or "-"
    return str(value)


HEADER_FIELDS = [
    ("drawing_number", "Drawing No."),
    ("title", "Title"),
    ("compressor_series_code", "Compressor Series Code"),
    ("cr", "CR"),
    ("classification", "Classification"),
    ("last_revision_date", "Last revision date"),
]

DRAWING_BLOCK_FIELDS = [
    ("materials", "Materials"),
    ("material_code", "Material Code"),
    ("drawn_by", "Drawn by"),
    ("approved_by", "Approved by"),
    ("drawing_code_ecm", "Drawing Code (ECM)"),
    ("date", "Date"),
    ("name_and_document_type", "Name and document type"),
    ("general_tolerance", "General tolerance"),
    ("angular_tolerance", "Angular tolerance"),
    ("scale", "Scale"),
    ("unit", "Unit"),
    ("replace", "Replace"),
    ("number", "Number"),
]


def _metadata_rows(
    values: dict[str, Any],
    fields: list[tuple[str, str]],
    label_style: ParagraphStyle,
    value_style: ParagraphStyle,
) -> list[list[Paragraph]]:
    return [
        [
            Paragraph(_escape(label), label_style),
            Paragraph(_escape(_display(values.get(key))), value_style),
        ]
        for key, label in fields
    ]


def _metadata_table(rows: list[list[Paragraph]]) -> Table:
    table = Table(rows, colWidths=[7.2 * cm, 18.3 * cm])
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.6, colors.HexColor("#333333")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#F7F7F7")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return table


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
    return Paragraph(f"- {_escape(text)}", style)


def build_unified_report(result: Any) -> bytes:
    """Build the customer report in metadata → diff → standards → findings order."""

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
    cell = ParagraphStyle("Cell", parent=body, fontSize=7.5, leading=8.5, spaceAfter=0)
    label_cell = ParagraphStyle("LabelCell", parent=cell, fontName="Helvetica-Bold")
    header = ParagraphStyle("HeaderCell", parent=cell, textColor=colors.white, alignment=TA_CENTER)

    story: list[Any] = [
        Paragraph("Relatório Integrado de Revisão CAD", title),
        Spacer(1, 0.25 * cm),
        Paragraph(f"Original: {_escape(result.original_name)}", body),
        Paragraph(f"Revisado: {_escape(result.revised_name)}", body),
        Spacer(1, 0.35 * cm),
        Paragraph("1. Header", heading),
    ]

    classification = result.part_classification
    header_values = dict(classification.get("header") or {})
    if not header_values.get("classification"):
        header_values["classification"] = classification.get("classificacao")
    drawing_block = dict(classification.get("drawing_block") or {})

    story.extend([
        _metadata_table(_metadata_rows(header_values, HEADER_FIELDS, label_cell, cell)),
        Spacer(1, 0.4 * cm),
        Paragraph("Drawing Block Transcription", heading),
        _metadata_table(_metadata_rows(drawing_block, DRAWING_BLOCK_FIELDS, label_cell, cell)),
        PageBreak(),
        Paragraph("2. Applied Standards", heading),
    ])

    cited = result.part_classification.get("lista_normas", []) or []
    evidence = result.part_classification.get("justificativas_normas", []) or []
    if cited:
        for index, standard in enumerate(cited):
            suffix = f" - {evidence[index]}" if index < len(evidence) and evidence[index] else ""
            story.append(_bullet(f"{standard}{suffix}", bullet))
    else:
        story.append(_bullet("No explicit standard was extracted from the revised drawing.", bullet))

    suggested = result.inferred_standards.get("normas_sugeridas", []) or []
    if suggested:
        story.append(Spacer(1, 0.15 * cm))
        story.append(Paragraph("Standards suggested for human validation", subheading))
        for standard in suggested:
            story.append(_bullet(str(standard), bullet))
        reasoning = result.inferred_standards.get("reasoning")
        if reasoning:
            story.append(Paragraph(f"Reasoning: {_escape(reasoning)}", body))

    story.extend([PageBreak(), Paragraph("3. Difference Map with IDs", heading)])

    comparison_images = 0
    for index, page in enumerate(result.comparison_pages):
        highlighted = getattr(page, "image_highlighted", None)
        if highlighted is None:
            continue
        if comparison_images:
            story.append(PageBreak())
        page_number = int(getattr(page, "page_index", index)) + 1
        story.append(Paragraph(f"Comparison - page {page_number}", subheading))
        story.append(_image_flowable(highlighted, max_width=25.5 * cm, max_height=15.8 * cm))
        comparison_images += 1
    if not comparison_images:
        story.append(_bullet("No comparison image was generated.", bullet))

    story.extend([PageBreak(), Paragraph("4. Difference Table", heading)])
    if result.paper_format_changes:
        story.append(Paragraph("Deterministic drawing format changes", subheading))
        for change in result.paper_format_changes:
            story.append(
                _bullet(
                    f"Page {change.get('page', '?')}: {change.get('description', 'format changed')}",
                    bullet,
                )
            )
        story.append(Spacer(1, 0.2 * cm))
    for index, page in enumerate(result.comparison_pages):
        if index:
            story.append(PageBreak())
        page_number = int(getattr(page, "page_index", index)) + 1
        story.append(Paragraph(f"Comparison - page {page_number}", subheading))
        true_changes = list(getattr(page, "true_changes", []) or [])
        false_positives = list(getattr(page, "false_positive_ids", []) or [])
        story.append(_bullet(f"Confirmed changes: {len(true_changes)}", bullet))
        story.append(_bullet(f"Filtered false positives: {len(false_positives)}", bullet))
        if true_changes:
            change_rows = [[Paragraph("ID", header), Paragraph("Difference found", header), Paragraph("Recommended Action", header)]]
            for change in true_changes:
                description = str(getattr(change, "description", "Change identified"))
                change_rows.append([
                    Paragraph(str(getattr(change, "index", "-")), cell),
                    Paragraph(_escape(description), cell),
                    Paragraph("Validate the change against the applicable technical requirement.", cell),
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
            story.append(_bullet("No significant change was confirmed.", bullet))

    story.append(PageBreak())
    comparison_by_id_count = 0
    for page_index, page in enumerate(result.comparison_pages):
        page_number = int(getattr(page, "page_index", page_index)) + 1
        for change in list(getattr(page, "true_changes", []) or []):
            if comparison_by_id_count:
                story.append(PageBreak())

            card: list[Any] = []
            if comparison_by_id_count == 0:
                card.append(Paragraph("5. Part Comparison by ID", heading))
            change_id = getattr(change, "index", "-")
            card.append(Paragraph(f"Page {page_number} - ID {change_id}", subheading))

            previous_crop = getattr(change, "original_crop", None)
            current_crop = getattr(change, "revised_crop", None)
            crop_rows: list[list[Any]] = [[
                Paragraph("Previous", label_cell),
                Paragraph("Current", label_cell),
            ]]
            crop_rows.append([
                _image_flowable(previous_crop, max_width=11.8 * cm, max_height=7.2 * cm)
                if previous_crop is not None
                else Paragraph("Image unavailable", cell),
                _image_flowable(current_crop, max_width=11.8 * cm, max_height=7.2 * cm)
                if current_crop is not None
                else Paragraph("Image unavailable", cell),
            ])
            crop_table = Table(crop_rows, colWidths=[12.5 * cm, 12.5 * cm])
            crop_table.setStyle(TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#777777")),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF3")),
                ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]))
            description = getattr(change, "description", "Change identified")
            location = (
                f"x={getattr(change, 'x', '-')}, y={getattr(change, 'y', '-')}, "
                f"w={getattr(change, 'width', '-')}, h={getattr(change, 'height', '-')}"
            )
            card.extend([
                crop_table,
                Spacer(1, 0.25 * cm),
                _bullet(f"Difference found: {description}", bullet),
                _bullet(
                    "Recommended Action: Validate the change against the applicable technical requirement.",
                    bullet,
                ),
                _bullet(f"Location: {location}", bullet),
            ])
            story.append(KeepTogether(card))
            comparison_by_id_count += 1

    if not comparison_by_id_count:
        story.extend([
            Paragraph("5. Part Comparison by ID", heading),
            _bullet("No significant change was confirmed.", bullet),
        ])

    story.append(PageBreak())
    for index, page in enumerate(result.gdt_pages):
        if index:
            story.append(PageBreak())
        summary = page.report.get("summary", {})
        page_story = []
        if index == 0:
            page_story.append(Paragraph("6. GD&amp;T and Datums", heading))
        page_story.append(Paragraph(f"Page {page.page_index + 1}", subheading))
        page_story.append(_bullet(f"GD&T detected: {summary.get('total_detections', 0)}", bullet))
        page_story.append(_bullet(f"Resolved datum references: {summary.get('resolved_datum_refs', 0)}", bullet))
        page_story.append(_bullet(f"Datum definitions found: {summary.get('datum_definitions_found', 0)}", bullet))
        if page.annotated_image is not None:
            page_story.append(Spacer(1, 0.15 * cm))
            page_story.append(
                _image_flowable(page.annotated_image, max_width=25.5 * cm, max_height=13.4 * cm)
            )
        story.append(KeepTogether(page_story))

    doc.build(story)
    return buffer.getvalue()


__all__ = ["build_unified_report"]
