"""Generate the customer-facing unified CAD Review PDF."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
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

from src.utils.standards import (
    filter_standard_entries,
    is_generic_standard,
    standard_key,
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

REFERENCE_ASSETS = Path(__file__).resolve().parents[2] / "assets" / "references"
REFERENCE_IMAGES = [
    ("gdt_datum/geometrical_characteristics.png", "ISO 1101 geometrical characteristics and datum requirements"),
    ("gdt_datum/datum_related_symbols.png", "Datum-related and size-tolerance-related symbols"),
    ("gdt_datum/toleranced_feature_identifiers.png", "Toleranced feature identifiers and tolerance indicators"),
]
UNTYPED_DIMENSION_METRICS = {
    "Quantidade de cotas HIC",
    "Quantidade de cotas CTQ",
    "Quantidade de cotas CTQ-S",
}


def _load_standards_catalog() -> list[dict[str, str]]:
    catalog_path = REFERENCE_ASSETS / "standards_catalog.json"
    try:
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return [row for row in payload if isinstance(row, dict)]


def _category_from_standard_evidence(evidence: str) -> str:
    """Classify an uncatalogued cited standard without inventing its contents."""
    normalized = str(evidence or "").upper()
    category_rules = [
        (("HAZARDOUS", "MATERIAL", "HFC", "SUBSTANCE"), "Material and compliance"),
        (("GEOMETRIC", "GD&T", "DATUM", "TOLERANCE", "DIMENSION"), "Dimensioning and tolerances"),
        (("MEASUREMENT", "DENSITY", "TEST", "PROCEDURE", "VALIDATION"), "Test and validation"),
        (("IDENTIFICATION", "LETTERING", "MARKING", "LABEL"), "Identification"),
    ]
    for keywords, category in category_rules:
        if any(keyword in normalized for keyword in keywords):
            return category
    return "General"


def _uncatalogued_standard_metadata(evidence: str) -> dict[str, str]:
    """Build complete, evidence-based metadata for a standard absent from the catalog."""
    evidence_text = str(evidence or "").strip()
    return {
        "content": evidence_text or "Explicit standard reference in the revised drawing",
        "category": _category_from_standard_evidence(evidence_text),
        "applicability": "Applicable as explicitly cited in the revised drawing",
    }


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
    raster_dpi: int = 450,
) -> Image:
    """Fit an image to the PDF at high resolution to avoid pixelation on zoom.

    raster_dpi controls how many raster pixels are embedded per display point.
    450 DPI (6.25x the 72 pt/inch PDF baseline) keeps full-page CAD drawings
    legible when zoomed in the report while keeping the file small enough to
    email as an attachment. The image is only downsampled when the source is
    actually larger than the raster budget, so smaller sources stay untouched.
    The image is only downsampled when the source is actually smaller than the
    raster budget, so large source images (300 DPI CAD pages) stay crisp.
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


def _annotate_page_with_bbox(
    full_page: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    change_id: int | str,
    *,
    box_color: tuple[int, int, int] = (0, 0, 220),
    alpha: float = 0.25,
    sub_differences: list[dict] | None = None,
    subbox_color: tuple[int, int, int] = (219, 112, 147),
    subbox_alpha: float = 0.30,
) -> np.ndarray:
    """Draw a numbered semi-transparent group box on a full-page image.

    Marks the change region so the reviewer sees it in the context of the
    entire drawing (same approach as relatorio_ia_pagina_N.pdf).

    When ``sub_differences`` holds more than one entry, each individual
    difference's box(es) are drawn as lilac boxes labeled with its ``sub_id``
    (e.g. "1.2"), and the red numbered group box is drawn on top.
    """
    img = full_page.copy()
    h_img, w_img = img.shape[:2]

    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(w_img, x + w)
    y1 = min(h_img, y + h)

    if x1 <= x0 or y1 <= y0:
        return img

    border_px = max(3, w_img // 600)
    font_scale = max(0.7, w_img / 2800.0)
    thickness = max(2, int(font_scale * 2.5))
    sub_font_scale = max(0.5, font_scale * 0.8)
    sub_thickness = max(2, int(sub_font_scale * 2.2))

    def _clip(bx, by, bw, bh):
        cx0 = max(0, bx)
        cy0 = max(0, by)
        cx1 = min(w_img, bx + bw)
        cy1 = min(h_img, by + bh)
        return cx0, cy0, cx1, cy1

    def _label(text, bx0, by0, color, fscale, fthick):
        (lw, lh), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, fscale, fthick
        )
        pad = 5
        lx = bx0 + pad
        ly = by0 - pad if by0 - pad > lh + pad else by0 + lh + pad * 2
        cv2.rectangle(
            img,
            (lx - pad, ly - lh - pad),
            (lx + lw + pad, ly + baseline + pad),
            (255, 255, 255),
            -1,
        )
        cv2.putText(
            img, text, (lx, ly),
            cv2.FONT_HERSHEY_SIMPLEX, fscale, color, fthick, cv2.LINE_AA,
        )

    # Draw lilac sub-difference boxes first (only for multi-difference groups).
    sub_differences = sub_differences or []
    if len(sub_differences) > 1:
        sub_overlay = img.copy()
        for sub in sub_differences:
            for (sbx, sby, sbw, sbh) in sub.get("boxes", []):
                sx0, sy0, sx1, sy1 = _clip(sbx, sby, sbw, sbh)
                if sx1 <= sx0 or sy1 <= sy0:
                    continue
                cv2.rectangle(sub_overlay, (sx0, sy0), (sx1, sy1), subbox_color, -1)
        cv2.addWeighted(sub_overlay, subbox_alpha, img, 1 - subbox_alpha, 0, img)
        for sub in sub_differences:
            sub_id = str(sub.get("sub_id", ""))
            for (sbx, sby, sbw, sbh) in sub.get("boxes", []):
                sx0, sy0, sx1, sy1 = _clip(sbx, sby, sbw, sbh)
                if sx1 <= sx0 or sy1 <= sy0:
                    continue
                cv2.rectangle(img, (sx0, sy0), (sx1, sy1), subbox_color, border_px)
                if sub_id:
                    _label(sub_id, sx0, sy0, subbox_color, sub_font_scale, sub_thickness)

    # Semi-transparent fill for the red group box
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), box_color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # Solid border + numeric group label
    cv2.rectangle(img, (x0, y0), (x1, y1), box_color, border_px)
    _label(str(change_id), x0, y0, box_color, font_scale, thickness)

    return img


def build_unified_report(result: Any) -> bytes:
    """Build the customer report.

    Section order:
        1. Header + Drawing Block
        2. Applied Standards
        3. GD&T and Datums
        4. Difference Map with IDs
        5. Differences by ID  (summary table + per-ID page with full-page images)
        6. References (objective metrics, full standards catalog and symbol references)
    """
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
    title_style = ParagraphStyle("ReportTitle", parent=styles["Title"], alignment=TA_CENTER, fontSize=19)
    heading    = ParagraphStyle("Section",    parent=styles["Heading1"], fontSize=15, spaceAfter=10)
    subheading = ParagraphStyle("Subsection", parent=styles["Heading2"], fontSize=12, spaceAfter=7)
    body       = ParagraphStyle("Body",       parent=styles["BodyText"], fontSize=9,  leading=12, spaceAfter=5)
    bullet_sty = ParagraphStyle("Bullet",     parent=body, leftIndent=12, firstLineIndent=-8)
    standards_bullet_sty = ParagraphStyle(
        "StandardsBullet", parent=bullet_sty, spaceAfter=0, leading=11
    )
    cell       = ParagraphStyle("Cell",       parent=body, fontSize=7.5, leading=8.5, spaceAfter=0)
    label_cell = ParagraphStyle("LabelCell",  parent=cell, fontName="Helvetica-Bold")
    tbl_header = ParagraphStyle("HeaderCell", parent=cell, textColor=colors.white, alignment=TA_CENTER)
    caption    = ParagraphStyle(
        "Caption", parent=cell,
        fontSize=8, textColor=colors.HexColor("#1565C0"),
        alignment=TA_CENTER,
    )

    # ── Cover / title ─────────────────────────────────────────────────────────
    story: list[Any] = [
        Paragraph("Integrated CAD Review Report", title_style),
        Spacer(1, 0.25 * cm),
        Paragraph(f"Original: {_escape(result.original_name)}", body),
        Paragraph(f"Revised: {_escape(result.revised_name)}", body),
        Spacer(1, 0.35 * cm),
    ]

    # ── 1. Header + Drawing Block ─────────────────────────────────────────────
    story.append(Paragraph("1. Header", heading))

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
    ])

    # ── 2. Applied Standards ──────────────────────────────────────────────────
    story.append(Paragraph("2. Applied Standards", heading))
    story.append(Paragraph("Standards cited in the revised drawing", subheading))

    cited, evidence = filter_standard_entries(
        result.part_classification.get("lista_normas", []) or [],
        result.part_classification.get("justificativas_normas", []) or [],
    )
    if cited:
        for idx, standard in enumerate(cited):
            suffix = f" - {evidence[idx]}" if idx < len(evidence) and evidence[idx] else ""
            story.append(_bullet(f"{standard}{suffix}", standards_bullet_sty))
    else:
        story.append(_bullet("No explicit standard was extracted from the revised drawing.", bullet_sty))

    suggested = result.inferred_standards.get("normas_sugeridas", []) or []
    if suggested:
        story.append(Spacer(1, 0.15 * cm))
        story.append(Paragraph("Standards suggested for human validation", subheading))
        for standard in suggested:
            story.append(_bullet(str(standard), bullet_sty))
        reasoning = result.inferred_standards.get("reasoning")
        if reasoning:
            story.append(Paragraph(f"Reasoning: {_escape(reasoning)}", body))

    story.append(PageBreak())

    # ── 3. GD&T and Datums ────────────────────────────────────────────────────
    for idx, page in enumerate(result.gdt_pages):
        if idx:
            story.append(PageBreak())
        summary = page.report.get("summary", {})
        page_story: list[Any] = []
        if idx == 0:
            page_story.append(Paragraph("3. GD&T and Datums", heading))
        page_story.append(Paragraph(f"Page {page.page_index + 1}", subheading))
        page_story.append(_bullet(f"GD&T detected: {summary.get('total_detections', 0)}", bullet_sty))
        page_story.append(_bullet(f"Resolved datum references: {summary.get('resolved_datum_refs', 0)}", bullet_sty))
        page_story.append(_bullet(f"Datum definitions found: {summary.get('datum_definitions_found', 0)}", bullet_sty))
        if page.annotated_image is not None:
            page_story.append(Spacer(1, 0.15 * cm))
            page_story.append(
                _image_flowable(page.annotated_image, max_width=25.5 * cm, max_height=13.4 * cm)
            )
        story.append(KeepTogether(page_story))

    story.append(PageBreak())

    # ── 4. Difference Map with IDs ────────────────────────────────────────────
    story.append(Paragraph("4. Difference Map with IDs", heading))

    comparison_images = 0
    for idx, page in enumerate(result.comparison_pages):
        highlighted = getattr(page, "image_highlighted", None)
        if highlighted is None:
            continue
        if comparison_images:
            story.append(PageBreak())
        page_number = int(getattr(page, "page_index", idx)) + 1
        story.append(Paragraph(f"Comparison - page {page_number}", subheading))
        story.append(_image_flowable(highlighted, max_width=25.5 * cm, max_height=15.8 * cm))
        comparison_images += 1
    if not comparison_images:
        story.append(_bullet("No comparison image was generated.", bullet_sty))

    story.append(PageBreak())

    # ── 5. Differences by ID ─────────────────────────────────────────────────
    story.append(Paragraph("5. Differences by ID", heading))

    if result.paper_format_changes:
        story.append(Paragraph("Deterministic drawing format changes", subheading))
        for change in result.paper_format_changes:
            story.append(
                _bullet(
                    f"Page {change.get('page', '?')}: {change.get('description', 'format changed')}",
                    bullet_sty,
                )
            )
        story.append(Spacer(1, 0.2 * cm))

    for pg_idx, page in enumerate(result.comparison_pages):
        if pg_idx:
            story.append(PageBreak())

        page_number = int(getattr(page, "page_index", pg_idx)) + 1
        story.append(Paragraph(f"Comparison - page {page_number}", subheading))

        true_changes  = list(getattr(page, "true_changes", []) or [])
        false_positives = list(getattr(page, "false_positive_ids", []) or [])
        story.append(_bullet(f"Confirmed changes: {len(true_changes)}", bullet_sty))
        story.append(_bullet(f"Filtered false positives: {len(false_positives)}", bullet_sty))

        if not true_changes:
            story.append(_bullet("No significant change was confirmed.", bullet_sty))
            continue

        # Summary table (all IDs on one page before the per-ID detail pages)
        change_rows = [[
            Paragraph("ID", tbl_header),
            Paragraph("Difference found", tbl_header),
            Paragraph("Recommended Action", tbl_header),
        ]]
        for change in true_changes:
            subs = list(getattr(change, "sub_differences", None) or [])
            if len(subs) > 1:
                # One line per sub-difference, prefixed with its sub_id.
                diff_html = "<br/>".join(
                    f"[{s.get('sub_id', '')}] {_escape(s.get('description', ''))}"
                    for s in subs
                )
            elif subs:
                diff_html = _escape(subs[0].get("description", ""))
            else:
                diff_html = _escape(str(getattr(change, "description", "Change identified")))
            change_rows.append([
                Paragraph(str(getattr(change, "index", "-")), cell),
                Paragraph(diff_html, cell),
                Paragraph("Validate the change against the applicable technical requirement.", cell),
            ])
        changes_table = Table(change_rows, colWidths=[1.4 * cm, 16.2 * cm, 8.0 * cm], repeatRows=1)
        changes_table.setStyle(TableStyle([
            ("BACKGROUND",   (0, 0), (-1, 0), colors.HexColor("#2E7D32")),
            ("GRID",         (0, 0), (-1, -1), 0.4, colors.grey),
            ("VALIGN",       (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F8F4")]),
            ("LEFTPADDING",  (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING",   (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
        ]))
        story.extend([Spacer(1, 0.2 * cm), changes_table])

        # Per-ID detail pages: each change on its own page
        img_original = getattr(page, "image_original", None)
        img_revised  = getattr(page, "image_revised_aligned", None)

        avail_w = 25.5 * cm
        col_w   = (avail_w / 2.0) - 0.4 * cm

        for change in true_changes:
            story.append(PageBreak())

            change_id   = getattr(change, "index", "-")
            description = getattr(change, "description", "Change identified")
            x = int(getattr(change, "x", 0))
            y = int(getattr(change, "y", 0))
            w = int(getattr(change, "width", 0))
            h = int(getattr(change, "height", 0))
            # Normalize sub_differences: coerce box coords to int tuples.
            sub_differences = []
            for sub in (getattr(change, "sub_differences", None) or []):
                sub_differences.append({
                    "sub_id": str(sub.get("sub_id", "")),
                    "description": sub.get("description", ""),
                    "boxes": [tuple(int(v) for v in b) for b in sub.get("boxes", [])],
                })

            story.append(Paragraph(f"<b>Change #{change_id}</b>", subheading))
            story.append(Spacer(1, 0.15 * cm))

            if img_original is not None and img_revised is not None:
                ann_orig = _annotate_page_with_bbox(img_original, x, y, w, h, change_id, sub_differences=sub_differences)
                ann_rev  = _annotate_page_with_bbox(img_revised,  x, y, w, h, change_id, sub_differences=sub_differences)

                aspect = img_original.shape[0] / max(1, img_original.shape[1])
                max_h  = col_w * aspect

                try:
                    flow_orig = _image_flowable(ann_orig, max_width=col_w, max_height=max_h)
                    flow_rev  = _image_flowable(ann_rev,  max_width=col_w, max_height=max_h)

                    img_tbl = Table(
                        [[flow_orig, flow_rev]],
                        colWidths=[col_w, col_w],
                    )
                    img_tbl.setStyle(TableStyle([
                        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
                        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
                        ("LEFTPADDING",   (0, 0), (-1, -1), 3),
                        ("RIGHTPADDING",  (0, 0), (-1, -1), 3),
                        ("TOPPADDING",    (0, 0), (-1, -1), 3),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ]))

                    cap_tbl = Table(
                        [[Paragraph("<i>Original</i>", caption), Paragraph("<i>Revised</i>", caption)]],
                        colWidths=[col_w, col_w],
                    )
                    cap_tbl.setStyle(TableStyle([
                        ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
                        ("LEFTPADDING",  (0, 0), (-1, -1), 3),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                    ]))

                    story.extend([img_tbl, cap_tbl])
                except (ValueError, TypeError):
                    story.append(Paragraph("<i>Images unavailable for this change.</i>", body))
            else:
                story.append(
                    Paragraph(f"<i>Change location: x={x}, y={y}, w={w}, h={h}</i>", body)
                )

            story.append(Spacer(1, 0.25 * cm))
            # Bullet topics prefixed with the sub_id shown on each lilac box.
            if len(sub_differences) > 1:
                story.append(Paragraph("<b>Differences in this group:</b>", body))
                for sub in sub_differences:
                    story.append(
                        _bullet(f"[{sub['sub_id']}] {_escape(sub['description'])}", bullet_sty)
                    )
            else:
                single = sub_differences[0]["description"] if sub_differences else description
                story.append(Paragraph(f"<b>Description:</b> {_escape(single)}", body))

    # ── 6. References ─────────────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("6. References", heading))

    story.append(Paragraph("Objective Metrics", subheading))
    metric_rows = [[
        Paragraph("Metric", tbl_header),
        Paragraph("Result", tbl_header),
    ]]
    objective_metrics = dict(getattr(result, "objective_metrics", {}) or {})
    metric_order = [
        "Quantidade de cotas",
        "Quantidade de cotas HIC",
        "Quantidade de cotas CTQ",
        "Quantidade de cotas CTQ-S",
        "Quantidade de GD&Ts",
        "Quantidade de Datums Reference",
        "Lista de datums reference",
        "Quantidade de revisões",
        "Quantidade de notas",
        "Quantidade de códigos",
    ]
    for metric in metric_order:
        metric_value = (
            "-"
            if metric in UNTYPED_DIMENSION_METRICS
            else objective_metrics.get(metric)
        )
        metric_rows.append([
            Paragraph(_escape(metric), label_cell),
            Paragraph(_escape(_display(metric_value)), cell),
        ])

    metric_table = Table(metric_rows, colWidths=[12.0 * cm, 13.5 * cm], repeatRows=1)
    metric_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#455A64")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F6F7")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(metric_table)

    standards_catalog = _load_standards_catalog()
    standards_map = {
        standard_key(row.get("standard", "")): row
        for row in standards_catalog
    }
    evidence_map = {
        standard_key(standard): evidence[index] if index < len(evidence) else ""
        for index, standard in enumerate(cited)
    }

    story.append(Spacer(1, 0.35 * cm))
    story.append(Paragraph("Applied Standards Table", subheading))
    applied_rows = [[
        Paragraph("Standard", tbl_header),
        Paragraph("Content", tbl_header),
        Paragraph("Category", tbl_header),
        Paragraph("Applicability", tbl_header),
    ]]

    mapped_applied = result.inferred_standards.get("applied_standards", []) or []
    applied_sources: list[Any] = [*cited, *mapped_applied]
    seen_applied: set[str] = set()
    for applied in applied_sources:
        if isinstance(applied, dict):
            standard_code = str(applied.get("standard", "")).strip()
            applicability_override = applied.get("applicability_match")
            content_override = applied.get("content")
            category_override = applied.get("category")
        else:
            standard_code = str(applied).strip()
            applicability_override = None
            content_override = None
            category_override = None
        code_key = standard_key(standard_code)
        if (
            not standard_code
            or not code_key
            or code_key in seen_applied
            or is_generic_standard(standard_code)
        ):
            continue
        seen_applied.add(code_key)
        catalog_record = standards_map.get(code_key, {})
        fallback = _uncatalogued_standard_metadata(evidence_map.get(code_key, ""))
        content = content_override or catalog_record.get("content") or fallback["content"]
        category = category_override or catalog_record.get("category") or fallback["category"]
        applicability = (
            applicability_override
            or catalog_record.get("applicability")
            or fallback["applicability"]
        )
        applied_rows.append([
            Paragraph(_escape(standard_code), cell),
            Paragraph(_escape(content), cell),
            Paragraph(_escape(category), cell),
            Paragraph(_escape(applicability), cell),
        ])

    if len(applied_rows) > 1:
        applied_table = Table(
            applied_rows,
            colWidths=[3.0 * cm, 6.7 * cm, 4.2 * cm, 11.6 * cm],
            repeatRows=1,
        )
        applied_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1565C0")),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EAF2FB")]),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(applied_table)
    else:
        story.append(_bullet("No applied standard was identified.", bullet_sty))

    story.append(PageBreak())
    story.append(Paragraph("Complete Standards Catalog", subheading))
    standards_rows = [[
        Paragraph("Standard", tbl_header),
        Paragraph("Content", tbl_header),
        Paragraph("Category", tbl_header),
        Paragraph("Compressor Series", tbl_header),
        Paragraph("Applicability", tbl_header),
    ]]
    for standard in standards_catalog:
        standards_rows.append([
            Paragraph(_escape(standard.get("standard", "")), cell),
            Paragraph(_escape(standard.get("content", "")), cell),
            Paragraph(_escape(standard.get("category", "")), cell),
            Paragraph(_escape(standard.get("compressor_series", "")), cell),
            Paragraph(_escape(standard.get("applicability", "")), cell),
        ])

    if len(standards_rows) > 1:
        standards_table = Table(
            standards_rows,
            colWidths=[2.7 * cm, 5.5 * cm, 3.5 * cm, 4.0 * cm, 9.8 * cm],
            repeatRows=1,
        )
        standards_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#00695C")),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EAF5F3")]),
            ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(standards_table)
    else:
        story.append(_bullet("The standards catalog asset is unavailable.", bullet_sty))

    for relative_path, image_caption in REFERENCE_IMAGES:
        story.append(PageBreak())
        image_path = REFERENCE_ASSETS / relative_path
        reference_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if reference_image is None:
            story.append(Paragraph("GD&T and Datum Symbol Reference", subheading))
            story.append(_bullet(f"Reference image unavailable: {relative_path}", bullet_sty))
            continue
        reference_table = Table(
            [
                [Paragraph("GD&T and Datum Symbol Reference", subheading)],
                [_image_flowable(
                    reference_image,
                    max_width=25.5 * cm,
                    max_height=13.6 * cm,
                )],
                [Paragraph(f"<i>{_escape(image_caption)}</i>", caption)],
            ],
            colWidths=[25.5 * cm],
        )
        reference_table.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]))
        story.append(reference_table)

    doc.build(story)
    return buffer.getvalue()


__all__ = ["build_unified_report"]
