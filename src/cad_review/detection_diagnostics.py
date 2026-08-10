"""Diagnostics that keep GD&T detection and symbol classification separate.

These artifacts are for Phase-3/multi-CAD validation.  A detector candidate is
never called a true GD&T frame until independently annotated ground truth is
available.

Per page the module writes:
- ``page_NNN_candidates.png``: Phase-1 geometry candidates only; labels contain
  only candidate IDs and therefore do not mix in classifier/compliance output.
- ``page_NNN_symbol_contact_sheet.png``: crop of each candidate plus the symbol
  classifier ranking supplied by the caller.

At drawing level it writes ``candidate_diagnostics.csv``.  The final three
columns are intentionally blank so an engineer/reviewer can independently mark
whether a candidate is a real FCF and its true characteristic.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import fitz
from PIL import Image, ImageDraw, ImageFont

_CANDIDATE_COLOR = (35, 95, 165)
_TEXT_COLOR = (25, 25, 25)


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    )
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _bbox(row: Mapping[str, Any], *keys: str) -> Optional[tuple[float, float, float, float]]:
    for key in keys:
        value = row.get(key)
        if isinstance(value, (list, tuple)) and len(value) == 4:
            return tuple(float(v) for v in value)
    return None


def _page_index(value: Any, page_count: int) -> Optional[int]:
    try:
        page = int(value)
    except (TypeError, ValueError):
        return None
    index = 0 if page == 0 else page - 1
    return index if 0 <= index < page_count else None


def _short_id(candidate_id: str) -> str:
    value = str(candidate_id)
    if value.startswith("GDT-CAND-"):
        return value[len("GDT-CAND-") :]
    return value


def _crop(
    image: Image.Image,
    bbox: tuple[float, float, float, float],
    scale: float,
    *,
    padding_pt: float = 5.0,
) -> Image.Image:
    x0, y0, x1, y1 = bbox
    pad = padding_pt * scale
    rect = (
        max(0, int(round(x0 * scale - pad))),
        max(0, int(round(y0 * scale - pad))),
        min(image.width, int(round(x1 * scale + pad))),
        min(image.height, int(round(y1 * scale + pad))),
    )
    return image.crop(rect)


def _fit_thumbnail(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    result = image.copy()
    result.thumbnail(size, Image.Resampling.LANCZOS)
    return result


def _score_ranking(row: Mapping[str, Any], top_k: int) -> list[tuple[str, float]]:
    scoring = row.get("symbol_scoring") or {}
    scores = scoring.get("class_scores") or {}
    output: list[tuple[str, float]] = []
    for name, value in scores.items():
        try:
            output.append((str(name), float(value)))
        except (TypeError, ValueError):
            continue
    output.sort(key=lambda item: item[1], reverse=True)
    return output[: max(1, int(top_k))]


def _draw_candidate_page(
    base: Image.Image,
    rows: list[Mapping[str, Any]],
    *,
    page_number: int,
    scale: float,
) -> Image.Image:
    image = base.copy()
    draw = ImageDraw.Draw(image)
    label_font = _font(max(12, int(round(7.5 * scale))), bold=True)
    header_font = _font(max(13, int(round(7 * scale))), bold=True)

    # Header states explicitly that this is pre-classifier Phase-1 evidence.
    header = f"Phase 1 candidates only | page {page_number} | candidates: {len(rows)}"
    hb = draw.textbbox((12, 12), header, font=header_font)
    draw.rectangle((8, 8, hb[2] + 8, hb[3] + 8), fill=(255, 255, 255), outline=_CANDIDATE_COLOR, width=2)
    draw.text((12, 12), header, fill=_TEXT_COLOR, font=header_font)

    for row in sorted(rows, key=lambda item: ((_bbox(item, "frame_bbox", "bbox") or (0, 0, 0, 0))[1], (_bbox(item, "frame_bbox", "bbox") or (0, 0, 0, 0))[0])):
        bbox = _bbox(row, "frame_bbox", "bbox")
        if bbox is None:
            continue
        rect = tuple(int(round(value * scale)) for value in bbox)
        draw.rectangle(rect, outline=_CANDIDATE_COLOR, width=max(3, int(round(scale))))
        candidate_id = str(row.get("candidate_id") or "CAND")
        label = f"CAND {_short_id(candidate_id)}"
        tb = draw.textbbox((0, 0), label, font=label_font)
        width = tb[2] - tb[0] + 8
        height = tb[3] - tb[1] + 8
        x = max(2, min(rect[0], image.width - width - 2))
        y = max(2, rect[1] - height - 4)
        if y <= 3:
            y = min(image.height - height - 2, rect[3] + 4)
        box = (x, y, x + width, y + height)
        draw.rectangle(box, fill=(255, 255, 255), outline=_CANDIDATE_COLOR, width=2)
        draw.text((x + 4, y + 3), label, fill=_CANDIDATE_COLOR, font=label_font)
    return image


def _contact_sheet(
    base: Image.Image,
    rows: list[Mapping[str, Any]],
    *,
    page_number: int,
    scale: float,
    top_k: int,
) -> Image.Image:
    columns = 2
    card_w = 760
    card_h = 250
    header_h = 70
    rows_count = max(1, (len(rows) + columns - 1) // columns)
    sheet = Image.new("RGB", (columns * card_w, header_h + rows_count * card_h), "white")
    draw = ImageDraw.Draw(sheet)
    title_font = _font(24, bold=True)
    body_font = _font(18)
    body_bold = _font(18, bold=True)
    draw.text((18, 16), f"Symbol ranking diagnostics | page {page_number} | candidates: {len(rows)}", fill=_TEXT_COLOR, font=title_font)

    for index, row in enumerate(rows):
        col = index % columns
        line = index // columns
        x0 = col * card_w
        y0 = header_h + line * card_h
        draw.rectangle((x0 + 5, y0 + 5, x0 + card_w - 5, y0 + card_h - 5), outline=(170, 170, 170), width=2)

        candidate_id = str(row.get("candidate_id") or f"CAND-{index + 1:03d}")
        draw.text((x0 + 18, y0 + 14), f"CAND {_short_id(candidate_id)}", fill=_CANDIDATE_COLOR, font=body_bold)

        frame_bbox = _bbox(row, "frame_bbox", "bbox")
        symbol_bbox = _bbox(row, "symbol_bbox")
        if frame_bbox is not None:
            frame_crop = _fit_thumbnail(_crop(base, frame_bbox, scale, padding_pt=5.0), (330, 150))
            sheet.paste(frame_crop, (x0 + 18, y0 + 52))
        if symbol_bbox is not None:
            symbol_crop = _fit_thumbnail(_crop(base, symbol_bbox, scale, padding_pt=3.0), (120, 120))
            sx = x0 + 365
            sy = y0 + 55
            sheet.paste(symbol_crop, (sx, sy))
            draw.rectangle((sx - 2, sy - 2, sx + symbol_crop.width + 2, sy + symbol_crop.height + 2), outline=(130, 130, 130), width=1)
            draw.text((sx, y0 + 185), "symbol cell", fill=_TEXT_COLOR, font=body_font)

        ranking = _score_ranking(row, top_k)
        tx = x0 + 510
        ty = y0 + 52
        scoring = row.get("symbol_scoring") or {}
        if not ranking:
            draw.text((tx, ty), "classification: not evaluated", fill=_TEXT_COLOR, font=body_font)
        else:
            draw.text((tx, ty), "ranking:", fill=_TEXT_COLOR, font=body_bold)
            for rank, (name, score) in enumerate(ranking, start=1):
                draw.text((tx, ty + 28 * rank), f"{rank}. {name}: {score:.4f}", fill=_TEXT_COLOR, font=body_font)
            margin = scoring.get("margin")
            if margin is not None:
                try:
                    margin_text = f"margin: {float(margin):.4f}"
                except (TypeError, ValueError):
                    margin_text = f"margin: {margin}"
                draw.text((tx, ty + 28 * (len(ranking) + 1) + 8), margin_text, fill=_TEXT_COLOR, font=body_font)

        num_cells = len(row.get("cell_bboxes") or [])
        draw.text((x0 + 18, y0 + 215), f"cells={num_cells} | detector status={row.get('detection_status', 'candidate_unvalidated')}", fill=_TEXT_COLOR, font=body_font)

    return sheet


def _write_candidate_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    headers = [
        "candidate_id",
        "page",
        "frame_bbox",
        "symbol_bbox",
        "num_cells",
        "top1_class",
        "top1_score",
        "top2_class",
        "top2_score",
        "margin",
        "decision_policy",
        "catalog_complete",
        "referenced_datums",
        "unresolved_fields",
        "human_is_real_gdt",
        "human_true_characteristic",
        "human_notes",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            ranking = _score_ranking(row, 2)
            scoring = row.get("symbol_scoring") or {}
            writer.writerow(
                {
                    "candidate_id": row.get("candidate_id"),
                    "page": row.get("page"),
                    "frame_bbox": list(_bbox(row, "frame_bbox", "bbox") or ()),
                    "symbol_bbox": list(_bbox(row, "symbol_bbox") or ()),
                    "num_cells": len(row.get("cell_bboxes") or []),
                    "top1_class": ranking[0][0] if len(ranking) >= 1 else "",
                    "top1_score": f"{ranking[0][1]:.6f}" if len(ranking) >= 1 else "",
                    "top2_class": ranking[1][0] if len(ranking) >= 2 else "",
                    "top2_score": f"{ranking[1][1]:.6f}" if len(ranking) >= 2 else "",
                    "margin": scoring.get("margin", ""),
                    "decision_policy": scoring.get("decision_policy", ""),
                    "catalog_complete": scoring.get("catalog_complete", ""),
                    "referenced_datums": ";".join(str(v) for v in (row.get("referenced_datums") or [])),
                    "unresolved_fields": ";".join(str(v) for v in (row.get("unresolved_fields") or [])),
                    "human_is_real_gdt": "",
                    "human_true_characteristic": "",
                    "human_notes": "",
                }
            )


def render_detection_diagnostics(
    pdf_bytes: bytes,
    *,
    output_dir: str | Path,
    gdt_candidates: Iterable[Mapping[str, Any]],
    dpi: int = 180,
    top_k: int = 3,
) -> dict:
    """Write Phase-1-only and Phase-2-ranking diagnostics for one drawing."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    candidate_rows = [dict(row) for row in gdt_candidates]
    csv_path = output / "candidate_diagnostics.csv"
    _write_candidate_csv(csv_path, candidate_rows)

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages: list[dict] = []
    try:
        scale = float(dpi) / 72.0
        for page_index in range(len(doc)):
            page = doc[page_index]
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=fitz.csRGB, alpha=False)
            base = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            rows = [
                row
                for row in candidate_rows
                if _page_index(row.get("page", 1), len(doc)) == page_index
            ]
            rows.sort(key=lambda row: str(row.get("candidate_id") or ""))

            candidates_image = _draw_candidate_page(base, rows, page_number=page_index + 1, scale=scale)
            candidates_path = output / f"page_{page_index + 1:03d}_candidates.png"
            candidates_image.save(candidates_path)

            contact = _contact_sheet(base, rows, page_number=page_index + 1, scale=scale, top_k=top_k)
            contact_path = output / f"page_{page_index + 1:03d}_symbol_contact_sheet.png"
            contact.save(contact_path)

            pages.append(
                {
                    "page": page_index + 1,
                    "candidate_count": len(rows),
                    "candidates_image": str(candidates_path.relative_to(output)),
                    "symbol_contact_sheet": str(contact_path.relative_to(output)),
                }
            )
    finally:
        doc.close()

    return {
        "validation_scope": "phase1_geometry_candidates_plus_phase2_ranking",
        "ground_truth_used": False,
        "candidate_semantics": "unvalidated detector proposals",
        "classification_semantics": "ranking only; not probability and no calibrated global acceptance threshold",
        "dpi": int(dpi),
        "top_k": int(top_k),
        "candidate_csv": csv_path.name,
        "pages": pages,
    }


__all__ = ["render_detection_diagnostics"]
