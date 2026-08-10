"""Visual evidence output for integrated CAD Review results.

The overlay is deliberately evidence-oriented: it shows *where* the pipeline
looked and what it inferred, without turning an unvalidated detector candidate
into ground truth. Generic batch detections are therefore labelled ``GDT-CAND``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import fitz
from PIL import Image, ImageDraw, ImageFont

_STATUS_COLORS = {
    "WARNING": (190, 40, 40),
    "NEEDS_CONTEXT": (190, 125, 0),
    "PASS": (35, 125, 65),
    "NOT_EVALUATED": (35, 95, 165),
}
_DATUM_COLOR = (110, 55, 150)
_PRIORITY = {"NOT_EVALUATED": 0, "PASS": 1, "NEEDS_CONTEXT": 2, "WARNING": 3}


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
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
    # Detector/datum contracts are 1-indexed. Accept zero only as an explicit
    # first-page value for diagnostic payloads.
    index = 0 if page == 0 else page - 1
    return index if 0 <= index < page_count else None


def _status_maps(findings: Iterable[Mapping[str, Any]]) -> tuple[dict[str, str], dict[str, str]]:
    by_candidate: dict[str, str] = {}
    by_datum: dict[str, str] = {}
    for finding in findings:
        status = str(finding.get("status") or "NOT_EVALUATED")
        if status not in _PRIORITY:
            status = "NOT_EVALUATED"
        candidate = finding.get("candidate_id")
        datum = finding.get("datum")
        if candidate:
            key = str(candidate)
            previous = by_candidate.get(key, "NOT_EVALUATED")
            if _PRIORITY[status] > _PRIORITY[previous]:
                by_candidate[key] = status
        if datum:
            key = str(datum).upper()
            previous = by_datum.get(key, "NOT_EVALUATED")
            if _PRIORITY[status] > _PRIORITY[previous]:
                by_datum[key] = status
    return by_candidate, by_datum


def _draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, color: tuple[int, int, int], *, size: int = 18) -> None:
    font = _font(size, bold=True)
    x, y = xy
    box = draw.textbbox((x, y), text, font=font)
    pad = 3
    draw.rectangle((box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad), fill=(255, 255, 255), outline=color, width=2)
    draw.text((x, y), text, fill=color, font=font)


def _crop_from_pdf_bbox(image: Image.Image, bbox: tuple[float, float, float, float], scale: float, *, padding_pt: float = 8.0) -> Image.Image:
    x0, y0, x1, y1 = bbox
    p = padding_pt * scale
    box = (
        max(0, int(round(x0 * scale - p))),
        max(0, int(round(y0 * scale - p))),
        min(image.width, int(round(x1 * scale + p))),
        min(image.height, int(round(y1 * scale + p))),
    )
    return image.crop(box)


def render_visual_evidence(
    pdf_bytes: bytes,
    *,
    output_dir: str | Path,
    gdt_candidates: Iterable[Mapping[str, Any]],
    datum_definitions: Iterable[Mapping[str, Any]],
    findings: Iterable[Mapping[str, Any]] = (),
    dpi: int = 180,
    save_crops: bool = True,
) -> dict:
    """Render annotated pages and evidence crops.

    Returns paths relative to ``output_dir`` so the caller can embed them in the
    per-CAD JSON without leaking machine-specific absolute paths.
    """

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    crop_dir = output / "crops"
    if save_crops:
        crop_dir.mkdir(parents=True, exist_ok=True)

    gdt_rows = [dict(row) for row in gdt_candidates]
    datum_rows = [dict(row) for row in datum_definitions]
    finding_rows = [dict(row) for row in findings]
    candidate_status, datum_status = _status_maps(finding_rows)

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    artifacts: list[dict] = []
    crop_paths: list[str] = []
    try:
        scale = float(dpi) / 72.0
        for page_index in range(len(doc)):
            page = doc[page_index]
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=fitz.csRGB, alpha=False)
            image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            draw = ImageDraw.Draw(image)

            page_gdt = [row for row in gdt_rows if _page_index(row.get("page", 1), len(doc)) == page_index]
            page_datums = [row for row in datum_rows if _page_index(row.get("page", 1), len(doc)) == page_index]

            for row in page_gdt:
                bbox = _bbox(row, "frame_bbox", "bbox")
                if bbox is None:
                    continue
                candidate_id = str(row.get("candidate_id") or row.get("gdt_id") or "GDT-CAND")
                status = candidate_status.get(candidate_id, str(row.get("status") or "NOT_EVALUATED"))
                color = _STATUS_COLORS.get(status, _STATUS_COLORS["NOT_EVALUATED"])
                rect = tuple(int(round(v * scale)) for v in bbox)
                draw.rectangle(rect, outline=color, width=max(3, int(round(scale))))
                characteristic = row.get("characteristic") or row.get("best_class") or "unclassified"
                refs = row.get("referenced_datums") or []
                refs_text = ",".join(str(v) for v in refs) if refs else "-"
                short_id = candidate_id.replace("GDT-CAND-P", "P")
                label = f"GDT-CAND {short_id} | {characteristic} | refs:{refs_text} | {status}"
                _draw_label(draw, (rect[0], max(0, rect[1] - 27)), label, color, size=max(12, int(round(7.5 * scale))))
                if save_crops:
                    crop = _crop_from_pdf_bbox(image, bbox, scale)
                    path = crop_dir / f"{candidate_id.replace(':', '_')}_frame.png"
                    crop.save(path)
                    crop_paths.append(str(path.relative_to(output)))

            for index, row in enumerate(page_datums, start=1):
                box_bbox = _bbox(row, "box_bbox", "bbox")
                marker_bbox = _bbox(row, "marker_bbox")
                if box_bbox is None:
                    continue
                label = str(row.get("label") or row.get("datum") or "?").upper()
                status = datum_status.get(label, "PASS")
                color = _STATUS_COLORS.get(status, _DATUM_COLOR) if status == "WARNING" else _DATUM_COLOR
                rect = tuple(int(round(v * scale)) for v in box_bbox)
                draw.rectangle(rect, outline=color, width=max(3, int(round(scale))))
                if marker_bbox is not None:
                    marker_rect = tuple(int(round(v * scale)) for v in marker_bbox)
                    draw.rectangle(marker_rect, outline=color, width=2)
                _draw_label(draw, (rect[0], max(0, rect[1] - 27)), f"DATUM-{label}", color, size=max(12, int(round(7.5 * scale))))
                if save_crops:
                    union = box_bbox
                    if marker_bbox is not None:
                        union = (
                            min(box_bbox[0], marker_bbox[0]), min(box_bbox[1], marker_bbox[1]),
                            max(box_bbox[2], marker_bbox[2]), max(box_bbox[3], marker_bbox[3]),
                        )
                    crop = _crop_from_pdf_bbox(image, union, scale, padding_pt=12.0)
                    path = crop_dir / f"DATUM-{label}_{page_index + 1:03d}_{index:02d}.png"
                    crop.save(path)
                    crop_paths.append(str(path.relative_to(output)))

            # Compact legend / page summary.
            legend_font = _font(max(13, int(round(7 * scale))), bold=True)
            summary = f"CAD Review | page {page_index + 1} | GDT candidates: {len(page_gdt)} | datum definitions: {len(page_datums)}"
            text_box = draw.textbbox((12, 12), summary, font=legend_font)
            draw.rectangle((8, 8, text_box[2] + 8, text_box[3] + 8), fill=(255, 255, 255), outline=(60, 60, 60), width=2)
            draw.text((12, 12), summary, fill=(30, 30, 30), font=legend_font)

            page_path = output / f"page_{page_index + 1:03d}_annotated.png"
            image.save(page_path)
            artifacts.append({
                "page": page_index + 1,
                "annotated_image": str(page_path.relative_to(output)),
                "gdt_candidate_count": len(page_gdt),
                "datum_definition_count": len(page_datums),
            })
    finally:
        doc.close()

    return {
        "dpi": int(dpi),
        "pages": artifacts,
        "crops": crop_paths,
        "label_policy": "GDT-CAND until independently validated",
    }


__all__ = ["render_visual_evidence"]
