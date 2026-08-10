"""Visual evidence output for integrated CAD Review results.

The overlay is deliberately evidence-oriented: it shows *where* the pipeline
looked and what it inferred, without turning an unvalidated detector candidate
into ground truth. Generic batch detections are therefore labelled ``GDT-CAND``.

Each page produces three views:
- combined: GD&T candidates + datum definitions;
- gdt: GD&T candidates only;
- datums: datum definitions only.

Labels are placed in free lanes around their source geometry when possible and
connected back to the frame/indicator, reducing overlap in dense drawings.
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


def _intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int = 4) -> bool:
    return not (
        a[2] + margin < b[0]
        or b[2] + margin < a[0]
        or a[3] + margin < b[1]
        or b[3] + margin < a[1]
    )


def _label_box(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    xy: tuple[int, int],
    *,
    pad: int = 3,
) -> tuple[int, int, int, int]:
    x, y = xy
    raw = draw.textbbox((x, y), text, font=font)
    return raw[0] - pad, raw[1] - pad, raw[2] + pad, raw[3] + pad


def _find_free_label_position(
    draw: ImageDraw.ImageDraw,
    image: Image.Image,
    text: str,
    font: ImageFont.ImageFont,
    source_rect: tuple[int, int, int, int],
    occupied: list[tuple[int, int, int, int]],
) -> tuple[tuple[int, int], tuple[int, int, int, int]]:
    probe = draw.textbbox((0, 0), text, font=font)
    width = probe[2] - probe[0] + 8
    height = probe[3] - probe[1] + 8
    x0, y0, x1, y1 = source_rect

    candidate_positions: list[tuple[int, int]] = []
    for lane in range(7):
        offset = lane * (height + 4)
        candidate_positions.extend(
            [
                (x0, y0 - height - 4 - offset),
                (x0, y1 + 4 + offset),
                (x1 + 6, y0 - offset),
                (x0 - width - 6, y0 - offset),
            ]
        )

    # Last-resort deterministic lane near the source; clipping keeps it on page.
    candidate_positions.append((x0, y0 - height - 4))

    for raw_x, raw_y in candidate_positions:
        x = max(2, min(int(raw_x), max(2, image.width - width - 2)))
        y = max(2, min(int(raw_y), max(2, image.height - height - 2)))
        box = _label_box(draw, text, font, (x, y))
        if not any(_intersects(box, other) for other in occupied):
            return (x, y), box

    x = max(2, min(x0, max(2, image.width - width - 2)))
    y = max(2, min(y0 - height - 4, max(2, image.height - height - 2)))
    return (x, y), _label_box(draw, text, font, (x, y))


def _draw_placed_label(
    draw: ImageDraw.ImageDraw,
    image: Image.Image,
    source_rect: tuple[int, int, int, int],
    text: str,
    color: tuple[int, int, int],
    occupied: list[tuple[int, int, int, int]],
    *,
    size: int,
) -> None:
    font = _font(size, bold=True)
    xy, box = _find_free_label_position(draw, image, text, font, source_rect, occupied)
    occupied.append(box)
    draw.rectangle(box, fill=(255, 255, 255), outline=color, width=2)
    draw.text(xy, text, fill=color, font=font)

    source_x = (source_rect[0] + source_rect[2]) // 2
    source_y = (source_rect[1] + source_rect[3]) // 2
    label_x = min(max(source_x, box[0]), box[2])
    label_y = min(max(source_y, box[1]), box[3])
    draw.line((source_x, source_y, label_x, label_y), fill=color, width=1)


def _draw_header(
    image: Image.Image,
    *,
    page_number: int,
    layer: str,
    gdt_count: int,
    datum_count: int,
    scale: float,
) -> None:
    draw = ImageDraw.Draw(image)
    font = _font(max(13, int(round(7 * scale))), bold=True)
    summary = (
        f"CAD Review | page {page_number} | layer={layer} | "
        f"GDT candidates: {gdt_count} | datum definitions: {datum_count}"
    )
    text_box = draw.textbbox((12, 12), summary, font=font)
    draw.rectangle(
        (8, 8, text_box[2] + 8, text_box[3] + 8),
        fill=(255, 255, 255),
        outline=(60, 60, 60),
        width=2,
    )
    draw.text((12, 12), summary, fill=(30, 30, 30), font=font)


def _draw_gdt_layer(
    image: Image.Image,
    rows: Iterable[Mapping[str, Any]],
    *,
    candidate_status: Mapping[str, str],
    scale: float,
) -> None:
    draw = ImageDraw.Draw(image)
    occupied: list[tuple[int, int, int, int]] = []
    sortable: list[tuple[tuple[float, float, float, float], Mapping[str, Any]]] = []
    for row in rows:
        bbox = _bbox(row, "frame_bbox", "bbox")
        if bbox is not None:
            sortable.append((bbox, row))

    for bbox, row in sorted(sortable, key=lambda item: (item[0][1], item[0][0])):
        candidate_id = str(row.get("candidate_id") or row.get("gdt_id") or "GDT-CAND")
        status = candidate_status.get(candidate_id, str(row.get("status") or "NOT_EVALUATED"))
        color = _STATUS_COLORS.get(status, _STATUS_COLORS["NOT_EVALUATED"])
        rect = tuple(int(round(v * scale)) for v in bbox)
        draw.rectangle(rect, outline=color, width=max(3, int(round(scale))))
        characteristic = row.get("characteristic") or row.get("best_class") or "unclassified"
        refs = row.get("referenced_datums") or []
        refs_text = ",".join(str(v) for v in refs) if refs else "-"
        short_id = candidate_id.replace("GDT-CAND-P", "P")
        label = f"CAND {short_id} | {characteristic} | refs:{refs_text} | {status}"
        _draw_placed_label(
            draw,
            image,
            rect,
            label,
            color,
            occupied,
            size=max(12, int(round(7.5 * scale))),
        )


def _draw_datum_layer(
    image: Image.Image,
    rows: Iterable[Mapping[str, Any]],
    *,
    datum_status: Mapping[str, str],
    scale: float,
) -> None:
    draw = ImageDraw.Draw(image)
    occupied: list[tuple[int, int, int, int]] = []
    for row in rows:
        box_bbox = _bbox(row, "box_bbox", "bbox")
        marker_bbox = _bbox(row, "marker_bbox")
        if box_bbox is None:
            continue
        label = str(row.get("label") or row.get("datum") or "?").upper()
        status = datum_status.get(label, "PASS")
        color = _STATUS_COLORS["WARNING"] if status == "WARNING" else _DATUM_COLOR
        rect = tuple(int(round(v * scale)) for v in box_bbox)
        draw.rectangle(rect, outline=color, width=max(3, int(round(scale))))
        if marker_bbox is not None:
            marker_rect = tuple(int(round(v * scale)) for v in marker_bbox)
            draw.rectangle(marker_rect, outline=color, width=2)
        _draw_placed_label(
            draw,
            image,
            rect,
            f"DATUM-{label}",
            color,
            occupied,
            size=max(12, int(round(7.5 * scale))),
        )


def _crop_from_pdf_bbox(
    image: Image.Image,
    bbox: tuple[float, float, float, float],
    scale: float,
    *,
    padding_pt: float = 8.0,
) -> Image.Image:
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
    """Render combined/split annotated pages and evidence crops.

    Returned paths are relative to ``output_dir`` so the JSON remains portable
    across machines and can be moved together with its RESULTS directory.
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
            pix = page.get_pixmap(
                matrix=fitz.Matrix(scale, scale),
                colorspace=fitz.csRGB,
                alpha=False,
            )
            base = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            page_gdt = [
                row for row in gdt_rows
                if _page_index(row.get("page", 1), len(doc)) == page_index
            ]
            page_datums = [
                row for row in datum_rows
                if _page_index(row.get("page", 1), len(doc)) == page_index
            ]

            combined = base.copy()
            gdt_only = base.copy()
            datums_only = base.copy()
            _draw_gdt_layer(combined, page_gdt, candidate_status=candidate_status, scale=scale)
            _draw_datum_layer(combined, page_datums, datum_status=datum_status, scale=scale)
            _draw_gdt_layer(gdt_only, page_gdt, candidate_status=candidate_status, scale=scale)
            _draw_datum_layer(datums_only, page_datums, datum_status=datum_status, scale=scale)

            _draw_header(
                combined,
                page_number=page_index + 1,
                layer="combined",
                gdt_count=len(page_gdt),
                datum_count=len(page_datums),
                scale=scale,
            )
            _draw_header(
                gdt_only,
                page_number=page_index + 1,
                layer="gdt",
                gdt_count=len(page_gdt),
                datum_count=0,
                scale=scale,
            )
            _draw_header(
                datums_only,
                page_number=page_index + 1,
                layer="datums",
                gdt_count=0,
                datum_count=len(page_datums),
                scale=scale,
            )

            combined_path = output / f"page_{page_index + 1:03d}_annotated.png"
            gdt_path = output / f"page_{page_index + 1:03d}_gdt.png"
            datums_path = output / f"page_{page_index + 1:03d}_datums.png"
            combined.save(combined_path)
            gdt_only.save(gdt_path)
            datums_only.save(datums_path)

            if save_crops:
                for row in page_gdt:
                    bbox = _bbox(row, "frame_bbox", "bbox")
                    if bbox is None:
                        continue
                    candidate_id = str(row.get("candidate_id") or row.get("gdt_id") or "GDT-CAND")
                    crop = _crop_from_pdf_bbox(combined, bbox, scale)
                    path = crop_dir / f"{candidate_id.replace(':', '_')}_frame.png"
                    crop.save(path)
                    crop_paths.append(str(path.relative_to(output)))

                for index, row in enumerate(page_datums, start=1):
                    box_bbox = _bbox(row, "box_bbox", "bbox")
                    marker_bbox = _bbox(row, "marker_bbox")
                    if box_bbox is None:
                        continue
                    label = str(row.get("label") or row.get("datum") or "?").upper()
                    union = box_bbox
                    if marker_bbox is not None:
                        union = (
                            min(box_bbox[0], marker_bbox[0]),
                            min(box_bbox[1], marker_bbox[1]),
                            max(box_bbox[2], marker_bbox[2]),
                            max(box_bbox[3], marker_bbox[3]),
                        )
                    crop = _crop_from_pdf_bbox(combined, union, scale, padding_pt=12.0)
                    path = crop_dir / f"DATUM-{label}_{page_index + 1:03d}_{index:02d}.png"
                    crop.save(path)
                    crop_paths.append(str(path.relative_to(output)))

            artifacts.append(
                {
                    "page": page_index + 1,
                    "annotated_image": str(combined_path.relative_to(output)),
                    "gdt_image": str(gdt_path.relative_to(output)),
                    "datums_image": str(datums_path.relative_to(output)),
                    "gdt_candidate_count": len(page_gdt),
                    "datum_definition_count": len(page_datums),
                }
            )
    finally:
        doc.close()

    return {
        "dpi": int(dpi),
        "pages": artifacts,
        "crops": crop_paths,
        "label_policy": "GDT-CAND until independently validated",
        "layers": ["combined", "gdt", "datums"],
        "label_placement": "non_overlapping_lanes_with_connectors",
    }


__all__ = ["render_visual_evidence"]
