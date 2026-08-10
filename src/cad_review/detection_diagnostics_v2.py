"""Visual diagnostics for GD&T Candidate Detector V2.

The diagnostic keeps proposal generation and validation visually separate:
- raw proposals: every region proposed by V1/vector/raster sources;
- accepted: proposals allowed downstream by the structural validator;
- rejected: proposals rejected by validator, with reasons;
- primitive audit: JSON-ready metadata is returned to the caller.

Nothing in these images is labelled TP/FP without independent ground truth.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import fitz
from PIL import Image, ImageDraw, ImageFont


def _font(size: int, bold: bool = False):
    paths = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in paths:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def _bbox(row: Mapping[str, Any]):
    value = row.get("frame_bbox")
    if isinstance(value, (list, tuple)) and len(value) == 4:
        return tuple(float(v) for v in value)
    return None


def _draw_rows(base: Image.Image, rows: Iterable[Mapping[str, Any]], scale: float, mode: str) -> Image.Image:
    img = base.copy()
    draw = ImageDraw.Draw(img)
    font = _font(max(12, int(7.5 * scale)), bold=True)
    header_font = _font(max(14, int(9 * scale)), bold=True)
    palette = {
        "raw": (40, 90, 180),
        "accepted": (35, 130, 65),
        "rejected": (190, 45, 45),
    }
    color = palette[mode]
    rows = list(rows)
    draw.rectangle((6, 6, min(img.width - 6, 820), 42), fill=(255, 255, 255), outline=(60, 60, 60), width=2)
    draw.text((12, 12), f"GDT Candidate Detector V2 | {mode.upper()} | count={len(rows)}", fill=(25, 25, 25), font=header_font)

    for row in rows:
        box = _bbox(row)
        if box is None:
            continue
        rect = tuple(int(round(v * scale)) for v in box)
        draw.rectangle(rect, outline=color, width=max(2, int(scale)))
        pid = str(row.get("proposal_id") or row.get("candidate_id") or "PROPOSAL")
        sources = "+".join(str(s) for s in (row.get("sources") or []))
        reasons = ",".join(str(s) for s in (row.get("rejection_reasons") or []))
        label = pid
        if mode == "raw" and sources:
            label += f" | {sources}"
        if mode == "rejected" and reasons:
            label += f" | {reasons}"
        tx = max(2, rect[0])
        ty = max(2, rect[1] - max(18, int(10 * scale)))
        tb = draw.textbbox((tx, ty), label, font=font)
        draw.rectangle((tb[0]-2, tb[1]-2, tb[2]+2, tb[3]+2), fill=(255,255,255), outline=color, width=1)
        draw.text((tx, ty), label, fill=color, font=font)
    return img


def render_v2_detection_diagnostics(
    pdf_bytes: bytes,
    *,
    output_dir: str | Path,
    page_results: Iterable[Mapping[str, Any]],
    dpi: int = 180,
) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    by_page = {int(row["page"]): dict(row) for row in page_results}
    artifacts = []
    scale = dpi / 72.0

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_index, page in enumerate(doc):
            page_number = page_index + 1
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=fitz.csRGB, alpha=False)
            base = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            result = by_page.get(page_number, {})
            raw = list(result.get("raw_proposals") or [])
            accepted_ids = set(result.get("accepted_candidate_ids") or [])
            rejected_ids = set(result.get("rejected_proposal_ids") or [])
            accepted = [r for r in raw if r.get("proposal_id") in accepted_ids]
            rejected = [r for r in raw if r.get("proposal_id") in rejected_ids]

            raw_path = output / f"page_{page_number:03d}_v2_raw_proposals.png"
            accepted_path = output / f"page_{page_number:03d}_v2_accepted.png"
            rejected_path = output / f"page_{page_number:03d}_v2_rejected.png"
            _draw_rows(base, raw, scale, "raw").save(raw_path)
            _draw_rows(base, accepted, scale, "accepted").save(accepted_path)
            _draw_rows(base, rejected, scale, "rejected").save(rejected_path)

            artifacts.append({
                "page": page_number,
                "raw_proposals_image": raw_path.name,
                "accepted_image": accepted_path.name,
                "rejected_image": rejected_path.name,
                "primitive_audit": result.get("primitive_audit") or {},
            })

    return {
        "validation_scope": "candidate_detector_v2_no_ground_truth",
        "ground_truth_used": False,
        "dpi": int(dpi),
        "pages": artifacts,
    }


__all__ = ["render_v2_detection_diagnostics"]
