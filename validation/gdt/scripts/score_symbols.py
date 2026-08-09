"""Pontua visualmente a primeira célula dos candidatos GD&T.

Uso:
    python validation/gdt/scripts/score_symbols.py \
      --case validation/gdt/cases/case_41_rev8.json \
      --templates assets/gdt/templates

Saídas:
    validation/gdt/outputs/<case_id>/symbol_scores.json
    validation/gdt/outputs/<case_id>/symbol_contact_sheet.png
    validation/gdt/outputs/<case_id>/symbol_crops/*.png

Importante: este script NÃO aplica threshold e NÃO decide se um candidato é
GD&T real. Ele apenas produz scores por classe/template para análise.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gdt.detector import GdtFrameDetector
from src.gdt.symbol_classifier import (
    DEFAULT_MARGIN,
    DEFAULT_TARGET_SIZE,
    SCORE_COMPONENTS,
    load_template_catalog,
    render_page_gray,
    score_candidates,
)


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _fit_crop(crop: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    if crop is None or crop.size == 0:
        return np.full((max_h, max_w), 255, np.uint8)
    h, w = crop.shape[:2]
    scale = min(max_w / max(w, 1), max_h / max(h, 1))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    canvas = np.full((max_h, max_w), 255, np.uint8)
    x = (max_w - new_w) // 2
    y = (max_h - new_h) // 2
    canvas[y:y + new_h, x:x + new_w] = resized
    return canvas


def _build_contact_sheet(scored_rows, *, columns: int = 3) -> np.ndarray:
    tile_w, tile_h = 430, 190
    rows = max(1, math.ceil(len(scored_rows) / columns))
    sheet = np.full((rows * tile_h, columns * tile_w, 3), 255, np.uint8)

    for index, (candidate, score, crop) in enumerate(scored_rows):
        row = index // columns
        col = index % columns
        x0 = col * tile_w
        y0 = row * tile_h
        tile = sheet[y0:y0 + tile_h, x0:x0 + tile_w]

        crop_img = _fit_crop(crop, 140, 105)
        crop_bgr = cv2.cvtColor(crop_img, cv2.COLOR_GRAY2BGR)
        tile[55:160, 15:155] = crop_bgr

        cv2.putText(tile, candidate.candidate_id, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(tile, f"cells={len(candidate.cells)}", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)

        if score.best_class is None:
            lines = ["best: n/a", "second: n/a", "margin: 0"]
        else:
            lines = [
                f"best: {score.best_class} {score.best_score:.3f}",
                f"second: {score.second_best_class or '-'} {score.second_best_score:.3f}",
                f"margin: {score.margin:.3f}",
            ]
        for line_idx, line in enumerate(lines):
            cv2.putText(tile, line, (175, 80 + line_idx * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)

        cv2.rectangle(tile, (0, 0), (tile_w - 1, tile_h - 1), (210, 210, 210), 1)

    return sheet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--templates", default="assets/gdt/templates")
    parser.add_argument("--target-size", type=int, default=DEFAULT_TARGET_SIZE)
    parser.add_argument("--margin", type=int, default=DEFAULT_MARGIN)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--output-root", default="validation/gdt/outputs")
    parser.add_argument(
        "--exclude-class",
        action="append",
        default=[],
        help="Classe de template a ignorar; pode repetir. Útil para controles legados.",
    )
    args = parser.parse_args()

    case_path = _project_path(args.case)
    config = json.loads(case_path.read_text(encoding="utf-8"))
    case_id = str(config["case_id"])
    pdf_path = _project_path(config["pdf"])
    page_index = int(config.get("page_index", 0))
    template_root = _project_path(args.templates)
    output_dir = _project_path(args.output_root) / case_id
    crop_dir = output_dir / "symbol_crops"
    crop_dir.mkdir(parents=True, exist_ok=True)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF não encontrado: {pdf_path}")

    pdf_bytes = pdf_path.read_bytes()
    detector = GdtFrameDetector()
    candidates = detector.detect_frames(pdf_bytes, page_index=page_index)
    page_gray, zoom = render_page_gray(pdf_bytes, page_index=page_index, dpi=args.dpi)
    templates = load_template_catalog(
        template_root,
        target_size=args.target_size,
        margin=args.margin,
    )

    excluded_classes = {str(name).strip().lower() for name in args.exclude_class if str(name).strip()}
    if excluded_classes:
        templates = [template for template in templates if template.class_name not in excluded_classes]
    if not templates:
        raise ValueError("Nenhum template ativo após aplicar --exclude-class.")

    scored = score_candidates(
        candidates,
        page_gray,
        zoom,
        templates,
        target_size=args.target_size,
        margin=args.margin,
    )

    results = []
    contact_rows = []
    for candidate, (score, crop) in zip(candidates, scored):
        item = score.to_dict()
        item.update(
            {
                "page": candidate.page,
                "frame_bbox": [round(v, 3) for v in candidate.frame_bbox.to_list()],
                "symbol_bbox": [round(v, 3) for v in candidate.symbol_bbox.to_list()],
                "cell_count": len(candidate.cells),
            }
        )
        if crop is not None:
            crop_path = crop_dir / f"{candidate.candidate_id}.png"
            cv2.imwrite(str(crop_path), crop)
            item["crop_path"] = str(crop_path.relative_to(PROJECT_ROOT))
        else:
            item["crop_path"] = None
        results.append(item)
        contact_rows.append((candidate, score, crop))

    classes = sorted({template.class_name for template in templates})
    payload = {
        "schema_version": 3,
        "phase": "symbol_scoring",
        "decision_applied": False,
        "case_id": case_id,
        "pdf": str(pdf_path),
        "page": page_index + 1,
        "candidate_count": len(candidates),
        "template_root": str(template_root),
        "template_count": len(templates),
        "classes": classes,
        "excluded_classes": sorted(excluded_classes),
        "config": {
            "dpi": args.dpi,
            "target_size": args.target_size,
            "margin": args.margin,
            "score_components": list(SCORE_COMPONENTS),
            "structure_descriptor": "occupancy_grid_plus_horizontal_vertical_projections",
            "template_score": "mean_of_score_components",
            "class_aggregation": "best_template_score_per_class",
        },
        "results": results,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "symbol_scores.json"
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    contact_path = output_dir / "symbol_contact_sheet.png"
    cv2.imwrite(str(contact_path), _build_contact_sheet(contact_rows))

    print(f"case={case_id}")
    print(f"candidates={len(candidates)}")
    print(f"templates={len(templates)}")
    print(f"classes={','.join(classes)}")
    print(f"score_components={','.join(SCORE_COMPONENTS)}")
    if excluded_classes:
        print(f"excluded_classes={','.join(sorted(excluded_classes))}")
    print("decision_applied=False")
    print(f"output={output_path}")
    print(f"contact_sheet={contact_path}")
    print(f"crops={crop_dir}")


if __name__ == "__main__":
    main()
