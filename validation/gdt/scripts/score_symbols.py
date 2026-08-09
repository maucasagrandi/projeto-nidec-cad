"""Pontua visualmente a primeira célula dos candidatos GD&T.

Uso:
    python validation/gdt/scripts/score_symbols.py \
      --case validation/gdt/cases/case_41_rev8.json \
      --templates assets/gdt/templates

Saídas:
    validation/gdt/outputs/<case_id>/symbol_scores.json
    validation/gdt/outputs/<case_id>/symbol_crops/*.png

Importante: este script NÃO aplica threshold e NÃO decide se um candidato é
GD&T real. Ele apenas produz scores por classe/template para análise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gdt.detector import GdtFrameDetector
from src.gdt.symbol_classifier import (
    DEFAULT_MARGIN,
    DEFAULT_TARGET_SIZE,
    load_template_catalog,
    render_page_gray,
    score_candidates,
)


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--templates", default="assets/gdt/templates")
    parser.add_argument("--target-size", type=int, default=DEFAULT_TARGET_SIZE)
    parser.add_argument("--margin", type=int, default=DEFAULT_MARGIN)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--output-root", default="validation/gdt/outputs")
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
    scored = score_candidates(
        candidates,
        page_gray,
        zoom,
        templates,
        target_size=args.target_size,
        margin=args.margin,
    )

    results = []
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

    classes = sorted({template.class_name for template in templates})
    payload = {
        "schema_version": 1,
        "phase": "symbol_scoring",
        "decision_applied": False,
        "case_id": case_id,
        "pdf": str(pdf_path),
        "page": page_index + 1,
        "candidate_count": len(candidates),
        "template_root": str(template_root),
        "template_count": len(templates),
        "classes": classes,
        "config": {
            "dpi": args.dpi,
            "target_size": args.target_size,
            "margin": args.margin,
            "representations": ["gray", "binary", "edges"],
            "class_aggregation": "best_template_mean_of_representations",
        },
        "results": results,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "symbol_scores.json"
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"case={case_id}")
    print(f"candidates={len(candidates)}")
    print(f"templates={len(templates)}")
    print(f"classes={','.join(classes)}")
    print("decision_applied=False")
    print(f"output={output_path}")
    print(f"crops={crop_dir}")


if __name__ == "__main__":
    main()
