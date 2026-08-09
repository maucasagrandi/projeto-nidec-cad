"""Gera artefatos iniciais para revisar um caso GD&T.

Uso preferido:
    python validation/gdt/scripts/bootstrap_case.py \
        --case validation/gdt/cases/case_41_rev8.json

Uso alternativo:
    python validation/gdt/scripts/bootstrap_case.py \
        --pdf "CAD_Review_Test_Battery_V1/2. Comparison Analysis/41/13358002_REV_8_draw_2.pdf" \
        --case-id case_41_rev8

Saidas:
    validation/gdt/outputs/<case-id>/candidates.json
    validation/gdt/outputs/<case-id>/candidates.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.gdt.detector import GdtFrameDetector


def _candidate_to_dict(candidate) -> dict:
    return {
        "candidate_id": candidate.candidate_id,
        "page": candidate.page,
        "frame_bbox": [round(v, 3) for v in candidate.frame_bbox.to_list()],
        "symbol_bbox": [round(v, 3) for v in candidate.symbol_bbox.to_list()],
        "cell_count": len(candidate.cells),
        "cells": [
            {
                "index": idx,
                "bbox": [round(v, 3) for v in cell.bbox.to_list()],
                "texts": list(cell.texts),
            }
            for idx, cell in enumerate(candidate.cells)
        ],
        "confidence_score": round(candidate.confidence_score, 4),
    }


def _resolve_case(args) -> tuple[str, Path, int, dict]:
    if args.case:
        config_path = Path(args.case)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        case_id = str(config["case_id"])
        pdf_path = Path(config["pdf"])
        page_index = int(config.get("page_index", 0))
        return case_id, pdf_path, page_index, config

    if not args.pdf or not args.case_id:
        raise SystemExit("Use --case OU informe --pdf e --case-id.")

    return args.case_id, Path(args.pdf), args.page_index, {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", help="JSON versionado em validation/gdt/cases")
    parser.add_argument("--pdf")
    parser.add_argument("--case-id")
    parser.add_argument("--page-index", type=int, default=0)
    parser.add_argument("--output-root", default="validation/gdt/outputs")
    args = parser.parse_args()

    case_id, pdf_path, page_index, case_config = _resolve_case(args)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF não encontrado: {pdf_path}")

    pdf_bytes = pdf_path.read_bytes()

    detector = GdtFrameDetector()
    candidates = detector.detect_frames(pdf_bytes, page_index=page_index)
    debug_image = detector.render_debug_image(candidates)

    output_dir = Path(args.output_root) / case_id
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": 1,
        "case_id": case_id,
        "pdf": str(pdf_path),
        "page": page_index + 1,
        "page_index": page_index,
        "candidate_count": len(candidates),
        "expected": case_config.get("expected"),
        "detector_config": {
            "min_cells": detector.min_cells,
            "max_cells": detector.max_cells,
            "min_frame_height": detector.min_frame_height,
            "max_frame_height": detector.max_frame_height,
            "min_frame_width": detector.min_frame_width,
            "max_frame_width": detector.max_frame_width,
            "min_cell_width": detector.min_cell_width,
            "symbol_aspect_min": detector.symbol_aspect_min,
            "symbol_aspect_max": detector.symbol_aspect_max,
            "line_tolerance": detector.line_tolerance,
            "merge_gap": detector.merge_gap,
            "endpoint_tolerance": detector.endpoint_tolerance,
            "page_border_margin": detector.page_border_margin,
        },
        "candidates": [_candidate_to_dict(candidate) for candidate in candidates],
    }

    json_path = output_dir / "candidates.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    image_path = output_dir / "candidates.png"
    if debug_image is not None:
        debug_image.save(image_path)

    print(f"case={case_id}")
    print(f"pdf={pdf_path}")
    print(f"candidates={len(candidates)}")
    if case_config.get("expected"):
        print(f"expected_frames={case_config['expected'].get('frame_count')}")
    print(f"json={json_path}")
    if debug_image is not None:
        print(f"image={image_path}")


if __name__ == "__main__":
    main()
