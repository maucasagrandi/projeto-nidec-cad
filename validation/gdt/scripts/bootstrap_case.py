"""Gera artefatos iniciais para anotar um caso GD&T manualmente.

Uso:
    python validation/gdt/scripts/bootstrap_case.py \
        --pdf "CAD_Review_Test_Battery_V1/2. Comparison Analysis/41/13358002_REV_8_draw_2.pdf" \
        --case-id case_41_rev8

Saidas:
    validation/gdt/outputs/<case-id>/candidates.json
    validation/gdt/outputs/<case-id>/candidates.png

O JSON lista todos os candidatos e seus bboxes. A imagem numera visualmente
cada candidato. Esses dois arquivos servem para montar o ground truth sem
inventar coordenadas.
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--page-index", type=int, default=0)
    parser.add_argument("--output-root", default="validation/gdt/outputs")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    pdf_bytes = pdf_path.read_bytes()

    detector = GdtFrameDetector()
    candidates = detector.detect_frames(pdf_bytes, page_index=args.page_index)
    debug_image = detector.render_debug_image(candidates)

    output_dir = Path(args.output_root) / args.case_id
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "case_id": args.case_id,
        "pdf": str(pdf_path),
        "page_index": args.page_index,
        "candidate_count": len(candidates),
        "candidates": [_candidate_to_dict(candidate) for candidate in candidates],
    }

    json_path = output_dir / "candidates.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    image_path = output_dir / "candidates.png"
    if debug_image is not None:
        debug_image.save(image_path)

    print(f"case={args.case_id}")
    print(f"candidates={len(candidates)}")
    print(f"json={json_path}")
    if debug_image is not None:
        print(f"image={image_path}")


if __name__ == "__main__":
    main()
