"""Valida recall geometrico de quadros GD&T contra ground truth manual.

Uso:
    python validation/gdt/scripts/validate_geometry.py \
        --pdf "CAD_Review_Test_Battery_V1/2. Comparison Analysis/41/13358002_REV_8_draw_2.pdf" \
        --ground-truth validation/gdt/ground_truth/case_41_rev8.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.gdt.geometry_validation import detect_and_validate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--page-index", type=int, default=0)
    parser.add_argument("--min-iou", type=float, default=0.35)
    parser.add_argument("--output", default="validation/gdt/outputs/geometry_metrics.json")
    args = parser.parse_args()

    candidates, metrics = detect_and_validate(
        args.pdf,
        args.ground_truth,
        page_index=args.page_index,
        min_iou=args.min_iou,
    )

    payload = {
        "pdf": str(args.pdf),
        "page_index": args.page_index,
        "candidate_count": len(candidates),
        "metrics": metrics.to_dict(),
        "candidates": [
            {
                "candidate_id": candidate.candidate_id,
                "page": candidate.page,
                "frame_bbox": [round(v, 3) for v in candidate.frame_bbox.to_list()],
                "symbol_bbox": [round(v, 3) for v in candidate.symbol_bbox.to_list()],
                "cell_count": len(candidate.cells),
                "confidence_score": round(candidate.confidence_score, 4),
            }
            for candidate in candidates
        ],
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"candidates={len(candidates)}")
    print(f"TP={metrics.true_positives} FN={metrics.false_negatives} FP={metrics.false_positives}")
    print(f"recall={metrics.recall:.3f} precision={metrics.precision:.3f}")
    print(f"output={output}")


if __name__ == "__main__":
    main()
