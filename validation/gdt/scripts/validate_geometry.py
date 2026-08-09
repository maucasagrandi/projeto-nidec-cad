"""Valida recall geometrico de quadros GD&T contra ground truth manual.

Uso:
    python validation/gdt/scripts/validate_geometry.py \
        --pdf "CAD_Review_Test_Battery_V1/2. Comparison Analysis/41/13358002_REV_8_draw_2.pdf" \
        --ground-truth validation/gdt/ground_truth/case_41_rev8.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gdt.geometry_validation import detect_and_validate


def _project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--page-index", type=int, default=0)
    parser.add_argument("--min-iou", type=float, default=0.35)
    parser.add_argument("--minimum-recall", type=float, default=0.95)
    parser.add_argument("--fail-on-gate", action="store_true")
    parser.add_argument("--output", default="validation/gdt/outputs/geometry_metrics.json")
    args = parser.parse_args()

    pdf_path = _project_path(args.pdf)
    ground_truth_path = _project_path(args.ground_truth)
    output = _project_path(args.output)

    candidates, metrics = detect_and_validate(
        pdf_path,
        ground_truth_path,
        page_index=args.page_index,
        min_iou=args.min_iou,
    )

    gate_passed = metrics.passes_recall_gate(args.minimum_recall)
    payload = {
        "schema_version": 1,
        "phase": "geometry",
        "pdf": str(pdf_path),
        "page_index": args.page_index,
        "min_iou": args.min_iou,
        "candidate_count": len(candidates),
        "metrics": metrics.to_dict(minimum_recall=args.minimum_recall),
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

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"candidates={len(candidates)}")
    print(f"GT={metrics.ground_truth_count} TP={metrics.true_positives} "
          f"FN={metrics.false_negatives} FP={metrics.false_positives}")
    print(f"recall={metrics.recall:.3f} precision={metrics.precision:.3f} f1={metrics.f1:.3f}")
    print(f"recall_gate>={args.minimum_recall:.3f}: {'PASS' if gate_passed else 'FAIL'}")
    print(f"output={output}")

    if args.fail_on_gate and not gate_passed:
        sys.exit(2)


if __name__ == "__main__":
    main()
