"""Executa um caso da Fase 3 depois da anotacao manual independente.

Fluxo:
1. valida geometria contra o ground truth manual;
2. grava ``baselines/<case_id>.geometry.json``;
3. interrompe se o recall geometrico ficar abaixo do gate;
4. roda o diagnostico visual das classes atualmente suportadas.

Uso:
    python validation/gdt/scripts/run_phase3_case.py --case validation/gdt/cases/case_28_rev22.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCRIPTS_DIR = Path(__file__).resolve().parent

from src.gdt.geometry_validation import detect_and_validate
from src.gdt.ground_truth import assert_independent_ground_truth, is_independent_ground_truth


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _run(command: list[str]) -> None:
    print("\n$ " + " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--templates", default="assets/gdt/templates")
    parser.add_argument("--minimum-recall", type=float)
    parser.add_argument("--min-iou", type=float)
    parser.add_argument("--min-overlap-smallest", type=float, default=0.50)
    parser.add_argument("--max-area-ratio", type=float, default=2.50)
    parser.add_argument(
        "--allow-geometry-fail",
        action="store_true",
        help="Continua para scoring mesmo se o gate geometrico falhar; somente para diagnostico.",
    )
    args = parser.parse_args()

    case_path = _project_path(args.case)
    config = json.loads(case_path.read_text(encoding="utf-8"))
    case_id = str(config["case_id"])
    pdf_path = _project_path(config["pdf"])
    page_index = int(config.get("page_index", 0))

    gate_cfg = config.get("geometry_gate", {})
    minimum_recall = float(
        args.minimum_recall if args.minimum_recall is not None else gate_cfg.get("minimum_recall", 0.95)
    )
    min_iou = float(args.min_iou if args.min_iou is not None else gate_cfg.get("min_iou", 0.35))

    gt_path = PROJECT_ROOT / "validation" / "gdt" / "ground_truth" / f"{case_id}.json"
    if not gt_path.exists():
        raise FileNotFoundError(
            f"Ground truth nao encontrado: {gt_path}\n"
            f"Rode primeiro: python validation/gdt/scripts/annotate_ground_truth.py --case {args.case}"
        )

    gt_payload = json.loads(gt_path.read_text(encoding="utf-8"))
    assert_independent_ground_truth(gt_payload)
    if not is_independent_ground_truth(gt_payload):
        raise SystemExit("Fase 3 exige ground truth independente.")

    candidates, metrics = detect_and_validate(
        pdf_path,
        gt_path,
        page_index=page_index,
        min_iou=min_iou,
        min_overlap_smallest=args.min_overlap_smallest,
        max_area_ratio=args.max_area_ratio,
    )

    matched_ids = {m.candidate_id for m in metrics.matches if m.matched and m.candidate_id}
    unmatched_candidates = [
        candidate.candidate_id
        for candidate in candidates
        if candidate.candidate_id not in matched_ids
    ]

    matches = [
        {
            "ground_truth_id": item.ground_truth_id,
            "candidate_id": item.candidate_id,
            "iou": round(item.iou, 6),
            "overlap_smallest": round(item.overlap_smallest, 6),
            "area_ratio": round(item.area_ratio, 6),
            "matched": item.matched,
            "match_reason": item.match_reason,
        }
        for item in metrics.matches
    ]

    baseline = {
        "schema_version": 2,
        "case_id": case_id,
        "phase": "geometry",
        "official_benchmark": True,
        "ground_truth_independent": True,
        "ground_truth": str(gt_path.relative_to(PROJECT_ROOT)),
        "pdf": str(pdf_path.relative_to(PROJECT_ROOT)) if pdf_path.is_relative_to(PROJECT_ROOT) else str(pdf_path),
        "page_index": page_index,
        "candidate_count": len(candidates),
        "ground_truth_count": metrics.ground_truth_count,
        "matching": {
            "min_iou": min_iou,
            "min_overlap_smallest": args.min_overlap_smallest,
            "max_area_ratio": args.max_area_ratio,
            "rule": "iou OR (overlap_smallest AND area_ratio)",
        },
        "metrics": metrics.to_dict(minimum_recall=minimum_recall),
        "matches": matches,
        "unmatched_candidates": unmatched_candidates,
        "notes": "Phase 3 geometry baseline generated from independent manual ground truth before symbol scoring.",
    }

    baseline_path = PROJECT_ROOT / "validation" / "gdt" / "baselines" / f"{case_id}.geometry.json"
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text(json.dumps(baseline, indent=2, ensure_ascii=False), encoding="utf-8")

    gate_passed = metrics.passes_recall_gate(minimum_recall)
    print("\n--- Geometry ---")
    print(f"case={case_id}")
    print(f"ground_truth={metrics.ground_truth_count}")
    print(f"candidates={len(candidates)}")
    print(f"TP={metrics.true_positives} FN={metrics.false_negatives} FP={metrics.false_positives}")
    print(f"recall={metrics.recall:.3f} precision={metrics.precision:.3f} f1={metrics.f1:.3f}")
    print(f"recall_gate>={minimum_recall:.3f}: {'PASS' if gate_passed else 'FAIL'}")
    print(f"baseline={baseline_path}")

    if not gate_passed and not args.allow_geometry_fail:
        print("\nScoring visual NAO executado: primeiro corrigimos o recall geometrico deste CAD.")
        raise SystemExit(2)

    _run([
        sys.executable,
        str(SCRIPTS_DIR / "run_phase2.py"),
        "--case",
        str(case_path),
        "--templates",
        str(_project_path(args.templates)),
    ])

    print("\nFase 3 do caso concluida.")
    print(f"evaluation={PROJECT_ROOT / 'validation' / 'gdt' / 'outputs' / case_id / 'symbol_evaluation.json'}")
    print("threshold_calibrated=False")


if __name__ == "__main__":
    main()
