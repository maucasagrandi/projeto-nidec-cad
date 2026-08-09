"""Avalia os scores visuais da Fase 2 contra a baseline geométrica.

Este script NÃO calibra threshold. Ele mede apenas:
- acerto de ranking da classe nos quadros reais localizados;
- scores/margens dos quadros reais;
- comportamento dos candidatos geométricos extras;
- existência (ou não) de gap entre reais e extras.

Uso:
    python validation/gdt/scripts/evaluate_symbol_scores.py \
      --scores validation/gdt/outputs/case_41_rev8/symbol_scores.json \
      --ground-truth validation/gdt/ground_truth/case_41_rev8.json \
      --geometry-baseline validation/gdt/baselines/case_41_rev8.geometry.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _numeric(rows: list[dict], field: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(field)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--geometry-baseline", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    scores_path = _project_path(args.scores)
    gt_path = _project_path(args.ground_truth)
    baseline_path = _project_path(args.geometry_baseline)

    scores = _load(scores_path)
    ground_truth = _load(gt_path)
    baseline = _load(baseline_path)

    gt_by_id = {item["id"]: item for item in ground_truth.get("frames", [])}
    score_by_candidate = {item["candidate_id"]: item for item in scores.get("results", [])}

    real_rows = []
    correct = 0
    total = 0
    for match in baseline.get("matches", []):
        gt_id = match["ground_truth_id"]
        candidate_id = match.get("candidate_id")
        if not candidate_id or gt_id not in gt_by_id or candidate_id not in score_by_candidate:
            continue

        expected = str(gt_by_id[gt_id]["characteristic"]).lower()
        scored = score_by_candidate[candidate_id]
        predicted = scored.get("best_class")
        is_correct = predicted == expected
        correct += int(is_correct)
        total += 1
        real_rows.append(
            {
                "ground_truth_id": gt_id,
                "candidate_id": candidate_id,
                "expected_class": expected,
                "best_class": predicted,
                "best_score": scored.get("best_score"),
                "second_best_class": scored.get("second_best_class"),
                "second_best_score": scored.get("second_best_score"),
                "margin": scored.get("margin"),
                "class_scores": scored.get("class_scores", {}),
                "ranking_correct": is_correct,
            }
        )

    unmatched = set(baseline.get("unmatched_candidates", []))
    extra_rows = []
    for candidate_id in sorted(unmatched):
        scored = score_by_candidate.get(candidate_id)
        if not scored:
            continue
        extra_rows.append(
            {
                "candidate_id": candidate_id,
                "best_class": scored.get("best_class"),
                "best_score": scored.get("best_score"),
                "second_best_class": scored.get("second_best_class"),
                "second_best_score": scored.get("second_best_score"),
                "margin": scored.get("margin"),
                "class_scores": scored.get("class_scores", {}),
            }
        )

    ranking_accuracy = correct / total if total else 0.0

    real_best = _numeric(real_rows, "best_score")
    extra_best = _numeric(extra_rows, "best_score")
    real_margin = _numeric(real_rows, "margin")
    extra_margin = _numeric(extra_rows, "margin")

    min_real_best = min(real_best) if real_best else None
    max_extra_best = max(extra_best) if extra_best else None
    min_real_margin = min(real_margin) if real_margin else None
    max_extra_margin = max(extra_margin) if extra_margin else None

    best_gap = (
        min_real_best - max_extra_best
        if min_real_best is not None and max_extra_best is not None
        else None
    )
    margin_gap = (
        min_real_margin - max_extra_margin
        if min_real_margin is not None and max_extra_margin is not None
        else None
    )

    payload = {
        "schema_version": 2,
        "phase": "symbol_scoring_evaluation",
        "decision_threshold_calibrated": False,
        "case_id": scores.get("case_id"),
        "real_frame_count": total,
        "extra_candidate_count": len(extra_rows),
        "ranking_metrics": {
            "correct": correct,
            "total": total,
            "accuracy": round(ranking_accuracy, 4),
            "note": "Top-ranked class on geometrically matched real frames; not final acceptance accuracy.",
        },
        "separation_diagnostics": {
            "min_real_best_score": round(min_real_best, 6) if min_real_best is not None else None,
            "max_extra_best_score": round(max_extra_best, 6) if max_extra_best is not None else None,
            "best_score_gap": round(best_gap, 6) if best_gap is not None else None,
            "clean_best_score_separation": bool(best_gap is not None and best_gap > 0),
            "min_real_margin": round(min_real_margin, 6) if min_real_margin is not None else None,
            "max_extra_margin": round(max_extra_margin, 6) if max_extra_margin is not None else None,
            "margin_gap": round(margin_gap, 6) if margin_gap is not None else None,
            "clean_margin_separation": bool(margin_gap is not None and margin_gap > 0),
            "note": "Diagnostics only. Do not convert these gaps into production thresholds from a single CAD.",
        },
        "real_frames": real_rows,
        "extra_candidates": extra_rows,
    }

    output_path = _project_path(
        args.output or f"validation/gdt/outputs/{scores.get('case_id', 'case')}/symbol_evaluation.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"real_frames={total}")
    print(f"ranking_correct={correct}")
    print(f"ranking_accuracy={ranking_accuracy:.3f}")
    print(f"extra_candidates={len(extra_rows)}")
    if best_gap is not None:
        print(f"best_score_gap={best_gap:.3f}")
    if margin_gap is not None:
        print(f"margin_gap={margin_gap:.3f}")
    print("decision_threshold_calibrated=False")
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
