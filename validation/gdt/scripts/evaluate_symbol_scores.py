"""Avalia os scores visuais contra a baseline geometrica.

A avaliacao distingue classes SUPORTADAS pelo catalogo atual de classes ainda
nao suportadas (por exemplo ``unknown`` durante a Fase 3). Classes nao
suportadas continuam registradas, mas nao contam como erro de ranking.

Este script NAO calibra threshold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
NEGATIVE_CLASS_NAMES = {"negative_controls", "negative", "background"}


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

    supported_classes = {
        str(name).lower()
        for name in scores.get("classes", [])
        if str(name).lower() not in NEGATIVE_CLASS_NAMES
    }

    gt_by_id = {item["id"]: item for item in ground_truth.get("frames", [])}
    score_by_candidate = {item["candidate_id"]: item for item in scores.get("results", [])}

    supported_rows: list[dict] = []
    unsupported_rows: list[dict] = []
    correct = 0

    for match in baseline.get("matches", []):
        gt_id = match["ground_truth_id"]
        candidate_id = match.get("candidate_id")
        if not candidate_id or gt_id not in gt_by_id or candidate_id not in score_by_candidate:
            continue

        expected = str(gt_by_id[gt_id]["characteristic"]).lower()
        scored = score_by_candidate[candidate_id]
        predicted = scored.get("best_class")
        row = {
            "ground_truth_id": gt_id,
            "candidate_id": candidate_id,
            "expected_class": expected,
            "best_class": predicted,
            "best_score": scored.get("best_score"),
            "second_best_class": scored.get("second_best_class"),
            "second_best_score": scored.get("second_best_score"),
            "margin": scored.get("margin"),
            "class_scores": scored.get("class_scores", {}),
        }

        if expected in supported_classes:
            is_correct = predicted == expected
            correct += int(is_correct)
            row["ranking_correct"] = is_correct
            supported_rows.append(row)
        else:
            row["ranking_correct"] = None
            row["evaluation_status"] = "unsupported_class_not_scored_as_error"
            unsupported_rows.append(row)

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

    total_supported = len(supported_rows)
    ranking_accuracy = correct / total_supported if total_supported else 0.0

    # Separacao e calculada SOMENTE com frames de classes suportadas. Um simbolo
    # ainda desconhecido nao deve ser tratado como falso positivo/erro do
    # classificador Position/Profile.
    real_best = _numeric(supported_rows, "best_score")
    extra_best = _numeric(extra_rows, "best_score")
    real_margin = _numeric(supported_rows, "margin")
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
        "schema_version": 3,
        "phase": "symbol_scoring_evaluation",
        "decision_threshold_calibrated": False,
        "case_id": scores.get("case_id"),
        "supported_classes": sorted(supported_classes),
        "matched_real_frame_count": len(supported_rows) + len(unsupported_rows),
        "supported_real_frame_count": len(supported_rows),
        "unsupported_real_frame_count": len(unsupported_rows),
        "extra_candidate_count": len(extra_rows),
        "ranking_metrics": {
            "correct": correct,
            "total": total_supported,
            "accuracy": round(ranking_accuracy, 4),
            "note": "Only currently supported ground-truth classes count toward ranking accuracy.",
        },
        "separation_diagnostics": {
            "population": "supported_real_frames_vs_extra_geometric_candidates",
            "min_supported_real_best_score": round(min_real_best, 6) if min_real_best is not None else None,
            "max_extra_best_score": round(max_extra_best, 6) if max_extra_best is not None else None,
            "best_score_gap": round(best_gap, 6) if best_gap is not None else None,
            "clean_best_score_separation": bool(best_gap is not None and best_gap > 0),
            "min_supported_real_margin": round(min_real_margin, 6) if min_real_margin is not None else None,
            "max_extra_margin": round(max_extra_margin, 6) if max_extra_margin is not None else None,
            "margin_gap": round(margin_gap, 6) if margin_gap is not None else None,
            "clean_margin_separation": bool(margin_gap is not None and margin_gap > 0),
            "note": "Diagnostics only. Do not derive production thresholds from one CAD.",
        },
        "supported_real_frames": supported_rows,
        "unsupported_real_frames": unsupported_rows,
        "extra_candidates": extra_rows,
    }

    output_path = _project_path(
        args.output or f"validation/gdt/outputs/{scores.get('case_id', 'case')}/symbol_evaluation.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"matched_real_frames={len(supported_rows) + len(unsupported_rows)}")
    print(f"supported_real_frames={len(supported_rows)}")
    print(f"unsupported_real_frames={len(unsupported_rows)}")
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
