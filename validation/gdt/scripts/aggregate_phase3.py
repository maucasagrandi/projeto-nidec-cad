"""Agrega avaliações de símbolos de vários casos da Fase 3.

O script não escolhe threshold. Ele mostra se ranking e separação observados no
caso 41 persistem quando adicionamos novos CADs independentes.

Exemplo:
    python validation/gdt/scripts/aggregate_phase3.py \
      validation/gdt/outputs/case_41_rev8/symbol_evaluation.json \
      validation/gdt/outputs/case_42/symbol_evaluation.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _finite(values):
    return [float(v) for v in values if v is not None]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evaluations", nargs="+")
    parser.add_argument("--output", default="validation/gdt/outputs/phase3_aggregate.json")
    args = parser.parse_args()

    cases = []
    real_scores = []
    real_margins = []
    extra_scores = []
    extra_margins = []
    total_real = 0
    total_correct = 0
    total_extras = 0

    for raw in args.evaluations:
        path = _project_path(raw)
        payload = json.loads(path.read_text(encoding="utf-8"))
        ranking = payload.get("ranking_metrics", {})
        real_rows = payload.get("real_frames", [])
        extra_rows = payload.get("extra_candidates", [])

        total_real += int(ranking.get("total", len(real_rows)))
        total_correct += int(ranking.get("correct", 0))
        total_extras += len(extra_rows)

        real_scores.extend(_finite(row.get("best_score") for row in real_rows))
        real_margins.extend(_finite(row.get("margin") for row in real_rows))
        extra_scores.extend(_finite(row.get("best_score") for row in extra_rows))
        extra_margins.extend(_finite(row.get("margin") for row in extra_rows))

        cases.append({
            "case_id": payload.get("case_id"),
            "source": str(path),
            "real_frames": len(real_rows),
            "ranking_correct": int(ranking.get("correct", 0)),
            "ranking_accuracy": ranking.get("accuracy"),
            "extra_candidates": len(extra_rows),
            "separation": payload.get("separation"),
        })

    ranking_accuracy = total_correct / total_real if total_real else 0.0
    min_real_score = min(real_scores) if real_scores else None
    max_extra_score = max(extra_scores) if extra_scores else None
    min_real_margin = min(real_margins) if real_margins else None
    max_extra_margin = max(extra_margins) if extra_margins else None

    score_gap = (
        min_real_score - max_extra_score
        if min_real_score is not None and max_extra_score is not None
        else None
    )
    margin_gap = (
        min_real_margin - max_extra_margin
        if min_real_margin is not None and max_extra_margin is not None
        else None
    )

    payload = {
        "schema_version": 1,
        "phase": "phase3_generalization_aggregate",
        "decision_threshold_calibrated": False,
        "case_count": len(cases),
        "cases": cases,
        "aggregate": {
            "real_frames": total_real,
            "ranking_correct": total_correct,
            "ranking_accuracy": round(ranking_accuracy, 4),
            "extra_candidates": total_extras,
            "min_real_best_score": round(min_real_score, 6) if min_real_score is not None else None,
            "max_extra_best_score": round(max_extra_score, 6) if max_extra_score is not None else None,
            "best_score_gap": round(score_gap, 6) if score_gap is not None else None,
            "min_real_margin": round(min_real_margin, 6) if min_real_margin is not None else None,
            "max_extra_margin": round(max_extra_margin, 6) if max_extra_margin is not None else None,
            "margin_gap": round(margin_gap, 6) if margin_gap is not None else None,
        },
        "interpretation": {
            "positive_best_score_gap": score_gap is not None and score_gap > 0,
            "positive_margin_gap": margin_gap is not None and margin_gap > 0,
            "note": "Positive gaps across independent CADs support later threshold calibration, but this script never chooses the threshold itself.",
        },
    }

    output = _project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"cases={len(cases)}")
    print(f"real_frames={total_real}")
    print(f"ranking_accuracy={ranking_accuracy:.3f}")
    print(f"extra_candidates={total_extras}")
    print(f"best_score_gap={score_gap if score_gap is not None else 'n/a'}")
    print(f"margin_gap={margin_gap if margin_gap is not None else 'n/a'}")
    print("threshold_calibrated=False")
    print(f"output={output}")


if __name__ == "__main__":
    main()
