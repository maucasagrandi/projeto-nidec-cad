"""Diagnóstico de similaridade entre templates de classes GD&T diferentes.

Este script NÃO mede acurácia em CAD e NÃO calibra threshold. Ele apenas mede,
para cada template, quão perto a classe visual concorrente mais semelhante fica
do próprio template. É uma ferramenta para descobrir pares potencialmente
confundíveis antes da validação em desenhos reais.

Uso:
    python validation/gdt/scripts/analyze_template_competition.py \
      --templates assets/gdt/templates \
      --exclude-class negative_controls
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

from src.gdt.symbol_classifier import load_template_catalog, score_crop


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--templates", default="assets/gdt/templates")
    parser.add_argument("--exclude-class", action="append", default=[])
    parser.add_argument(
        "--output",
        default="validation/gdt/outputs/phase4/template_competition.json",
    )
    args = parser.parse_args()

    template_root = _project_path(args.templates)
    excluded = {
        str(name).strip().lower()
        for name in args.exclude_class
        if str(name).strip()
    }

    templates = load_template_catalog(template_root)
    templates = [item for item in templates if item.class_name not in excluded]
    if len(templates) < 2:
        raise ValueError("São necessários templates de pelo menos duas classes.")

    rows: list[dict] = []
    for template in templates:
        gray = cv2.imread(template.path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f"Não foi possível ler template: {template.path}")

        self_scores, _ = score_crop(gray, [template])
        self_score = float(self_scores[template.class_name])

        competitors = [
            item for item in templates if item.class_name != template.class_name
        ]
        competitor_scores, _ = score_crop(gray, competitors)
        best_class, best_cross_score = max(
            competitor_scores.items(),
            key=lambda item: item[1],
        )
        gap = self_score - float(best_cross_score)

        rows.append(
            {
                "class_name": template.class_name,
                "template_name": template.template_name,
                "template_path": str(Path(template.path).relative_to(PROJECT_ROOT)),
                "self_score": round(self_score, 6),
                "closest_cross_class": best_class,
                "closest_cross_class_score": round(float(best_cross_score), 6),
                "self_vs_cross_class_gap": round(gap, 6),
                "cross_class_scores": {
                    key: round(float(value), 6)
                    for key, value in sorted(competitor_scores.items())
                },
            }
        )

    rows.sort(key=lambda row: row["self_vs_cross_class_gap"])
    min_gap = min(row["self_vs_cross_class_gap"] for row in rows)

    payload = {
        "schema_version": 1,
        "phase": "phase4_template_competition_diagnostic",
        "decision_threshold_calibrated": False,
        "template_count": len(templates),
        "class_count": len({item.class_name for item in templates}),
        "excluded_classes": sorted(excluded),
        "minimum_self_vs_cross_class_gap": round(float(min_gap), 6),
        "note": (
            "Template-to-template diagnostic only. A large gap does not prove CAD "
            "classification accuracy; a small gap identifies a pair worth reviewing."
        ),
        "templates": rows,
    }

    output_path = _project_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("phase=phase4_template_competition_diagnostic")
    print(f"templates={len(templates)}")
    print(f"classes={payload['class_count']}")
    print(f"minimum_self_vs_cross_class_gap={min_gap:.3f}")
    print("closest_pairs:")
    for row in rows[: min(10, len(rows))]:
        print(
            f"  {row['class_name']}/{row['template_name']} -> "
            f"{row['closest_cross_class']} "
            f"score={row['closest_cross_class_score']:.3f} "
            f"gap={row['self_vs_cross_class_gap']:.3f}"
        )
    print("decision_threshold_calibrated=False")
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
