"""Executa a regressão da Fase 4 no caso 41.

Objetivo: adicionar novas classes ao catálogo SEM perder o ranking correto dos
3 Position + 3 Profile já validados no caso 41.

Este script NÃO calibra threshold e NÃO valida as novas classes em CAD real.
Ele apenas testa competição entre classes no caso de referência.

Uso:
    python validation/gdt/scripts/run_phase4_regression.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCRIPTS_DIR = Path(__file__).resolve().parent
CASE_ID = "case_41_rev8"
CASE_PATH = PROJECT_ROOT / "validation" / "gdt" / "cases" / f"{CASE_ID}.json"
GT_PATH = PROJECT_ROOT / "validation" / "gdt" / "ground_truth" / f"{CASE_ID}.json"
GEOMETRY_BASELINE = PROJECT_ROOT / "validation" / "gdt" / "baselines" / f"{CASE_ID}.geometry.json"
OUTPUT_DIR = PROJECT_ROOT / "validation" / "gdt" / "outputs" / "phase4" / CASE_ID
SCORES_PATH = OUTPUT_DIR / "symbol_scores.json"
EVALUATION_PATH = OUTPUT_DIR / "symbol_evaluation.json"

EXPECTED_ACTIVE_CLASSES = {
    "position",
    "profile",
    "straightness",
    "flatness",
    "circularity",
    "cylindricity",
}


def _run(command: list[str]) -> None:
    print("\n$ " + " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    # 1) Copia/prepara as referências versionadas de cotas/ para o catálogo local.
    _run([sys.executable, str(SCRIPTS_DIR / "sync_phase4_templates.py")])

    # 2) Pontua o caso 41 com as seis classes válidas. O círculo que era usado
    # como controle negativo é excluído porque agora ele é um símbolo válido de
    # Circularity e não pode competir como classe negativa.
    _run(
        [
            sys.executable,
            str(SCRIPTS_DIR / "score_symbols.py"),
            "--case",
            str(CASE_PATH),
            "--templates",
            str(PROJECT_ROOT / "assets" / "gdt" / "templates"),
            "--exclude-class",
            "negative_controls",
            "--output-root",
            str(PROJECT_ROOT / "validation" / "gdt" / "outputs" / "phase4"),
        ]
    )

    # 3) Avalia somente contra o ground truth independente já existente.
    _run(
        [
            sys.executable,
            str(SCRIPTS_DIR / "evaluate_symbol_scores.py"),
            "--scores",
            str(SCORES_PATH),
            "--ground-truth",
            str(GT_PATH),
            "--geometry-baseline",
            str(GEOMETRY_BASELINE),
            "--output",
            str(EVALUATION_PATH),
        ]
    )

    scores = _load(SCORES_PATH)
    evaluation = _load(EVALUATION_PATH)

    active_classes = set(scores.get("classes", []))
    missing_classes = sorted(EXPECTED_ACTIVE_CLASSES - active_classes)
    metrics = evaluation.get("ranking_metrics", {})
    correct = int(metrics.get("correct", 0))
    total = int(metrics.get("total", 0))
    accuracy = float(metrics.get("accuracy", 0.0))

    ranking_pass = total == 6 and correct == 6 and accuracy == 1.0
    catalog_pass = not missing_classes
    passed = ranking_pass and catalog_pass

    print("\n=== Phase 4 regression summary ===")
    print("active_classes=" + ",".join(sorted(active_classes)))
    print("negative_controls_active=False")
    print(f"case41_ranking={correct}/{total}")
    print(f"case41_ranking_accuracy={accuracy:.3f}")
    print("threshold_calibrated=False")
    if missing_classes:
        print("missing_classes=" + ",".join(missing_classes))
    print(f"phase4_case41_regression={'PASS' if passed else 'FAIL'}")
    print(f"scores={SCORES_PATH}")
    print(f"evaluation={EVALUATION_PATH}")
    print(f"contact_sheet={OUTPUT_DIR / 'symbol_contact_sheet.png'}")

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
