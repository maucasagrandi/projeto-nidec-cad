"""Executa o ciclo completo de diagnóstico da Fase 2 sem calibrar threshold.

Uso:
    python validation/gdt/scripts/run_phase2.py --case validation/gdt/cases/case_41_rev8.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = Path(__file__).resolve().parent


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
    args = parser.parse_args()

    case_path = _project_path(args.case)
    config = json.loads(case_path.read_text(encoding="utf-8"))
    case_id = str(config["case_id"])

    ground_truth = PROJECT_ROOT / "validation" / "gdt" / "ground_truth" / f"{case_id}.json"
    baseline = PROJECT_ROOT / "validation" / "gdt" / "baselines" / f"{case_id}.geometry.json"
    scores = PROJECT_ROOT / "validation" / "gdt" / "outputs" / case_id / "symbol_scores.json"

    if not ground_truth.exists():
        raise FileNotFoundError(f"Ground truth não encontrado: {ground_truth}")
    if not baseline.exists():
        raise FileNotFoundError(f"Baseline geométrica não encontrada: {baseline}")

    _run([
        sys.executable,
        str(SCRIPTS_DIR / "score_symbols.py"),
        "--case",
        str(case_path),
        "--templates",
        str(_project_path(args.templates)),
    ])

    _run([
        sys.executable,
        str(SCRIPTS_DIR / "evaluate_symbol_scores.py"),
        "--scores",
        str(scores),
        "--ground-truth",
        str(ground_truth),
        "--geometry-baseline",
        str(baseline),
    ])

    print("\nFase 2 diagnóstica concluída.")
    print(f"scores={scores}")
    print(f"contact_sheet={scores.parent / 'symbol_contact_sheet.png'}")
    print(f"evaluation={scores.parent / 'symbol_evaluation.json'}")
    print("threshold_calibrated=False")


if __name__ == "__main__":
    main()
