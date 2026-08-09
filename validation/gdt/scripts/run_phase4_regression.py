"""Executa a regressão da Fase 4 no caso 41.

Objetivo: adicionar novas classes ao catálogo SEM perder o ranking correto dos
3 Position + 3 Profile já validados no caso 41.

As classes esperadas são lidas de ``validation/gdt/reference_catalog.json``.
Assim o mesmo teste continua válido à medida que o catálogo cresce.

Este script NÃO calibra threshold e NÃO valida as novas classes em CAD real.
Ele testa competição entre classes no caso de referência e gera um diagnóstico
separado de similaridade template-a-template.

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
REFERENCE_CATALOG = PROJECT_ROOT / "validation" / "gdt" / "reference_catalog.json"
TEMPLATE_ROOT = PROJECT_ROOT / "assets" / "gdt" / "templates"
OUTPUT_DIR = PROJECT_ROOT / "validation" / "gdt" / "outputs" / "phase4" / CASE_ID
SCORES_PATH = OUTPUT_DIR / "symbol_scores.json"
EVALUATION_PATH = OUTPUT_DIR / "symbol_evaluation.json"
COMPETITION_PATH = PROJECT_ROOT / "validation" / "gdt" / "outputs" / "phase4" / "template_competition.json"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def _run(command: list[str]) -> None:
    print("\n$ " + " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _expected_active_classes() -> set[str]:
    catalog = _load(REFERENCE_CATALOG)
    classes = {"position", "profile"}
    for entry in catalog.get("entries", []):
        if str(entry.get("status", "active")).lower() != "active":
            continue
        class_name = str(entry.get("class_name", "")).strip().lower()
        if class_name:
            classes.add(class_name)
    return classes


def _has_image_template(class_name: str) -> bool:
    class_dir = TEMPLATE_ROOT / class_name
    if not class_dir.exists():
        return False
    return any(
        path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        for path in class_dir.iterdir()
    )


def _check_base_templates() -> None:
    missing = [name for name in ("position", "profile") if not _has_image_template(name)]
    if not missing:
        return

    print("missing_base_templates=" + ",".join(missing))
    print(
        "Rode register_templates.py novamente com os arquivos de Position/Profile "
        "antes da regressão da Fase 4."
    )
    raise SystemExit(2)


def main() -> None:
    expected_active_classes = _expected_active_classes()

    # Position/Profile são templates-base locais já validados no caso 41.
    _check_base_templates()

    # 1) Prepara referências e exige cobertura completa de cotas/ pelo manifesto.
    _run([sys.executable, str(SCRIPTS_DIR / "sync_phase4_templates.py")])

    # 2) Diagnóstico auxiliar de competição visual entre classes. Não é uma
    # validação em CAD e não produz threshold; serve para apontar pares próximos.
    _run(
        [
            sys.executable,
            str(SCRIPTS_DIR / "analyze_template_competition.py"),
            "--templates",
            str(TEMPLATE_ROOT),
            "--exclude-class",
            "negative_controls",
            "--output",
            str(COMPETITION_PATH),
        ]
    )

    # 3) Pontua o caso 41 com todas as classes válidas. O círculo usado no
    # experimento inicial como controle negativo é excluído porque agora ele é
    # um símbolo válido de Circularity e não pode competir como classe negativa.
    _run(
        [
            sys.executable,
            str(SCRIPTS_DIR / "score_symbols.py"),
            "--case",
            str(CASE_PATH),
            "--templates",
            str(TEMPLATE_ROOT),
            "--exclude-class",
            "negative_controls",
            "--output-root",
            str(PROJECT_ROOT / "validation" / "gdt" / "outputs" / "phase4"),
        ]
    )

    # 4) Avalia somente contra o ground truth independente já existente.
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
    missing_classes = sorted(expected_active_classes - active_classes)
    unexpected_classes = sorted(active_classes - expected_active_classes)

    metrics = evaluation.get("ranking_metrics", {})
    correct = int(metrics.get("correct", 0))
    total = int(metrics.get("total", 0))
    accuracy = float(metrics.get("accuracy", 0.0))

    ranking_pass = total == 6 and correct == 6 and accuracy == 1.0
    catalog_pass = not missing_classes and not unexpected_classes
    passed = ranking_pass and catalog_pass

    print("\n=== Phase 4 regression summary ===")
    print("expected_active_classes=" + ",".join(sorted(expected_active_classes)))
    print("active_classes=" + ",".join(sorted(active_classes)))
    print(f"expected_class_count={len(expected_active_classes)}")
    print(f"active_class_count={len(active_classes)}")
    print("negative_controls_active=False")
    print(f"case41_ranking={correct}/{total}")
    print(f"case41_ranking_accuracy={accuracy:.3f}")
    print("threshold_calibrated=False")
    if missing_classes:
        print("missing_classes=" + ",".join(missing_classes))
    if unexpected_classes:
        print("unexpected_classes=" + ",".join(unexpected_classes))
    print(f"phase4_case41_regression={'PASS' if passed else 'FAIL'}")
    print(f"template_competition={COMPETITION_PATH}")
    print(f"scores={SCORES_PATH}")
    print(f"evaluation={EVALUATION_PATH}")
    print(f"contact_sheet={OUTPUT_DIR / 'symbol_contact_sheet.png'}")

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
