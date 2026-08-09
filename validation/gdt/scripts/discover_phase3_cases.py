"""Descobre CADs úteis para a Fase 3 de generalização GD&T.

O objetivo NÃO é avaliar qualidade nem criar ground truth automaticamente.
Este script apenas percorre PDFs de desenho, executa a geometria atual com
renderização leve e produz um inventário para escolher casos variados antes de
qualquer nova calibração.

Exemplo (PowerShell):
    python validation/gdt/scripts/discover_phase3_cases.py

Saída padrão:
    validation/gdt/outputs/phase3_discovery.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gdt.detector import GdtFrameDetector


DEFAULT_ROOT = PROJECT_ROOT / "CAD_Review_Test_Battery_V1"


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _iter_drawings(root: Path, comparison_only: bool) -> list[Path]:
    pdfs = []
    for path in root.rglob("*.pdf"):
        name = path.name.lower()
        # Queremos desenhos individuais, não PDFs de comparação/relatório.
        if "_draw_" not in name:
            continue
        if comparison_only and "2. Comparison Analysis" not in str(path):
            continue
        pdfs.append(path)
    return sorted(pdfs)


def _folder_number(path: Path) -> int | None:
    for part in reversed(path.parts):
        try:
            value = int(part)
        except ValueError:
            continue
        if 1 <= value <= 999:
            return value
    return None


def _candidate_summary(candidates) -> dict:
    cell_counts = Counter(len(item.cells) for item in candidates)
    widths = [item.frame_bbox.width for item in candidates]
    heights = [item.frame_bbox.height for item in candidates]
    return {
        "candidate_count": len(candidates),
        "cell_count_distribution": {str(k): v for k, v in sorted(cell_counts.items())},
        "mean_confidence": round(
            sum(item.confidence_score for item in candidates) / len(candidates), 4
        ) if candidates else 0.0,
        "frame_width_range": [round(min(widths), 3), round(max(widths), 3)] if widths else None,
        "frame_height_range": [round(min(heights), 3), round(max(heights), 3)] if heights else None,
    }


def _recommend(rows: list[dict], limit: int) -> list[dict]:
    """Escolhe casos com candidatos e diversidade de peça/pasta.

    Não usa classe prevista nem score visual para evitar selecionar somente
    desenhos parecidos com os templates atuais.
    """
    useful = [row for row in rows if row.get("status") == "ok" and row["candidate_count"] > 0]
    useful.sort(key=lambda row: (-row["candidate_count"], row.get("folder") or 9999, row["pdf"]))

    selected: list[dict] = []
    seen_stems: set[str] = set()
    seen_folders: set[int] = set()

    # Primeiro prioriza diversidade de prefixo da peça e pasta.
    for row in useful:
        stem_prefix = Path(row["pdf"]).name.split("_REV")[0].split("_rev")[0]
        folder = row.get("folder")
        if stem_prefix in seen_stems or folder in seen_folders:
            continue
        selected.append(row)
        seen_stems.add(stem_prefix)
        if folder is not None:
            seen_folders.add(folder)
        if len(selected) >= limit:
            return selected

    # Completa caso não haja diversidade suficiente.
    for row in useful:
        if row in selected:
            continue
        selected.append(row)
        if len(selected) >= limit:
            break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--comparison-only", action="store_true", default=True)
    parser.add_argument("--all-sections", action="store_true", help="Inclui também Single Analysis.")
    parser.add_argument("--dpi", type=int, default=96, help="DPI leve; não altera geometria vetorial.")
    parser.add_argument("--recommend", type=int, default=5)
    parser.add_argument("--max-files", type=int, default=0, help="0 = sem limite.")
    parser.add_argument("--output", default="validation/gdt/outputs/phase3_discovery.json")
    args = parser.parse_args()

    root = _project_path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Bateria não encontrada: {root}")

    comparison_only = not args.all_sections
    pdfs = _iter_drawings(root, comparison_only=comparison_only)
    if args.max_files > 0:
        pdfs = pdfs[: args.max_files]

    rows: list[dict] = []
    started = time.perf_counter()

    print(f"drawings={len(pdfs)}")
    print(f"scan_root={root}")
    print(f"crop_dpi={args.dpi}")

    for index, pdf_path in enumerate(pdfs, start=1):
        relative = pdf_path.relative_to(PROJECT_ROOT)
        t0 = time.perf_counter()
        row = {
            "pdf": str(relative).replace("\\", "/"),
            "folder": _folder_number(pdf_path.parent),
        }
        try:
            detector = GdtFrameDetector(crop_dpi=args.dpi)
            candidates = detector.detect_frames(pdf_path.read_bytes(), page_index=0)
            row.update(_candidate_summary(candidates))
            row["status"] = "ok"
        except Exception as exc:  # discovery deve continuar mesmo com um PDF ruim
            row.update({
                "status": "error",
                "candidate_count": 0,
                "error": f"{type(exc).__name__}: {exc}",
            })
        row["runtime_seconds"] = round(time.perf_counter() - t0, 3)
        rows.append(row)
        print(
            f"[{index:02d}/{len(pdfs):02d}] candidates={row['candidate_count']:>3} "
            f"folder={row.get('folder')} {pdf_path.name}"
        )

    recommendations = _recommend(rows, args.recommend)
    payload = {
        "schema_version": 1,
        "phase": "phase3_case_discovery",
        "purpose": "case_selection_only",
        "threshold_calibrated": False,
        "scan": {
            "root": str(root),
            "comparison_only": comparison_only,
            "crop_dpi": args.dpi,
            "drawing_count": len(pdfs),
            "runtime_seconds": round(time.perf_counter() - started, 3),
        },
        "summary": {
            "with_candidates": sum(1 for row in rows if row.get("candidate_count", 0) > 0),
            "without_candidates": sum(1 for row in rows if row.get("status") == "ok" and row.get("candidate_count", 0) == 0),
            "errors": sum(1 for row in rows if row.get("status") == "error"),
        },
        "recommended_cases": recommendations,
        "drawings": rows,
    }

    output = _project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nRecommended for manual review:")
    for row in recommendations:
        print(f"  folder={row.get('folder')} candidates={row['candidate_count']:>3} {row['pdf']}")
    print(f"\noutput={output}")


if __name__ == "__main__":
    main()
