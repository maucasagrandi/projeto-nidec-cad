"""Diagnóstico da Fase 5A: leitura textual das células GD&T no caso 41.

Este script NÃO valida a exatidão de tolerâncias/datums, pois ainda não existe
ground truth independente para o conteúdo interno dos quadros. Ele serve para
mostrar exatamente o que PyMuPDF já extraiu por célula e o que o parser
conseguiu estruturar sem OCR/LLM.

Pré-requisito: a regressão da Fase 4 deve ter gerado ``symbol_scores.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gdt.detector import GdtFrameDetector
from src.gdt.frame_parser import parse_feature_control_frame

CASE_ID = "case_41_rev8"
CASE_PATH = PROJECT_ROOT / "validation" / "gdt" / "cases" / f"{CASE_ID}.json"
GEOMETRY_BASELINE = PROJECT_ROOT / "validation" / "gdt" / "baselines" / f"{CASE_ID}.geometry.json"
PHASE4_SCORES = PROJECT_ROOT / "validation" / "gdt" / "outputs" / "phase4" / CASE_ID / "symbol_scores.json"
OUTPUT_DIR = PROJECT_ROOT / "validation" / "gdt" / "outputs" / "phase5" / CASE_ID
OUTPUT_PATH = OUTPUT_DIR / "text_parsing_diagnostic.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    if not PHASE4_SCORES.exists():
        raise FileNotFoundError(
            "symbol_scores.json da Fase 4 não encontrado. Rode "
            "python validation/gdt/scripts/run_phase4_regression.py primeiro."
        )

    case = _load(CASE_PATH)
    baseline = _load(GEOMETRY_BASELINE)
    phase4 = _load(PHASE4_SCORES)

    pdf_path = PROJECT_ROOT / case["pdf"]
    page_index = int(case.get("page_index", 0))
    pdf_bytes = pdf_path.read_bytes()

    detector = GdtFrameDetector()
    candidates = detector.detect_frames(pdf_bytes, page_index=page_index)

    class_by_candidate = {
        row["candidate_id"]: row.get("best_class")
        for row in phase4.get("results", [])
    }
    benchmark_ids = {
        row["candidate_id"]
        for row in baseline.get("matches", [])
        if row.get("candidate_id")
    }

    results = []
    for candidate in candidates:
        parsed = parse_feature_control_frame(
            candidate,
            characteristic=class_by_candidate.get(candidate.candidate_id),
        )
        item = parsed.to_dict()
        item["benchmark_real_frame"] = candidate.candidate_id in benchmark_ids
        item["frame_bbox"] = [round(v, 3) for v in candidate.frame_bbox.to_list()]
        results.append(item)

    real_rows = [row for row in results if row["benchmark_real_frame"]]
    tolerance_resolved = sum(row["tolerance_value"] is not None for row in real_rows)
    diameter_text_resolved = sum(row["diameter_zone"] is True for row in real_rows)

    payload = {
        "schema_version": 1,
        "phase": "phase5_text_parsing_diagnostic",
        "case_id": CASE_ID,
        "validation_status": "DIAGNOSTIC_ONLY",
        "ground_truth_for_internal_content": False,
        "ocr_used": False,
        "llm_used": False,
        "candidate_count": len(results),
        "benchmark_real_frame_count": len(real_rows),
        "benchmark_real_tolerance_text_resolved": tolerance_resolved,
        "benchmark_real_diameter_text_resolved": diameter_text_resolved,
        "results": results,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("phase=phase5_text_parsing_diagnostic")
    print("validation_status=DIAGNOSTIC_ONLY")
    print("ground_truth_for_internal_content=False")
    print("ocr_used=False")
    print("llm_used=False")
    print(f"candidates={len(results)}")
    print(f"benchmark_real_frames={len(real_rows)}")
    print(f"benchmark_real_tolerance_text_resolved={tolerance_resolved}/{len(real_rows)}")
    print(f"benchmark_real_diameter_text_resolved={diameter_text_resolved}/{len(real_rows)}")
    print("\nbenchmark_real_frame_parsing:")
    for row in real_rows:
        print(
            f"  {row['candidate_id']} characteristic={row['characteristic']} "
            f"cells={row['cell_texts']} tolerance_raw={row['tolerance_raw']} "
            f"tolerance_value={row['tolerance_value']} diameter_zone={row['diameter_zone']} "
            f"datums={row['referenced_datums']} unresolved={row['unresolved_tokens']}"
        )
    print(f"\noutput={OUTPUT_PATH}")


if __name__ == "__main__":
    main()
