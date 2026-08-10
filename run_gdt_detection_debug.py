"""Run only GD&T Phase 1 + Phase 2/4 diagnostics for one CAD PDF.

This runner intentionally does NOT call Gemini, Normas.xlsx, ISO rules, datum
consistency, or the Compliance Engine.  It exists to isolate two questions:

1. Did the geometry detector propose the real feature control frame?
2. For each proposed candidate, how did the symbol classifier rank the classes?

Outputs are written to ``<output-folder>/<pdf-stem>/``:
- debug_result.json
- candidate_diagnostics.csv
- page_NNN_candidates.png
- page_NNN_symbol_contact_sheet.png

Usage (PowerShell):
    python run_gdt_detection_debug.py --pdf "CADS\\drawing.pdf" --output-folder DEBUG_RESULTS
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cad_review.detection_diagnostics import render_detection_diagnostics
from src.gdt.detector import GdtFrameDetector
from src.gdt.symbol_classifier import load_template_catalog, render_page_gray, score_candidates


def _safe_stem(path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("._")
    return stem or "cad"


def _bbox(value: Any) -> list[float]:
    return [round(float(v), 3) for v in value.to_list()]


def _sync_phase4_templates() -> None:
    script = PROJECT_ROOT / "validation" / "gdt" / "scripts" / "sync_phase4_templates.py"
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.stdout.strip():
        print(completed.stdout.strip())
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"Phase 4 template sync failed (code={completed.returncode}): {detail}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolated Phase 1 + Phase 2/4 GD&T diagnostics")
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--output-folder", type=Path, default=PROJECT_ROOT / "DEBUG_RESULTS")
    parser.add_argument("--templates", type=Path, default=PROJECT_ROOT / "assets" / "gdt" / "templates")
    parser.add_argument("--visual-dpi", type=int, default=180)
    parser.add_argument("--symbol-dpi", type=int, default=300)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--skip-template-sync", action="store_true")
    parser.add_argument(
        "--detector-only",
        action="store_true",
        help="Skip symbol ranking completely and generate Phase-1 candidates only.",
    )
    args = parser.parse_args()

    pdf = args.pdf.resolve()
    if not pdf.exists() or not pdf.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf}")
    pdf_bytes = pdf.read_bytes()

    if not args.detector_only and not args.skip_template_sync:
        print("template_sync=START")
        _sync_phase4_templates()
        print("template_sync=PASS")

    templates = []
    template_classes: list[str] = []
    classification_error = None
    if not args.detector_only:
        try:
            templates = load_template_catalog(args.templates)
            template_classes = sorted({row.class_name for row in templates})
        except Exception as exc:  # detector diagnostics remain useful without classifier
            classification_error = f"{type(exc).__name__}: {exc}"
            print(f"symbol_ranking=DISABLED reason={classification_error}")

    output = args.output_folder.resolve() / _safe_stem(pdf)
    output.mkdir(parents=True, exist_ok=True)

    detector = GdtFrameDetector()
    rows: list[dict] = []

    # PyMuPDF is already used by detector/render_page_gray.  Opening only to get
    # the page count keeps this runner independent from the rest of CAD Review.
    import fitz

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        page_count = len(doc)

    for page_index in range(page_count):
        candidates = detector.detect_frames(pdf_bytes, page_index=page_index)
        score_by_id = {}
        if templates and candidates:
            page_gray, zoom = render_page_gray(pdf_bytes, page_index=page_index, dpi=args.symbol_dpi)
            for candidate, (score, _crop) in zip(candidates, score_candidates(candidates, page_gray, zoom, templates)):
                score_by_id[candidate.candidate_id] = score

        for candidate in candidates:
            score = score_by_id.get(candidate.candidate_id)
            rows.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "page": candidate.page,
                    "frame_bbox": _bbox(candidate.frame_bbox),
                    "symbol_bbox": _bbox(candidate.symbol_bbox),
                    "cell_bboxes": [_bbox(cell.bbox) for cell in candidate.cells],
                    "cell_texts": [list(cell.texts) for cell in candidate.cells],
                    "detection_status": "candidate_unvalidated",
                    "referenced_datums": [],
                    "unresolved_fields": [],
                    "symbol_scoring": (
                        {
                            **score.to_dict(),
                            "decision_policy": "ranking_only_no_global_threshold",
                            "catalog_complete": None,
                        }
                        if score is not None
                        else {
                            "best_class": None,
                            "decision_policy": "not_evaluated",
                            "reason": "detector_only" if args.detector_only else (classification_error or "no score"),
                        }
                    ),
                }
            )

    diagnostics = render_detection_diagnostics(
        pdf_bytes,
        output_dir=output,
        gdt_candidates=rows,
        dpi=args.visual_dpi,
        top_k=args.top_k,
    )

    payload = {
        "phase": "gdt_detection_debug",
        "validation_status": "DIAGNOSTIC_ONLY",
        "production_claim": False,
        "drawing": {"name": pdf.name, "source_path": str(pdf)},
        "page_count": page_count,
        "candidate_count": len(rows),
        "ground_truth_used": False,
        "candidate_semantics": "unvalidated detector proposals",
        "template_classes": template_classes,
        "symbol_ranking_enabled": bool(templates),
        "symbol_ranking_error": classification_error,
        "candidates": rows,
        "artifacts": diagnostics,
        "next_validation_step": (
            "Independently mark real FCFs in candidate_diagnostics.csv or a separate GT file, "
            "then compute detector recall/precision before evaluating symbol accuracy."
        ),
    }
    result_path = output / "debug_result.json"
    result_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("phase=gdt_detection_debug")
    print(f"pdf={pdf}")
    print(f"pages={page_count}")
    print(f"candidates={len(rows)}")
    print(f"symbol_ranking_enabled={bool(templates)}")
    print(f"output={output}")
    print(f"result={result_path}")
    print(f"candidate_csv={output / diagnostics['candidate_csv']}")
    for page in diagnostics["pages"]:
        print(f"page={page['page']} candidates_image={output / page['candidates_image']}")
        print(f"page={page['page']} symbol_contact_sheet={output / page['symbol_contact_sheet']}")


if __name__ == "__main__":
    main()
