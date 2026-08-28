"""Batch-run the integrated CAD review pipeline over local example folders.

Runs `run_integrated_review` + `save_integrated_review` for every folder in
cads_docs_examples/, mirroring the orchestrator's PDF-resolution logic:
  - 2+ PDFs -> comparison mode (lowest name = original, highest = revised)
  - 1 PDF   -> single mode (revised only)

Processing artifacts (review reports, side-by-side outputs) are ignored so
re-runs are idempotent.

Usage (from CloudRun_functions/pipeline/, with pipeline deps + ADC):
    python run_all_examples.py
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

# Number of GD&T template-matching workers. Local batch runs can use more
# (the machine has plenty of RAM); Cloud Run stays at 1 to avoid OOM.
GDT_WORKERS = int(os.getenv("GDT_WORKERS", "1"))

# Resolve repo root and examples dir relative to this file
PIPELINE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PIPELINE_DIR.parent.parent
EXAMPLES_DIR = REPO_ROOT / "cads_docs_examples"
OUTPUT_ROOT = REPO_ROOT / "REVIEW_RESULTS_TEST"

# Ensure the pipeline package (src/, prompts.py, assets/) is importable
sys.path.insert(0, str(PIPELINE_DIR))

from src.cad_review.integrated_review import run_integrated_review, save_integrated_review
from src.utils.opencv_cad_compare import CompareConfig

# Filename suffixes that indicate a processing artifact, not a source drawing
ARTIFACT_SUFFIXES = (
    "_review_report.pdf",
    "_side_by_side.pdf",
    "_vs.pdf",
    "_diff.pdf",
)


def is_source_pdf(path: Path) -> bool:
    name = path.name.lower()
    if not name.endswith(".pdf"):
        return False
    return not any(name.endswith(suffix) for suffix in ARTIFACT_SUFFIXES)


def resolve_pdfs(folder: Path) -> tuple[Path | None, Path]:
    """Return (original, revised). original is None in single mode."""
    pdfs = sorted(p for p in folder.iterdir() if is_source_pdf(p))
    if not pdfs:
        raise ValueError(f"No source PDFs in {folder}")
    if len(pdfs) == 1:
        return None, pdfs[0]
    return pdfs[0], pdfs[-1]


def run_one(folder: Path) -> dict:
    original_path, revised_path = resolve_pdfs(folder)
    mode = "comparison" if original_path else "single"

    revised_bytes = revised_path.read_bytes()
    original_bytes = original_path.read_bytes() if original_path else None

    result = run_integrated_review(
        revised_bytes,
        original_pdf=original_bytes,
        original_name=original_path.name if original_path else "",
        revised_name=revised_path.name,
        comparison_model="gemini-2.5-flash",
        gdt_dpi=150,
        gdt_threshold=0.74,
        gdt_workers=GDT_WORKERS,
        opencv_config=CompareConfig(dpi=150, diff_threshold=40, merge_distance=50),
    )

    output_dir = OUTPUT_ROOT / folder.name
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = save_integrated_review(result, output_dir)
    report = paths.get("report") or (output_dir / "integrated_review_report.pdf")

    return {
        "folder": folder.name,
        "mode": mode,
        "original": original_path.name if original_path else None,
        "revised": revised_path.name,
        "report": str(report),
        "report_exists": Path(report).exists(),
    }


def main() -> int:
    folders = sorted(
        (p for p in EXAMPLES_DIR.iterdir() if p.is_dir()),
        key=lambda p: int(p.name) if p.name.isdigit() else p.name,
    )

    results: list[dict] = []
    failures: list[dict] = []

    print(f"Found {len(folders)} example folders in {EXAMPLES_DIR}\n")

    for folder in folders:
        started = time.time()
        print(f"[{folder.name}] starting...", flush=True)
        try:
            info = run_one(folder)
            elapsed = time.time() - started
            info["seconds"] = round(elapsed, 1)
            results.append(info)
            print(
                f"[{folder.name}] OK ({info['mode']}, {elapsed:.1f}s) "
                f"-> {info['report']}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            elapsed = time.time() - started
            failures.append({"folder": folder.name, "error": str(exc)})
            print(f"[{folder.name}] FAILED ({elapsed:.1f}s): {exc}", flush=True)
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"SUMMARY: {len(results)} succeeded, {len(failures)} failed")
    print("=" * 60)
    for info in results:
        flag = "OK " if info["report_exists"] else "NO-REPORT"
        print(f"  {flag} {info['folder']:>4} | {info['mode']:<10} | {info['seconds']}s")
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  {f['folder']}: {f['error']}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
