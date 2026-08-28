"""Standalone test for the CAD comparison step (first step of the pipeline).

Runs the pure OpenCV comparison (rasterize -> align via title block ->
pixel diff -> red highlights) on the two PDFs inside a given example folder
and exports a side-by-side PDF:

    [ Original ] | [ Revised (aligned) with red highlights on the changes ]

This deliberately uses ONLY the deterministic OpenCV step
(compare_cad_pages_opencv + save_side_by_side) so it needs no GCP/LLM
credentials.

Usage:
    python standalone_comparison_test.py <example_folder>

Defaults to the "22" example folder under cads_docs_examples.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the pipeline `src` package importable when run from anywhere.
PIPELINE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE_ROOT))

from src.utils.opencv_cad_compare import (  # noqa: E402
    CompareConfig,
    compare_cad_pages_opencv,
    save_side_by_side,
)

# Repo root = .../nidec-cad-review  (pipeline is at CloudRun_functions/pipeline)
REPO_ROOT = PIPELINE_ROOT.parent.parent
EXAMPLES_ROOT = REPO_ROOT / "cads_docs_examples"


def _find_pdf_pair(folder: Path) -> tuple[Path, Path]:
    """Return (original, revised) PDF paths from a folder holding source drawings.

    Only source drawing PDFs (those with 'draw' in the filename) are considered,
    so generated outputs (comparison/report PDFs) in the same folder are ignored.
    The revision letter/number in the filename determines ordering: the file that
    sorts first is treated as the original, the later one as the revised.
    """
    pdfs = sorted(
        p for p in folder.iterdir()
        if p.suffix.lower() == ".pdf" and "draw" in p.name.lower()
    )
    if len(pdfs) != 2:
        raise ValueError(
            f"Expected exactly 2 source drawing PDFs in {folder}, found "
            f"{len(pdfs)}: {[p.name for p in pdfs]}"
        )
    return pdfs[0], pdfs[1]


def main() -> None:
    example = sys.argv[1] if len(sys.argv) > 1 else "22"
    folder = EXAMPLES_ROOT / example

    if not folder.is_dir():
        raise SystemExit(f"Example folder not found: {folder}")

    original_pdf, revised_pdf = _find_pdf_pair(folder)
    print(f"Original (reference): {original_pdf.name}")
    print(f"Revised:              {revised_pdf.name}")

    config = CompareConfig(dpi=200)

    result = compare_cad_pages_opencv(
        original_pdf.read_bytes(),
        revised_pdf.read_bytes(),
        page_index=0,
        config=config,
    )

    print(f"Alignment method: "
          f"{'homography' if result.homography_matrix is not None else 'resize'} "
          f"(score={result.alignment_score:.3f})")
    print(f"Detected difference regions: {result.num_differences}")

    output_path = folder / f"{example}_comparison_side_by_side.pdf"
    save_side_by_side(result, output_path)
    print(f"\nExported side-by-side comparison PDF to:\n  {output_path}")


if __name__ == "__main__":
    main()
