"""Diagnostic harness for the OpenCV CAD comparison step.

Runs compare_cad_pages_opencv over a set of example folders at production
settings and prints, per folder: alignment method/score, number of accepted
difference boxes, and whether any box is "giant" (covers a large fraction of
the page — the failure signature we are fixing).

Usage:
    python diag_compare.py            # sample set
    python diag_compare.py 22 30 41   # specific folders
"""

from __future__ import annotations

import sys
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE_ROOT))

from src.utils.opencv_cad_compare import (  # noqa: E402
    CompareConfig,
    compare_cad_pages_opencv,
)

REPO_ROOT = PIPELINE_ROOT.parent.parent
EXAMPLES_ROOT = REPO_ROOT / "cads_docs_examples"

GIANT_AREA_FRAC = 0.40  # box area over this fraction of page => "giant"


def _pdf_pair(folder: Path) -> tuple[Path, Path] | None:
    pdfs = sorted(
        p for p in folder.iterdir()
        if p.suffix.lower() == ".pdf" and "draw" in p.name.lower()
    )
    if len(pdfs) != 2:
        # fall back to any two PDFs
        pdfs = sorted(p for p in folder.iterdir() if p.suffix.lower() == ".pdf")
    if len(pdfs) < 2:
        return None
    return pdfs[0], pdfs[1]


def run_folder(example: str, config: CompareConfig) -> dict:
    folder = EXAMPLES_ROOT / example
    pair = _pdf_pair(folder)
    if pair is None:
        return {"folder": example, "error": "no PDF pair"}
    orig, rev = pair
    r = compare_cad_pages_opencv(
        orig.read_bytes(), rev.read_bytes(), 0, config, include_visualization=False
    )
    page_area = r.image1.shape[0] * r.image1.shape[1]
    pw, ph = r.image1.shape[1], r.image1.shape[0]
    giant = 0
    max_frac = 0.0
    for (x, y, w, h) in r.diff_bboxes:
        frac = (w * h) / page_area
        max_frac = max(max_frac, frac)
        if frac > GIANT_AREA_FRAC or w > 0.9 * pw or h > 0.9 * ph:
            giant += 1
    return {
        "folder": example,
        "align": "homography" if r.homography_matrix is not None else "resize",
        "score": round(r.alignment_score, 3),
        "boxes": r.num_differences,
        "giant": giant,
        "max_box_frac": round(max_frac, 3),
    }


def main() -> None:
    args = sys.argv[1:]
    if args:
        folders = args
    else:
        folders = [str(n) for n in range(15, 51)]

    config = CompareConfig(dpi=150, diff_threshold=40, merge_distance=50)

    print(f"{'folder':>7} {'align':>11} {'score':>6} {'boxes':>6} "
          f"{'giant':>6} {'maxfrac':>8}")
    print("-" * 52)
    n_giant_folders = 0
    for f in folders:
        try:
            res = run_folder(f, config)
        except Exception as e:  # noqa: BLE001
            print(f"{f:>7}  ERROR: {e}")
            continue
        if "error" in res:
            print(f"{f:>7}  {res['error']}")
            continue
        if res["giant"] > 0:
            n_giant_folders += 1
        print(f"{res['folder']:>7} {res['align']:>11} {res['score']:>6} "
              f"{res['boxes']:>6} {res['giant']:>6} {res['max_box_frac']:>8}")
    print("-" * 52)
    print(f"folders with a giant box: {n_giant_folders}")


if __name__ == "__main__":
    main()
