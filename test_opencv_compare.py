"""Standalone test/demo script for the OpenCV-based CAD comparison pipeline.

Runs the pipeline against test cases from the CAD_Review_Test_Battery_V1 folder
and saves results to a timestamped output directory.

Usage:
    uv run python test_opencv_compare.py

The script will:
1. Iterate over test cases in the battery (both single and comparison analysis)
2. Run the OpenCV pipeline on each pair
3. Save highlighted diff images and a summary report
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.opencv_cad_compare import (
    CompareConfig,
    CompareResult,
    compare_cad_pages_opencv,
    save_result_image,
    save_side_by_side,
)


# ==============================================================================
# Test case discovery
# ==============================================================================

BATTERY_DIR = PROJECT_ROOT / "CAD_Review_Test_Battery_V1"
SINGLE_DIR = BATTERY_DIR / "1. Single Analysis"
COMPARISON_DIR = BATTERY_DIR / "2. Comparison Analysis"

OUTPUT_DIR = PROJECT_ROOT / "opencv_compare_results"


def discover_single_analysis_cases() -> list[tuple[str, Path, Path]]:
    """Find test cases in the Single Analysis folder.

    Each subfolder has two PDFs: one '*_original.pdf' and one '*_review.pdf'.
    Returns list of (case_name, original_pdf_path, review_pdf_path).
    """
    cases = []
    if not SINGLE_DIR.exists():
        return cases

    for folder in sorted(SINGLE_DIR.iterdir()):
        if not folder.is_dir():
            continue

        pdfs = sorted(folder.glob("*.pdf"))
        if len(pdfs) < 2:
            continue

        # Try to identify original vs review by filename
        original = None
        review = None
        for pdf in pdfs:
            name_lower = pdf.name.lower()
            if "original" in name_lower or "sap" in name_lower:
                original = pdf
            elif "review" in name_lower or "check" in name_lower or "cad" in name_lower:
                review = pdf

        # Fallback: first is original, second is review (alphabetical)
        if original is None or review is None:
            original = pdfs[0]
            review = pdfs[1]

        cases.append((folder.name, original, review))

    return cases


def discover_comparison_analysis_cases() -> list[tuple[str, Path, Path]]:
    """Find test cases in the Comparison Analysis folder.

    Each subfolder has draw_1 (original) and draw_2 (revised), plus a *_vs.pdf.
    Returns list of (case_name, draw1_path, draw2_path).
    """
    cases = []
    if not COMPARISON_DIR.exists():
        return cases

    for folder in sorted(COMPARISON_DIR.iterdir()):
        if not folder.is_dir():
            continue

        pdfs = sorted(folder.glob("*.pdf"))
        if len(pdfs) < 2:
            continue

        draw1 = None
        draw2 = None
        for pdf in pdfs:
            name_lower = pdf.name.lower()
            if "draw_1" in name_lower or "rev_2" in name_lower.replace("rev_2vs", ""):
                draw1 = pdf
            elif "draw_2" in name_lower:
                draw2 = pdf

        # Fallback: skip the *_vs.pdf, use the other two
        if draw1 is None or draw2 is None:
            non_vs = [p for p in pdfs if "_vs" not in p.name.lower()]
            if len(non_vs) >= 2:
                draw1, draw2 = non_vs[0], non_vs[1]
            elif len(pdfs) >= 2:
                draw1, draw2 = pdfs[0], pdfs[1]
            else:
                continue

        cases.append((folder.name, draw1, draw2))

    return cases


# ==============================================================================
# Run pipeline
# ==============================================================================

def run_test_case(
    case_name: str,
    pdf1_path: Path,
    pdf2_path: Path,
    output_dir: Path,
    config: CompareConfig,
) -> dict:
    """Run the comparison pipeline on a single test case.

    Returns a dict with timing and result metadata.
    """
    print(f"\n{'='*60}")
    print(f"  Case: {case_name}")
    print(f"  Original: {pdf1_path.name}")
    print(f"  Revised:  {pdf2_path.name}")
    print(f"{'='*60}")

    case_output = output_dir / case_name
    case_output.mkdir(parents=True, exist_ok=True)

    pdf1_bytes = pdf1_path.read_bytes()
    pdf2_bytes = pdf2_path.read_bytes()

    start = time.time()
    try:
        result = compare_cad_pages_opencv(pdf1_bytes, pdf2_bytes, page_index=0, config=config)
    except Exception as e:
        elapsed = time.time() - start
        print(f"  ERROR: {e}")
        return {
            "case": case_name,
            "status": "error",
            "error": str(e),
            "elapsed_s": elapsed,
        }

    elapsed = time.time() - start

    # Print summary
    print(f"  Title block 1: {'FOUND' if result.title_block_bbox1 else 'NOT FOUND'}")
    print(f"  Title block 2: {'FOUND' if result.title_block_bbox2 else 'NOT FOUND'}")
    print(f"  Homography:    {'ESTIMATED' if result.homography_matrix is not None else 'FALLBACK (resize)'}")
    print(f"  Alignment:     {result.alignment_score:.2%} inlier ratio")
    print(f"  Differences:   {result.num_differences} regions detected")
    print(f"  Time:          {elapsed:.2f}s")

    # Save outputs
    save_result_image(result, case_output / "diff_highlighted.png")
    save_side_by_side(result, case_output / "side_by_side.png")

    # Also save just the aligned revised image for inspection
    import cv2
    cv2.imwrite(str(case_output / "img2_aligned.png"), result.image2_aligned)

    print(f"  Output:        {case_output}")

    return {
        "case": case_name,
        "status": "ok",
        "title_block_1": result.title_block_bbox1 is not None,
        "title_block_2": result.title_block_bbox2 is not None,
        "homography": result.homography_matrix is not None,
        "alignment_score": result.alignment_score,
        "num_differences": result.num_differences,
        "elapsed_s": elapsed,
    }


def main():
    print("=" * 60)
    print("  OpenCV CAD Comparison — Test Battery Runner")
    print("=" * 60)

    config = CompareConfig(
        dpi=300,
        diff_threshold=40,
        min_contour_area=200,
        min_divergence_pct=8.0,
        highlight_alpha=0.35,
        box_padding=8,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Discover test cases
    single_cases = discover_single_analysis_cases()
    comparison_cases = discover_comparison_analysis_cases()

    all_cases = [
        ("single", name, p1, p2) for name, p1, p2 in single_cases
    ] + [
        ("comparison", name, p1, p2) for name, p1, p2 in comparison_cases
    ]

    if not all_cases:
        print("\n  No test cases found. Check that CAD_Review_Test_Battery_V1/ exists.")
        sys.exit(1)

    print(f"\n  Found {len(single_cases)} single analysis cases")
    print(f"  Found {len(comparison_cases)} comparison analysis cases")
    print(f"  Total: {len(all_cases)} test cases")
    print(f"  Output: {OUTPUT_DIR}")

    # Run each case
    results = []
    for category, case_name, pdf1, pdf2 in all_cases:
        sub_dir = OUTPUT_DIR / category
        sub_dir.mkdir(parents=True, exist_ok=True)
        r = run_test_case(case_name, pdf1, pdf2, sub_dir, config)
        results.append(r)

    # Print summary table
    print("\n\n")
    print("=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    print(f"  {'Case':<20} {'Status':<8} {'TB1':<5} {'TB2':<5} {'Homog':<7} {'Score':<8} {'Diffs':<7} {'Time':<7}")
    print(f"  {'-'*20} {'-'*8} {'-'*5} {'-'*5} {'-'*7} {'-'*8} {'-'*7} {'-'*7}")

    for r in results:
        if r["status"] == "error":
            print(f"  {r['case']:<20} {'ERROR':<8} {'-':<5} {'-':<5} {'-':<7} {'-':<8} {'-':<7} {r['elapsed_s']:<7.2f}")
        else:
            tb1 = "YES" if r["title_block_1"] else "NO"
            tb2 = "YES" if r["title_block_2"] else "NO"
            hom = "YES" if r["homography"] else "NO"
            print(
                f"  {r['case']:<20} {'OK':<8} {tb1:<5} {tb2:<5} {hom:<7} "
                f"{r['alignment_score']:<8.2%} {r['num_differences']:<7} {r['elapsed_s']:<7.2f}"
            )

    # Overall stats
    ok_results = [r for r in results if r["status"] == "ok"]
    error_results = [r for r in results if r["status"] == "error"]
    tb_found = sum(1 for r in ok_results if r["title_block_1"] and r["title_block_2"])
    hom_found = sum(1 for r in ok_results if r["homography"])
    total_time = sum(r["elapsed_s"] for r in results)

    print(f"\n  Total cases:    {len(results)}")
    print(f"  Successful:     {len(ok_results)}")
    print(f"  Errors:         {len(error_results)}")
    print(f"  Title blocks:   {tb_found}/{len(ok_results)} (both detected)")
    print(f"  Homographies:   {hom_found}/{len(ok_results)} (estimated)")
    print(f"  Total time:     {total_time:.1f}s")
    print(f"  Avg per case:   {total_time/len(results):.1f}s")
    print(f"\n  Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
