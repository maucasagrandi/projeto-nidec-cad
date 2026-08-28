"""Run the integrated CAD review pipeline for tests 41-50 in batch.

Outputs per test go to:
    scripts/results/test_<N>/integrated_review.json
    scripts/results/test_<N>/integrated_review_report.pdf

Usage:
    python scripts/run_batch.py
    python scripts/run_batch.py --tests 41 42 43
    python scripts/run_batch.py --tests 41 --opencv-dpi 150 --gdt-workers 1
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Map: test number -> (original_draw, revised_draw)
# Filenames exactly as found in CAD_Review_Test_Battery_V1/2. Comparison Analysis/<N>/
# ---------------------------------------------------------------------------

TESTS: dict[int, tuple[str, str]] = {
    41: ("13358002_REV_7_draw_1.pdf",    "13358002_REV_8_draw_2.pdf"),
    42: ("13751188_REV_0_draw_1.pdf",    "13751188_REV_A_draw_2.pdf"),
    43: ("13851052_rev_1_draw_1.pdf",    "13851054_rev0_draw_2.pdf"),
    44: ("13958000_REV_2_draw_1.pdf",    "13958000_REV_3_draw_2.pdf"),
    45: ("113891052_RevC_draw_1.pdf",    "113891052_Rev1_draw_2.pdf"),
    46: ("13751188_REV_A_draw_1.pdf",    "13751188_REV_A1_draw_2.pdf"),
    47: ("113340048_REV5_draw_1.pdf",    "113340048_REV5A_draw_2.pdf"),
    48: ("113390048_Rev7_draw_1.pdf",    "113390048_REV8_draw_2.pdf"),
    49: ("14040156_REV_B_draw_1.pdf",    "14040156_REV_C_draw_2.pdf"),
    50: ("14040157_REV_B_draw_1.pdf",    "14040157_REV_C_draw_2.pdf"),
}

BASE_DIR    = Path("CAD_Review_Test_Battery_V1/2. Comparison Analysis")
RESULTS_DIR = Path("scripts/results")


def run_test(
    test_number: int,
    opencv_dpi: int,
    gdt_workers: int,
    extra_args: list[str],
) -> bool:
    """Run run_review.py for a single test. Returns True on success."""
    test_dir    = BASE_DIR / str(test_number)
    output_dir  = RESULTS_DIR / f"test_{test_number}"

    if test_number not in TESTS:
        logger.error("Test %d not found in TESTS map.", test_number)
        return False

    original_name, revised_name = TESTS[test_number]
    original_path = test_dir / original_name
    revised_path  = test_dir / revised_name

    if not original_path.exists():
        logger.error("[test %d] Original PDF not found: %s", test_number, original_path)
        return False
    if not revised_path.exists():
        logger.error("[test %d] Revised PDF not found: %s", test_number, revised_path)
        return False

    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "run_review.py",
        str(original_path),
        str(revised_path),
        "-o", str(output_dir),
        "--opencv-dpi", str(opencv_dpi),
        "--gdt-workers", str(gdt_workers),
        *extra_args,
    ]

    logger.info("[test %d] Running: %s", test_number, " ".join(cmd))
    t0 = time.time()

    result = subprocess.run(cmd, capture_output=False, text=True)

    elapsed = time.time() - t0
    if result.returncode == 0:
        logger.info("[test %d] Done in %.1fs -> %s", test_number, elapsed, output_dir)
        return True
    else:
        logger.error("[test %d] FAILED (exit code %d) after %.1fs", test_number, result.returncode, elapsed)
        return False


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Batch CAD review — tests 41-50")
    parser.add_argument(
        "--tests",
        type=int,
        nargs="+",
        default=list(TESTS.keys()),
        help="Test numbers to run (default: all 41-50)",
    )
    parser.add_argument("--opencv-dpi", type=int, default=150)
    parser.add_argument("--gdt-workers", type=int, default=1)
    parser.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra arguments forwarded to run_review.py",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    total   = len(args.tests)
    passed  = 0
    failed  = []
    batch_start = time.time()

    logger.info("Starting batch: %d tests -> %s", total, RESULTS_DIR)

    for n in args.tests:
        ok = run_test(n, args.opencv_dpi, args.gdt_workers, args.extra or [])
        if ok:
            passed += 1
        else:
            failed.append(n)

    elapsed = time.time() - batch_start
    logger.info("=" * 60)
    logger.info("Batch complete in %.0fs: %d/%d passed", elapsed, passed, total)
    if failed:
        logger.warning("Failed tests: %s", failed)
    else:
        logger.info("All tests passed.")


if __name__ == "__main__":
    main()
