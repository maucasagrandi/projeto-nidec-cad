"""Run the complete original-vs-revised CAD Review workflow.

Usage:
    python run_review.py original.pdf revised.pdf -o REVIEW_RESULTS
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent
_PIPELINE_ROOT = _PROJECT_ROOT / "CloudRun_functions" / "pipeline"
if not (_PIPELINE_ROOT / "src" / "cad_review").is_dir():
    raise RuntimeError(f"Pipeline package not found: {_PIPELINE_ROOT}")
sys.path.insert(0, str(_PIPELINE_ROOT))

from src.cad_review.integrated_review import (
    _format_elapsed,
    run_integrated_review,
    save_integrated_review,
)
from src.utils.opencv_cad_compare import CompareConfig

logger = logging.getLogger(__name__)


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Integrated Part Classification, GD&T and CAD comparison")
    parser.add_argument("original", type=Path, help="Original/reference drawing PDF")
    parser.add_argument("revised", type=Path, help="Revised drawing PDF")
    parser.add_argument("-o", "--output", type=Path, default=Path("REVIEW_RESULTS"))
    parser.add_argument("--classification-model", default=None)
    parser.add_argument("--comparison-model", default="gemini-2.5-flash")
    parser.add_argument("--gdt-dpi", type=int, default=150)
    parser.add_argument("--gdt-threshold", type=float, default=0.74)
    parser.add_argument(
        "--gdt-workers",
        type=int,
        default=1,
        help="Parallel GD&T template matches (default: 1, memory-safe on Windows)",
    )
    parser.add_argument("--opencv-threshold", type=int, default=40)
    parser.add_argument(
        "--opencv-dpi",
        type=int,
        default=150,
        help="OpenCV rasterization DPI (default: 150, memory-safe on Windows)",
    )
    parser.add_argument("--merge-distance", type=int, default=50)
    args = parser.parse_args()

    for pdf in (args.original, args.revised):
        if not pdf.is_file():
            parser.error(f"PDF not found: {pdf}")

    command_started = perf_counter()
    logger.info("TEMPO | Execução completa iniciada | acumulado=00:00:00.0")
    try:
        result = run_integrated_review(
            args.original.read_bytes(),
            args.revised.read_bytes(),
            original_name=args.original.name,
            revised_name=args.revised.name,
            classification_model=args.classification_model,
            comparison_model=args.comparison_model,
            gdt_dpi=args.gdt_dpi,
            gdt_threshold=args.gdt_threshold,
            gdt_workers=args.gdt_workers,
            template_root=_PIPELINE_ROOT / "assets" / "gdt" / "templates",
            opencv_config=CompareConfig(
                dpi=args.opencv_dpi,
                diff_threshold=args.opencv_threshold,
                merge_distance=args.merge_distance,
            ),
        )
        paths = save_integrated_review(result, args.output)
        print(f"JSON:   {paths['json']}")
        print(f"Report: {paths['report']}")
    finally:
        logger.info(
            "TEMPO | Execução completa encerrada | total=%s",
            _format_elapsed(perf_counter() - command_started),
        )


if __name__ == "__main__":
    main()
