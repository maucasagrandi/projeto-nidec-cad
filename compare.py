"""Compare two CAD drawing PDFs and produce a verified change report.

Usage:
    uv run python compare.py -i original.pdf -r revised.pdf -o output/
    uv run python compare.py -i original.pdf -r revised.pdf -o output/ --page 0
    uv run python compare.py -i original.pdf -r revised.pdf -o output/ --all-pages
"""

import argparse
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="Compare two CAD drawing PDFs and report verified changes."
    )
    parser.add_argument(
        "-i", "--input", required=True, type=Path,
        help="Path to the original (reference) PDF",
    )
    parser.add_argument(
        "-r", "--revised", required=True, type=Path,
        help="Path to the revised PDF",
    )
    parser.add_argument(
        "-o", "--output", required=True, type=Path,
        help="Output directory for results",
    )
    parser.add_argument(
        "--page", type=int, default=0,
        help="Page index to compare (0-based, default: 0)",
    )
    parser.add_argument(
        "--all-pages", action="store_true",
        help="Compare all common pages (overrides --page)",
    )
    parser.add_argument(
        "--model", type=str, default="gemini-2.5-flash",
        help="Gemini model to use (default: gemini-2.5-flash)",
    )
    parser.add_argument(
        "--merge-distance", type=int, default=50,
        help="Max pixel gap to merge nearby detections (default: 50)",
    )
    parser.add_argument(
        "--threshold", type=int, default=40,
        help="Pixel diff threshold 0-255 (default: 40)",
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: original PDF not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    if not args.revised.exists():
        print(f"Error: revised PDF not found: {args.revised}", file=sys.stderr)
        sys.exit(1)

    from src.modeling.llm_verify_changes import (
        run_verification_pipeline,
        run_verification_pipeline_all_pages,
        save_verification_result,
    )
    from src.utils.opencv_cad_compare import CompareConfig

    config = CompareConfig(
        diff_threshold=args.threshold,
        merge_distance=args.merge_distance,
    )

    pdf1_bytes = args.input.read_bytes()
    pdf2_bytes = args.revised.read_bytes()

    print(f"Original: {args.input}")
    print(f"Revised:  {args.revised}")
    print(f"Output:   {args.output}")
    print()

    start = time.time()

    if args.all_pages:
        results = run_verification_pipeline_all_pages(
            pdf1_bytes, pdf2_bytes, model=args.model, opencv_config=config
        )
        for result in results:
            page_dir = args.output / f"page_{result.page_index + 1:02d}"
            save_verification_result(result, page_dir)
            print(result.report_text())
            print()
    else:
        result = run_verification_pipeline(
            pdf1_bytes, pdf2_bytes,
            page_index=args.page,
            model=args.model,
            opencv_config=config,
        )
        save_verification_result(result, args.output)
        print(result.report_text())

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s. Results saved to: {args.output}")


if __name__ == "__main__":
    main()
