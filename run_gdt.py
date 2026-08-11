"""GD&T analysis — single command entry point.

Takes a PDF drawing and produces:
- An annotated image highlighting detected GD&T constraints, FCF frames,
  datum reference cells, and datum definition boxes.
- A structured JSON report with all findings.

Usage:
    python run_gdt.py CADS/113340048_REV5_draw_1.pdf
    python run_gdt.py CADS/113340048_REV5_draw_1.pdf -o results/
    python run_gdt.py CADS/113340048_REV5_draw_1.pdf --dpi 200 --threshold 0.72
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gdt.gdt_report import (
    analyze_page,
    render_annotated_page,
    save_report,
    save_visualization,
)


def _safe_stem(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("._") or "cad"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GD&T constraint detection, FCF extraction, and datum finding.",
    )
    parser.add_argument("pdf", type=Path, help="Input PDF drawing.")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output directory. Default: GDT_RESULTS/<pdf_stem>/")
    parser.add_argument("--dpi", type=int, default=150,
                        help="Detection DPI (default 150). Extraction always uses 300.")
    parser.add_argument("--threshold", type=float, default=0.74,
                        help="Template matching threshold (default 0.74).")
    parser.add_argument("--workers", type=int, default=8,
                        help="Thread pool size for template matching (default 8).")
    parser.add_argument("--page", type=int, default=0,
                        help="Page index, 0-based (default 0).")
    args = parser.parse_args()

    pdf = args.pdf.resolve()
    if not pdf.exists():
        print(f"Error: PDF not found: {pdf}", file=sys.stderr)
        raise SystemExit(1)

    output = args.output or (PROJECT_ROOT / "GDT_RESULTS" / _safe_stem(pdf))
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    pdf_bytes = pdf.read_bytes()

    t0 = time.time()
    report, detections, frames, extractions, datum_defs = analyze_page(
        pdf_bytes,
        page_index=args.page,
        template_root=str(PROJECT_ROOT / "assets" / "gdt" / "templates"),
        dpi=args.dpi,
        score_threshold=args.threshold,
        pdf_name=pdf.name,
        max_workers=args.workers,
    )
    t1 = time.time()

    # Visualization — render at a reasonable display DPI (reuse analysis render
    # if DPI matches, otherwise re-render at 150 for a smaller image file)
    vis_dpi = min(args.dpi, 200)
    vis = render_annotated_page(
        pdf_bytes, detections, frames, extractions, datum_defs,
        page_index=args.page, dpi=vis_dpi,
    )
    t2 = time.time()

    report_path = save_report(report, output)
    vis_path = save_visualization(vis, output, page_index=args.page)
    t3 = time.time()

    # Print summary
    s = report.summary
    print(f"pdf:              {pdf.name}")
    print(f"detections:       {s['total_detections']}")
    print(f"fcf_frames:       {s['fcf_frames_expanded']}")
    print(f"with_datums:      {s['constraints_with_datums']}")
    print(f"datum_refs:       {s['total_datum_refs']}")
    print(f"datum_defs:       {s['datum_definitions_found']}")
    print(f"types:            {s['constraint_types']}")
    print(f"time:             {t3-t0:.1f}s (analysis={t1-t0:.1f}s viz={t2-t1:.1f}s)")
    print(f"report:           {report_path}")
    print(f"image:            {vis_path}")


if __name__ == "__main__":
    main()
