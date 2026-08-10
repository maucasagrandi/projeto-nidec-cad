"""Run integrated CAD Review over every PDF in a folder.

Default layout:

    CADS/
      *.pdf

    RESULTS/
      summary_engineering.xlsx
      summary_technical.xlsx
      manifest.json
      <cad_stem>/
        result.json
        candidate_diagnostics.csv
        page_001_candidates.png
        page_001_symbol_contact_sheet.png
        page_001_annotated.png
        page_001_gdt.png
        page_001_datums.png
        crops/
          ...

Usage (PowerShell):
    python run_folder_cad_review.py --input-folder CADS --output-folder RESULTS

This is a validation batch while Phase 3 multi-CAD validation is still open.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prompts import classificacao_enriquecida_prompt
from src.cad_review.folder_pipeline import process_cad_pdf
from src.cad_review.result_exports import (
    engineering_row,
    technical_row,
    write_batch_workbooks,
    write_manifest,
)


def _safe_stem(path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("._")
    return stem or "cad"


def _pdfs(folder: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.pdf" if recursive else "*.pdf"
    return sorted(path for path in folder.glob(pattern) if path.is_file())


def _visual_paths(result: dict, dirname: str) -> list[str]:
    paths: list[str] = []
    artifacts = result.get("artifacts") or {}
    pages = (artifacts.get("visual_evidence") or {}).get("pages", [])
    for page in pages:
        for key in ("annotated_image", "gdt_image", "datums_image"):
            value = page.get(key)
            if value:
                paths.append(str(Path(dirname) / str(value)))

    diagnostic_pages = (artifacts.get("detection_diagnostics") or {}).get("pages", [])
    for page in diagnostic_pages:
        for key in ("candidates_image", "symbol_contact_sheet"):
            value = page.get(key)
            if value:
                paths.append(str(Path(dirname) / str(value)))
    return paths


def _diagnostic_paths(result: dict, dirname: str) -> dict:
    diagnostics = ((result.get("artifacts") or {}).get("detection_diagnostics") or {})
    candidate_csv = diagnostics.get("candidate_csv")
    return {
        "candidate_csv": str(Path(dirname) / str(candidate_csv)) if candidate_csv else None,
        "candidate_images": [
            str(Path(dirname) / str(page[key]))
            for page in diagnostics.get("pages", [])
            for key in ("candidates_image", "symbol_contact_sheet")
            if page.get(key)
        ],
    }


def _sync_phase4_templates() -> None:
    script = PROJECT_ROOT / "validation" / "gdt" / "scripts" / "sync_phase4_templates.py"
    if not script.exists():
        raise FileNotFoundError(f"template sync script not found: {script}")
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


def _error_result(pdf: Path, error: Exception) -> dict:
    return {
        "drawing": {"name": pdf.name, "source_path": str(pdf)},
        "review_context": {
            "compressor_series": "ALL",
            "compressor_series_source": "temporary_default_until_windchill",
        },
        "part_classification": {},
        "cited_standards": [],
        "applicable_standards": [],
        "standards_comparison": {},
        "gdt_frames": [],
        "datum_definitions": [],
        "findings": [],
        "summary": {"PASS": 0, "WARNING": 0, "NEEDS_CONTEXT": 0, "NOT_EVALUATED": 1},
        "validation_status": "BATCH_EXECUTION_ERROR",
        "production_claim": False,
        "error": {"type": type(error).__name__, "message": str(error)},
        "artifacts": {},
        "provenance": {},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Integrated CAD Review folder validation batch")
    parser.add_argument("--input-folder", type=Path, default=PROJECT_ROOT / "CADS")
    parser.add_argument("--output-folder", type=Path, default=PROJECT_ROOT / "RESULTS")
    parser.add_argument("--normas-xlsx", type=Path, default=PROJECT_ROOT / "Normas.xlsx")
    parser.add_argument("--templates", type=Path, default=PROJECT_ROOT / "assets" / "gdt" / "templates")
    parser.add_argument(
        "--iso1101-rules",
        type=Path,
        default=PROJECT_ROOT / "validation" / "gdt" / "configs" / "iso1101_2017_reference_rules.json",
    )
    parser.add_argument(
        "--reference-catalog",
        type=Path,
        default=PROJECT_ROOT / "validation" / "gdt" / "reference_catalog.json",
    )
    parser.add_argument("--visual-dpi", type=int, default=180)
    parser.add_argument("--symbol-dpi", type=int, default=300)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument(
        "--skip-template-sync",
        action="store_true",
        help="Do not synchronize versioned Phase 4 templates from cotas/ before the batch.",
    )
    parser.add_argument(
        "--allow-incomplete-symbol-catalog",
        action="store_true",
        help="Allow ranking against an incomplete symbol catalog. Diagnostic only; default is fail-closed.",
    )
    args = parser.parse_args()

    input_dir = args.input_folder.resolve()
    output_dir = args.output_folder.resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"input folder not found: {input_dir}")
    if not args.normas_xlsx.exists():
        raise FileNotFoundError(
            f"standards workbook not found: {args.normas_xlsx}. "
            "Pass --normas-xlsx with the customer applicability workbook."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_template_sync:
        print("template_sync=START")
        _sync_phase4_templates()
        print("template_sync=PASS")

    pdfs = _pdfs(input_dir, args.recursive)
    print("phase=folder_cad_review_batch")
    print(f"input={input_dir}")
    print(f"output={output_dir}")
    print("compressor_series=ALL source=temporary_default_until_windchill")
    print(f"pdf_count={len(pdfs)}")

    engineering_rows = []
    technical_rows = []
    manifest_entries = []
    used_dirs: set[str] = set()

    for index, pdf in enumerate(pdfs, start=1):
        dirname = _safe_stem(pdf)
        if dirname in used_dirs:
            dirname = f"{dirname}_{index:03d}"
        used_dirs.add(dirname)
        cad_dir = output_dir / dirname
        cad_dir.mkdir(parents=True, exist_ok=True)
        result_rel = str(Path(dirname) / "result.json")
        error_text = None
        print(f"[{index}/{len(pdfs)}] {pdf.name}")
        try:
            result = process_cad_pdf(
                pdf,
                output_dir=cad_dir,
                classification_prompt=classificacao_enriquecida_prompt,
                normas_path=args.normas_xlsx,
                template_root=args.templates,
                iso1101_rules_path=args.iso1101_rules,
                reference_catalog_path=args.reference_catalog,
                visual_dpi=args.visual_dpi,
                symbol_dpi=args.symbol_dpi,
                allow_incomplete_symbol_catalog=args.allow_incomplete_symbol_catalog,
            )
            print(
                f"  OK candidates={len(result.get('gdt_frames', []))} "
                f"warnings={(result.get('summary') or {}).get('WARNING', 0)}"
            )
        except Exception as exc:  # isolate one CAD from the rest of the batch
            error_text = f"{type(exc).__name__}: {exc}"
            result = _error_result(pdf, exc)
            result["error"]["traceback"] = traceback.format_exc(limit=8)
            (cad_dir / "result.json").write_text(
                json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"  ERROR {error_text}")

        visual_paths = _visual_paths(result, dirname)
        diagnostics = _diagnostic_paths(result, dirname)
        engineering_rows.append(engineering_row(result))
        tech_row = technical_row(result, result_json_path=result_rel, error=error_text)
        tech_row["Annotated_Images"] = "; ".join(visual_paths)
        technical_rows.append(tech_row)
        manifest_entries.append(
            {
                "cad": pdf.name,
                "status": "ERROR" if error_text else "OK",
                "result_json": result_rel,
                "annotated_images": visual_paths,
                "candidate_diagnostics_csv": diagnostics["candidate_csv"],
                "candidate_diagnostic_images": diagnostics["candidate_images"],
                "error": error_text,
            }
        )

    workbook_paths = write_batch_workbooks(
        output_dir,
        engineering_rows=engineering_rows,
        technical_rows=technical_rows,
    )
    manifest_path = write_manifest(output_dir, manifest_entries, workbook_paths=workbook_paths)

    ok = sum(row["status"] == "OK" for row in manifest_entries)
    errors = len(manifest_entries) - ok
    print("\noutputs:")
    print(f"  engineering={output_dir / workbook_paths['engineering']}")
    print(f"  technical={output_dir / workbook_paths['technical']}")
    print(f"  manifest={manifest_path}")
    print(f"completed={ok} errors={errors}")


if __name__ == "__main__":
    main()
