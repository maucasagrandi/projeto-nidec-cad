"""Run isolated GD&T Candidate Detector diagnostics over every PDF in a folder.

This is a convenience wrapper around ``run_gdt_detection_debug.py``. It keeps
one failure from stopping the remaining CADs and writes a small manifest.

Default detector is V2 shadow/diagnostic mode.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GD&T detector diagnostics for a folder")
    parser.add_argument("--input-folder", type=Path, default=PROJECT_ROOT / "CADS")
    parser.add_argument("--output-folder", type=Path, default=PROJECT_ROOT / "DEBUG_RESULTS_V2")
    parser.add_argument("--detector-version", choices=("v1", "v2"), default="v2")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--skip-template-sync", action="store_true")
    parser.add_argument("--detector-only", action="store_true")
    args = parser.parse_args()

    input_dir = args.input_folder.resolve()
    output_dir = args.output_folder.resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"input folder not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    pattern = "**/*.pdf" if args.recursive else "*.pdf"
    pdfs = sorted(p for p in input_dir.glob(pattern) if p.is_file())
    manifest = []

    print(f"detector_version={args.detector_version}")
    print(f"pdf_count={len(pdfs)}")
    for index, pdf in enumerate(pdfs, start=1):
        print(f"[{index}/{len(pdfs)}] {pdf.name}")
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "run_gdt_detection_debug.py"),
            "--pdf",
            str(pdf),
            "--output-folder",
            str(output_dir),
            "--detector-version",
            args.detector_version,
        ]
        # The single-file runner synchronizes templates by default. For a folder,
        # only the first CAD needs to do it; following CADs reuse the synchronized
        # catalog to avoid repeated work.
        if args.skip_template_sync or index > 1:
            cmd.append("--skip-template-sync")
        if args.detector_only:
            cmd.append("--detector-only")

        completed = subprocess.run(cmd, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
        if completed.stdout.strip():
            print(completed.stdout.strip())
        if completed.returncode == 0:
            status = "OK"
            error = None
        else:
            status = "ERROR"
            error = completed.stderr.strip() or completed.stdout.strip()
            print(error)
        manifest.append({"cad": pdf.name, "status": status, "error": error})

    manifest_path = output_dir / "debug_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    ok = sum(row["status"] == "OK" for row in manifest)
    print(f"completed={ok} errors={len(manifest)-ok}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
