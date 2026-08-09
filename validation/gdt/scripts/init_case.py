"""Cria configuração mínima de um caso GD&T sem inventar ground truth.

Exemplo:
    python validation/gdt/scripts/init_case.py \
      --case-id case_42_rev_b \
      --pdf "CAD_Review_Test_Battery_V1/2. Comparison Analysis/42/example_draw_2.pdf"

O arquivo criado contém apenas localização do PDF/página e metadados de
validação. Quantidade/classe de frames só entram depois da anotação humana.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--page-index", type=int, default=0)
    parser.add_argument("--notes", default="Phase 3 generalization case.")
    parser.add_argument("--output")
    args = parser.parse_args()

    pdf_path = _project_path(args.pdf)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF não encontrado: {pdf_path}")

    try:
        relative_pdf = pdf_path.relative_to(PROJECT_ROOT)
        pdf_value = str(relative_pdf).replace("\\", "/")
    except ValueError:
        pdf_value = str(pdf_path)

    payload = {
        "schema_version": 1,
        "case_id": args.case_id,
        "pdf": pdf_value,
        "page_index": args.page_index,
        "phase": "generalization",
        "expected": None,
        "notes": args.notes,
    }

    output = _project_path(
        args.output or f"validation/gdt/cases/{args.case_id}.json"
    )
    if output.exists():
        raise FileExistsError(
            f"Caso já existe: {output}. Use outro --case-id ou remova conscientemente o arquivo."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"case={args.case_id}")
    print(f"pdf={pdf_value}")
    print(f"page_index={args.page_index}")
    print("expected=None")
    print(f"output={output}")


if __name__ == "__main__":
    main()
