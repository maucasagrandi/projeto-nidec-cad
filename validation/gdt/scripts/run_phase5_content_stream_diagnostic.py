"""Diagnóstico da Fase 5: inspeciona content streams e fontes do PDF sem OCR/LLM.

Objetivo:
- verificar como a camada copiável do PDF é codificada;
- listar operadores de texto presentes nos content streams (BT/ET/Tf/Tm/Td/Tj/TJ/Tr);
- verificar fontes, encoding e presença de ToUnicode;
- NÃO tentar interpretar tolerância/datum ainda.

Uso:
    python validation/gdt/scripts/run_phase5_content_stream_diagnostic.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import fitz

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CASE_ID = "case_41_rev8"
CASE_PATH = PROJECT_ROOT / "validation" / "gdt" / "cases" / f"{CASE_ID}.json"
OUTPUT_DIR = PROJECT_ROOT / "validation" / "gdt" / "outputs" / "phase5" / CASE_ID / "content_stream_diagnostic"
OUTPUT_PATH = OUTPUT_DIR / "content_stream_summary.json"

TEXT_OPERATORS = ("BT", "ET", "Tf", "Tm", "Td", "TD", "T*", "Tj", "TJ", "Tr")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_json(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _safe_json(v) for k, v in value.items()}
    return str(value)


def _count_operator(stream: bytes, operator: str) -> int:
    token = re.escape(operator.encode("ascii"))
    pattern = rb"(?<![A-Za-z0-9*'])" + token + rb"(?![A-Za-z0-9*'])"
    return len(re.findall(pattern, stream))


def _count_invisible_text_mode(stream: bytes) -> int:
    return len(re.findall(rb"(?<!\S)3\s+Tr(?!\S)", stream))


def _xref_key(doc: fitz.Document, xref: int, key: str) -> dict:
    try:
        kind, value = doc.xref_get_key(xref, key)
        return {"type": kind, "value": value}
    except Exception as exc:
        return {"error": str(exc)}


def _extract_xref_from_ref(value: str | None) -> int | None:
    if not value:
        return None
    match = re.search(r"(\d+)\s+0\s+R", value)
    return int(match.group(1)) if match else None


def main() -> None:
    case = _load(CASE_PATH)
    pdf_path = PROJECT_ROOT / case["pdf"]
    page_index = int(case.get("page_index", 0))

    doc = fitz.open(pdf_path)
    try:
        page = doc[page_index]
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        content_xrefs = page.get_contents()
        if isinstance(content_xrefs, int):
            content_xrefs = [content_xrefs]
        content_xrefs = list(content_xrefs or [])

        stream_rows = []
        for index, xref in enumerate(content_xrefs):
            try:
                stream = doc.xref_stream(xref) or b""
            except Exception as exc:
                stream_rows.append({"xref": xref, "error": str(exc)})
                continue

            operator_counts = {op: _count_operator(stream, op) for op in TEXT_OPERATORS}
            row = {
                "xref": xref,
                "byte_count": len(stream),
                "operator_counts": operator_counts,
                "invisible_text_mode_3_Tr_count": _count_invisible_text_mode(stream),
            }
            stream_rows.append(row)

            dump_path = OUTPUT_DIR / f"content_stream_{index:02d}_xref_{xref}.txt"
            dump_path.write_text(stream.decode("latin-1", errors="replace"), encoding="utf-8")

        font_rows = []
        for font in page.get_fonts(full=True):
            values = list(font)
            xref = int(values[0]) if values and isinstance(values[0], int) else None
            row: dict[str, Any] = {"raw": _safe_json(values), "xref": xref}
            if xref and xref > 0:
                row["Subtype"] = _xref_key(doc, xref, "Subtype")
                row["BaseFont"] = _xref_key(doc, xref, "BaseFont")
                row["Encoding"] = _xref_key(doc, xref, "Encoding")
                row["ToUnicode"] = _xref_key(doc, xref, "ToUnicode")
                row["DescendantFonts"] = _xref_key(doc, xref, "DescendantFonts")

                to_unicode_value = row.get("ToUnicode", {}).get("value")
                to_unicode_xref = _extract_xref_from_ref(to_unicode_value)
                row["to_unicode_xref"] = to_unicode_xref
                if to_unicode_xref:
                    try:
                        cmap = doc.xref_stream(to_unicode_xref) or b""
                        cmap_name = f"font_{xref}_ToUnicode_xref_{to_unicode_xref}.txt"
                        (OUTPUT_DIR / cmap_name).write_text(
                            cmap.decode("latin-1", errors="replace"), encoding="utf-8"
                        )
                        row["to_unicode_dump"] = cmap_name
                    except Exception as exc:
                        row["to_unicode_error"] = str(exc)
            font_rows.append(row)

        payload = {
            "schema_version": 1,
            "phase": "phase5_content_stream_diagnostic",
            "case_id": CASE_ID,
            "validation_status": "DIAGNOSTIC_ONLY",
            "ocr_used": False,
            "llm_used": False,
            "page_index": page_index,
            "content_streams": stream_rows,
            "fonts": font_rows,
        }
        OUTPUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        print("phase=phase5_content_stream_diagnostic")
        print("validation_status=DIAGNOSTIC_ONLY")
        print("ocr_used=False")
        print("llm_used=False")
        print(f"content_stream_count={len(stream_rows)}")
        print(f"font_count={len(font_rows)}")
        print("\ncontent_streams:")
        for row in stream_rows:
            if "error" in row:
                print(f"  xref={row['xref']} error={row['error']}")
                continue
            counts = row["operator_counts"]
            compact = " ".join(f"{op}={counts[op]}" for op in TEXT_OPERATORS)
            print(
                f"  xref={row['xref']} bytes={row['byte_count']} {compact} "
                f"3Tr={row['invisible_text_mode_3_Tr_count']}"
            )

        print("\nfonts:")
        for row in font_rows:
            xref = row.get("xref")
            if not xref:
                print(f"  raw={row.get('raw')}")
                continue
            basefont = row.get("BaseFont", {}).get("value")
            subtype = row.get("Subtype", {}).get("value")
            encoding = row.get("Encoding", {}).get("value")
            tounicode = row.get("ToUnicode", {}).get("value")
            print(
                f"  xref={xref} subtype={subtype!r} basefont={basefont!r} "
                f"encoding={encoding!r} ToUnicode={tounicode!r}"
            )

        print(f"\noutput={OUTPUT_PATH}")
        print(f"stream_dumps={OUTPUT_DIR}")
    finally:
        doc.close()


if __name__ == "__main__":
    main()
