"""Batch result exporters for CAD Review validation.

Two workbooks are intentionally produced:

``summary_engineering.xlsx``
    Mirrors the existing NIDEC/engineering validation layout so engineers can
    add ground truth / feedback without learning a new format.

``summary_technical.xlsx``
    Pipeline-oriented operational view with GD&T/ISO counts and artifact paths.

The XLSX writer uses only Python's standard library and emits a small, valid
Office Open XML workbook; no spreadsheet runtime dependency is required here.
"""

from __future__ import annotations

import json
import math
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from xml.sax.saxutils import escape

ENGINEERING_HEADERS = [
    "CAD",
    "Classificação_Ground_Truth",
    "Normas_Ground_Truth",
    "Classificação_LLM",
    "Match_Classif",
    "Normas_LLM",
    "Match_Normas",
    "Normas_Sugeridas_LLM",
    "Reasoning_Sugeridas",
    "Precisamos do feedback time NIDEC sobre Match reasoning (Plausível ou não)",
    "Input_Tokens",
    "Output_Tokens",
    "Latência (ms)",
    "Justificativas_Normas",
    "Observações",
]

TECHNICAL_HEADERS = [
    "CAD",
    "Status_Execucao",
    "Classificacao_LLM",
    "Normas_Detectadas",
    "Normas_Aplicaveis_ALL",
    "Normas_Aplicaveis_Faltantes",
    "Compressor_Series_Context",
    "Compressor_Series_Source",
    "GDT_Candidate_Count",
    "GDT_Classified_Count",
    "Datum_Reference_Count",
    "Datum_Definition_Count",
    "ISO1101_PASS",
    "ISO1101_WARNING",
    "ISO1101_NEEDS_CONTEXT",
    "ISO5459_PASS",
    "ISO5459_WARNING",
    "Overall_PASS",
    "Overall_WARNING",
    "Overall_NEEDS_CONTEXT",
    "Overall_NOT_EVALUATED",
    "Annotated_Images",
    "Result_JSON",
    "Input_Tokens",
    "Output_Tokens",
    "Latencia_ms",
    "Validation_Status",
    "Erro_Processamento",
]


def _scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple, set)):
        return "; ".join(str(v) for v in value)
    return str(value)


def _col_name(index: int) -> str:
    result = ""
    n = index + 1
    while n:
        n, rem = divmod(n - 1, 26)
        result = chr(65 + rem) + result
    return result


def _cell_xml(ref: str, value: Any, *, style: int = 0) -> str:
    value = _scalar(value)
    if value is None:
        return f'<c r="{ref}" s="{style}"/>'
    if isinstance(value, bool):
        return f'<c r="{ref}" s="{style}" t="b"><v>{1 if value else 0}</v></c>'
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        return f'<c r="{ref}" s="{style}"><v>{value}</v></c>'
    text = escape(str(value))
    preserve = ' xml:space="preserve"' if text != text.strip() or "\n" in text else ""
    return f'<c r="{ref}" s="{style}" t="inlineStr"><is><t{preserve}>{text}</t></is></c>'


def _sheet_xml(headers: Sequence[str], rows: Sequence[Mapping[str, Any]], widths: Sequence[float]) -> str:
    max_row = len(rows) + 1
    max_col = len(headers)
    dimension = f"A1:{_col_name(max_col - 1)}{max_row}"
    cols = "".join(
        f'<col min="{i + 1}" max="{i + 1}" width="{float(width):.1f}" customWidth="1"/>'
        for i, width in enumerate(widths)
    )
    row_xml = []
    header_cells = "".join(_cell_xml(f"{_col_name(i)}1", header, style=1) for i, header in enumerate(headers))
    row_xml.append(f'<row r="1" ht="32" customHeight="1">{header_cells}</row>')
    for r_idx, row in enumerate(rows, start=2):
        cells = []
        for c_idx, header in enumerate(headers):
            value = row.get(header)
            style = 2 if isinstance(value, str) and ("\n" in value or len(value) > 60) else 0
            cells.append(_cell_xml(f"{_col_name(c_idx)}{r_idx}", value, style=style))
        row_xml.append(f'<row r="{r_idx}">{"".join(cells)}</row>')
    auto_filter = f'<autoFilter ref="A1:{_col_name(max_col - 1)}{max_row}"/>' if max_row >= 1 else ""
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<dimension ref="{dimension}"/>'
        '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>'
        '<sheetFormatPr defaultRowHeight="18"/>'
        f'<cols>{cols}</cols><sheetData>{"".join(row_xml)}</sheetData>{auto_filter}'
        '</worksheet>'
    )


def write_xlsx(
    path: str | Path,
    *,
    headers: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    widths: Sequence[float] | None = None,
) -> Path:
    """Write one-sheet XLSX with header style, filters, freeze row and wrapping."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if widths is None:
        widths = [22.0] * len(headers)
    if len(widths) != len(headers):
        raise ValueError("widths length must match headers length")

    content_types = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>'''
    root_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>'''
    workbook = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Results" sheetId="1" r:id="rId1"/></sheets></workbook>'''
    workbook_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>'''
    styles = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
<fonts count="2"><font><sz val="10"/><name val="Calibri"/></font><font><b/><color rgb="FFFFFFFF"/><sz val="10"/><name val="Calibri"/></font></fonts>
<fills count="3"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill><fill><patternFill patternType="solid"><fgColor rgb="FF1F4E78"/><bgColor indexed="64"/></patternFill></fill></fills>
<borders count="2"><border/><border><left style="thin"><color rgb="FFD9E1F2"/></left><right style="thin"><color rgb="FFD9E1F2"/></right><top style="thin"><color rgb="FFD9E1F2"/></top><bottom style="thin"><color rgb="FFD9E1F2"/></bottom></border></borders>
<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
<cellXfs count="3"><xf numFmtId="0" fontId="0" fillId="0" borderId="1" xfId="0" applyBorder="1"><alignment vertical="top"/></xf><xf numFmtId="0" fontId="1" fillId="2" borderId="1" xfId="0" applyFont="1" applyFill="1" applyBorder="1"><alignment horizontal="center" vertical="center" wrapText="1"/></xf><xf numFmtId="0" fontId="0" fillId="0" borderId="1" xfId="0" applyBorder="1"><alignment vertical="top" wrapText="1"/></xf></cellXfs>
<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>'''
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    core = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"><dc:creator>CAD Review Pipeline</dc:creator><dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created></cp:coreProperties>'''
    app = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"><Application>CAD Review Pipeline</Application></Properties>'''
    sheet = _sheet_xml(headers, rows, widths)

    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", root_rels)
        zf.writestr("xl/workbook.xml", workbook)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        zf.writestr("xl/styles.xml", styles)
        zf.writestr("xl/worksheets/sheet1.xml", sheet)
        zf.writestr("docProps/core.xml", core)
        zf.writestr("docProps/app.xml", app)
    return target


def engineering_row(result: Mapping[str, Any]) -> dict:
    classification = result.get("part_classification") or {}
    component = (classification.get("component") or {}).get("value")
    cited = result.get("cited_standards") or []
    comparison = result.get("standards_comparison") or {}
    missing = comparison.get("missing") or []
    provenance = (result.get("provenance") or {}).get("part_classification") or {}
    metadata = provenance.get("llm_metadata") or {}
    reasons = [str(row.get("source_text") or "") for row in cited if row.get("source_text")]
    summary = result.get("summary") or {}
    gdt = result.get("gdt_frames") or result.get("gdt_candidates") or []
    datum_defs = result.get("datum_definitions") or []
    missing_text = "; ".join(str(v) for v in missing) if missing else "none"
    notes = (
        f"GDT candidates={len(gdt)}; datum definitions={len(datum_defs)}; "
        f"warnings={summary.get('WARNING', 0)}; needs_context={summary.get('NEEDS_CONTEXT', 0)}; "
        f"deterministic applicable standards missing={missing_text}. "
        "Legacy free-form LLM standard-suggestion columns are intentionally left blank."
    )
    return {
        "CAD": (result.get("drawing") or {}).get("name"),
        "Classificação_Ground_Truth": "",
        "Normas_Ground_Truth": "",
        "Classificação_LLM": component or "",
        "Match_Classif": "",
        "Normas_LLM": "; ".join(str(row.get("standard")) for row in cited if row.get("standard")),
        "Match_Normas": "",
        "Normas_Sugeridas_LLM": "",
        "Reasoning_Sugeridas": "",
        "Precisamos do feedback time NIDEC sobre Match reasoning (Plausível ou não)": "",
        "Input_Tokens": metadata.get("prompt_tokens"),
        "Output_Tokens": metadata.get("completion_tokens"),
        "Latência (ms)": metadata.get("latency_ms"),
        "Justificativas_Normas": "\n".join(reasons),
        "Observações": notes,
    }


def technical_row(result: Mapping[str, Any], *, result_json_path: str, error: str | None = None) -> dict:
    classification = result.get("part_classification") or {}
    component = (classification.get("component") or {}).get("value")
    cited = result.get("cited_standards") or []
    applicable = result.get("applicable_standards") or []
    comparison = result.get("standards_comparison") or {}
    gdt = result.get("gdt_frames") or result.get("gdt_candidates") or []
    datum_defs = result.get("datum_definitions") or []
    findings = result.get("findings") or []
    context = result.get("review_context") or {}
    summary = result.get("summary") or {}
    provenance = (result.get("provenance") or {}).get("part_classification") or {}
    metadata = provenance.get("llm_metadata") or {}
    visuals = (result.get("artifacts") or {}).get("visual_evidence") or {}
    images = [str(row.get("annotated_image")) for row in visuals.get("pages", []) if row.get("annotated_image")]

    def count(domain: str, status: str) -> int:
        return sum(1 for row in findings if row.get("domain") == domain and row.get("status") == status)

    ref_count = sum(len(row.get("referenced_datums") or []) for row in gdt)
    classified_count = sum(bool(row.get("characteristic")) for row in gdt)
    return {
        "CAD": (result.get("drawing") or {}).get("name"),
        "Status_Execucao": "ERROR" if error else "OK",
        "Classificacao_LLM": component or "",
        "Normas_Detectadas": "; ".join(str(row.get("standard")) for row in cited if row.get("standard")),
        "Normas_Aplicaveis_ALL": "; ".join(str(row.get("standard")) for row in applicable if row.get("standard")),
        "Normas_Aplicaveis_Faltantes": "; ".join(str(v) for v in (comparison.get("missing") or [])),
        "Compressor_Series_Context": context.get("compressor_series", "ALL"),
        "Compressor_Series_Source": context.get("compressor_series_source", "temporary_default_until_windchill"),
        "GDT_Candidate_Count": len(gdt),
        "GDT_Classified_Count": classified_count,
        "Datum_Reference_Count": ref_count,
        "Datum_Definition_Count": len(datum_defs),
        "ISO1101_PASS": count("iso1101", "PASS"),
        "ISO1101_WARNING": count("iso1101", "WARNING"),
        "ISO1101_NEEDS_CONTEXT": count("iso1101", "NEEDS_CONTEXT"),
        "ISO5459_PASS": count("iso5459", "PASS"),
        "ISO5459_WARNING": count("iso5459", "WARNING"),
        "Overall_PASS": summary.get("PASS", 0),
        "Overall_WARNING": summary.get("WARNING", 0),
        "Overall_NEEDS_CONTEXT": summary.get("NEEDS_CONTEXT", 0),
        "Overall_NOT_EVALUATED": summary.get("NOT_EVALUATED", 0),
        "Annotated_Images": "; ".join(images),
        "Result_JSON": result_json_path,
        "Input_Tokens": metadata.get("prompt_tokens"),
        "Output_Tokens": metadata.get("completion_tokens"),
        "Latencia_ms": metadata.get("latency_ms"),
        "Validation_Status": result.get("validation_status", "BATCH_VALIDATION_ONLY"),
        "Erro_Processamento": error or "",
    }


def write_batch_workbooks(
    output_dir: str | Path,
    *,
    engineering_rows: Sequence[Mapping[str, Any]],
    technical_rows: Sequence[Mapping[str, Any]],
) -> dict:
    output = Path(output_dir)
    eng_path = output / "summary_engineering.xlsx"
    tech_path = output / "summary_technical.xlsx"
    eng_widths = [28, 28, 38, 28, 14, 42, 14, 42, 48, 42, 14, 14, 16, 55, 55]
    tech_widths = [30, 18, 30, 42, 42, 42, 18, 34, 18, 18, 18, 18, 14, 16, 20, 14, 16, 14, 16, 20, 20, 55, 50, 14, 14, 16, 26, 55]
    write_xlsx(eng_path, headers=ENGINEERING_HEADERS, rows=engineering_rows, widths=eng_widths)
    write_xlsx(tech_path, headers=TECHNICAL_HEADERS, rows=technical_rows, widths=tech_widths)
    return {"engineering": eng_path.name, "technical": tech_path.name}


def write_manifest(
    output_dir: str | Path,
    entries: Sequence[Mapping[str, Any]],
    *,
    workbook_paths: Mapping[str, str],
) -> Path:
    output = Path(output_dir)
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "workbooks": dict(workbook_paths),
        "entries": [dict(row) for row in entries],
    }
    path = output / "manifest.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


__all__ = [
    "ENGINEERING_HEADERS",
    "TECHNICAL_HEADERS",
    "engineering_row",
    "technical_row",
    "write_batch_workbooks",
    "write_manifest",
    "write_xlsx",
]
