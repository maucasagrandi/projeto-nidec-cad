"""Deterministic counters used by the report and validation spreadsheet."""

from __future__ import annotations

import re
from typing import Any

import fitz


_DIMENSION_TOKEN = re.compile(
    r"^\(?\d+[.,]\d*(?:\s*[±+\-]\s*\d+[.,]\d+)?\)?$"
)


def count_cotas(pdf_bytes: bytes) -> int:
    """Count decimal dimension tokens in all pages of the revised PDF."""

    total = 0
    with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
        for page in document:
            for block in page.get_text("dict").get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        token = str(span.get("text", "")).strip()
                        if token and _DIMENSION_TOKEN.fullmatch(token):
                            total += 1
    return total


def _as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _datum_labels(gdt_pages: list[Any]) -> list[str]:
    labels: list[str] = []
    for page in gdt_pages:
        report = page.report if hasattr(page, "report") else page
        for definition in report.get("datum_definitions", []) or []:
            label = definition.get("label") or definition.get("datum_label")
            if label is not None:
                normalized = str(label).strip().upper()
                if normalized and normalized not in labels:
                    labels.append(normalized)
    return labels


def build_objective_metrics(
    revised_pdf: bytes,
    classification: dict[str, Any],
    gdt_pages: list[Any],
) -> dict[str, Any]:
    """Build the ten customer-facing Objective Metrics in display order."""

    total_gdts = 0
    total_datums = 0
    for page in gdt_pages:
        report = page.report if hasattr(page, "report") else page
        summary = report.get("summary", {}) or {}
        total_gdts += int(summary.get("total_detections", 0) or 0)
        total_datums += int(summary.get("datum_definitions_found", 0) or 0)

    datum_labels = _datum_labels(gdt_pages)
    return {
        "Quantidade de cotas": count_cotas(revised_pdf),
        "Quantidade de cotas HIC": _as_int(classification.get("quantidade_cotas_hic")),
        "Quantidade de cotas CTQ": _as_int(classification.get("quantidade_cotas_ctq")),
        "Quantidade de cotas CTQ-S": _as_int(classification.get("quantidade_cotas_ctq_s")),
        "Quantidade de GD&Ts": total_gdts,
        "Quantidade de Datums Reference": total_datums,
        "Lista de datums reference": datum_labels,
        "Quantidade de revisões": _as_int(classification.get("quantidade_revisoes")),
        "Quantidade de notas": _as_int(classification.get("quantidade_notas")),
        "Quantidade de códigos": _as_int(classification.get("quantidade_codigos")),
    }


__all__ = ["build_objective_metrics", "count_cotas"]
