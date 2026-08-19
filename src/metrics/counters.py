"""Deterministic counters for CAD drawing metrics.

Provides:
    count_gdt_and_datums(review_json)  — from integrated_review.json
    count_cotas(pdf_path_or_bytes)     — from PDF via PyMuPDF regex
"""

from __future__ import annotations

import re
import statistics
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# GD&T and Datums — read from integrated_review.json
# ---------------------------------------------------------------------------

def count_gdt_and_datums(review_json: dict[str, Any]) -> dict[str, int]:
    """Count GD&T detections and datum definitions from an integrated review JSON.

    Args:
        review_json: Parsed content of integrated_review.json.

    Returns:
        Dict with keys:
            total_gdts      — sum of total_detections across all pages
            total_datums    — sum of datum_definitions_found across all pages
    """
    total_gdts = 0
    total_datums = 0

    for page in review_json.get("gdt", []):
        summary = page.get("summary", {})
        total_gdts   += int(summary.get("total_detections",       0) or 0)
        total_datums += int(summary.get("datum_definitions_found", 0) or 0)

    return {
        "total_gdts":   total_gdts,
        "total_datums": total_datums,
    }


# ---------------------------------------------------------------------------
# Cotas — deterministic regex on PDF text (adapted from scripts/cotas.py)
# ---------------------------------------------------------------------------

# Regex that matches dimensional values:
#   103,9  /  3.5  /  25±0.1  /  (12,5)  /  3.5+0.1  /  5,0-0,2
_REGEX_COTA = re.compile(
    r'^\(?'
    r'\d+[.,]\d*'                        # integer part + decimal separator + decimals
    r'(?:\s*[±+\-]\s*\d+[.,]\d+)?'      # optional tolerance: ±0.1 or +0.1 or -0.1
    r'\)?$'
)

# Grid markers in drawing margins
_REGEX_LETRA  = re.compile(r'^[A-H]$', re.IGNORECASE)
_REGEX_NUMERO = re.compile(r'^(?:[1-9]|1[0-5])$')


def _calcular_limites(marcadores_dict: dict) -> list[dict]:
    if not marcadores_dict:
        return []
    marcadores_ordenados = sorted(marcadores_dict.items(), key=lambda item: item[1])
    limites = []
    for i in range(len(marcadores_ordenados) - 1):
        id_atual, coord_atual = marcadores_ordenados[i]
        _id_prox, coord_prox  = marcadores_ordenados[i + 1]
        limites.append({"id": id_atual, "fim": (coord_atual + coord_prox) / 2.0})
    limites.append({"id": marcadores_ordenados[-1][0], "fim": float("inf")})
    return limites


def _localizar_quadrante(x: float, y: float, limites_x: list, limites_y: list) -> str:
    linha  = "?"
    coluna = "?"
    for lim in limites_x:
        if x <= lim["fim"]:
            coluna = lim["id"]
            break
    for lim in limites_y:
        if y <= lim["fim"]:
            linha = lim["id"]
            break
    return f"{linha}{coluna}"


def count_cotas(pdf_source: str | Path | bytes) -> dict[str, Any]:
    """Count dimensional annotations (cotas) in a CAD drawing PDF.

    Works on all pages (not just page 0).
    Returns gracefully with count=0 if grid markers are not found.

    Args:
        pdf_source: Path to PDF file, Path object, or raw PDF bytes.

    Returns:
        Dict with keys:
            total_cotas     — total count across all pages
            per_page        — list of per-page counts
            cotas_detail    — list of {cota, quadrante, page, bbox} for all found cotas
    """
    import fitz  # PyMuPDF

    if isinstance(pdf_source, (str, Path)):
        doc = fitz.open(str(pdf_source))
    else:
        doc = fitz.open(stream=pdf_source, filetype="pdf")

    all_cotas: list[dict] = []
    per_page:  list[int]  = []

    try:
        for page_idx in range(len(doc)):
            page = doc[page_idx]

            largura_pagina = page.rect.width
            altura_pagina  = page.rect.height

            ZONA_TOPO     = altura_pagina  * 0.08
            ZONA_BASE     = altura_pagina  * 0.92
            ZONA_ESQUERDA = largura_pagina * 0.08
            ZONA_DIREITA  = largura_pagina * 0.92

            text_data = page.get_text("dict")

            letras_y:    dict[str, list[float]] = {}
            numeros_x:   dict[str, list[float]] = {}
            cotas_brutas: list[dict] = []

            for block in text_data.get("blocks", []):
                if "lines" not in block:
                    continue
                for line in block["lines"]:
                    for span in line["spans"]:
                        texto = span["text"].strip().upper()
                        if not texto:
                            continue

                        bbox     = span["bbox"]
                        centro_x = (bbox[0] + bbox[2]) / 2.0
                        centro_y = (bbox[1] + bbox[3]) / 2.0

                        if _REGEX_LETRA.fullmatch(texto) and (
                            centro_x < ZONA_ESQUERDA or centro_x > ZONA_DIREITA
                        ):
                            letras_y.setdefault(texto, []).append(centro_y)

                        elif _REGEX_NUMERO.fullmatch(texto) and (
                            centro_y < ZONA_TOPO or centro_y > ZONA_BASE
                        ):
                            numeros_x.setdefault(texto, []).append(centro_x)

                        elif _REGEX_COTA.fullmatch(texto):
                            cotas_brutas.append({
                                "valor":   texto,
                                "x":       centro_x,
                                "y":       centro_y,
                                "bbox":    bbox,
                                "page":    page_idx,
                            })

            # Build grid — fallback to "?" quadrant if no markers found
            try:
                marcadores_y = {l: statistics.median(c) for l, c in letras_y.items()}
                marcadores_x = {n: statistics.median(c) for n, c in numeros_x.items()}
                limites_y    = _calcular_limites(marcadores_y)
                limites_x    = _calcular_limites(marcadores_x)
            except statistics.StatisticsError:
                limites_y = []
                limites_x = []

            page_cotas: list[dict] = []
            for cota in cotas_brutas:
                quadrante = _localizar_quadrante(cota["x"], cota["y"], limites_x, limites_y)
                page_cotas.append({
                    "cota":      cota["valor"],
                    "quadrante": quadrante,
                    "page":      page_idx,
                    "bbox":      cota["bbox"],
                })

            per_page.append(len(page_cotas))
            all_cotas.extend(page_cotas)

    finally:
        doc.close()

    return {
        "total_cotas":  len(all_cotas),
        "per_page":     per_page,
        "cotas_detail": all_cotas,
    }
