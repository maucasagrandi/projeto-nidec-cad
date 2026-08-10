"""Part-classification branch for the integrated CAD Review pipeline.

The existing LLM remains responsible for extracting component/material/cited
standards from vector PDF text. Standards applicability remains deterministic.
Compressor series is external context: Phase 8 temporarily uses ``ALL`` until
Windchill supplies the real value; the LLM classification itself is not altered.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

from src.utils.standards_applicability import (
    DEFAULT_NORMAS_PATH,
    StandardsApplicabilityEngine,
    compare_standards,
    extract_note_number,
    normalize_standard,
)

DEFAULT_COMPRESSOR_SERIES_CONTEXT = "ALL"
DEFAULT_COMPRESSOR_SERIES_SOURCE = "temporary_default_until_windchill"


def _dump(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool, list, dict)):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return str(value)


def _default_classifier(text: str, prompt: str):
    # Lazy import avoids initializing the Vertex/Gemini client merely by importing
    # the deterministic Phase 8 aggregation modules or running unit tests.
    from src.modeling.llm_models import classify_cad_enriched

    return classify_cad_enriched(text, prompt)


def _extract_pdf_text(pdf_bytes: bytes, page_index: int) -> str:
    from src.modeling.llm_models import extract_text_from_pdf

    return extract_text_from_pdf(pdf_bytes, page_index=page_index)


def run_part_classification_branch(
    pdf_bytes: bytes,
    *,
    classification_prompt: str,
    page_index: int = 0,
    compressor_series_context: str = DEFAULT_COMPRESSOR_SERIES_CONTEXT,
    compressor_series_source: str = DEFAULT_COMPRESSOR_SERIES_SOURCE,
    normas_path: Path = DEFAULT_NORMAS_PATH,
    classifier_fn: Optional[Callable[[str, str], Any]] = None,
    applicability_engine: Optional[Any] = None,
    extracted_text: Optional[str] = None,
) -> dict:
    """Run LLM classification + deterministic standards applicability.

    ``compressor_series_context`` is deliberately separate from
    ``classification.compressor_series``. The latter remains whatever the LLM
    explicitly extracted from the CAD (normally ``None`` for current drawings).
    """

    text = extracted_text if extracted_text is not None else _extract_pdf_text(pdf_bytes, page_index)
    classifier = classifier_fn or _default_classifier
    classified = classifier(text, classification_prompt)
    if isinstance(classified, tuple) and len(classified) == 2:
        classification, metadata = classified
    else:
        classification, metadata = classified, None

    classification_dict = _dump(classification)
    if not isinstance(classification_dict, dict):
        raise TypeError("part classifier must return a mapping/Pydantic model")

    component_field = classification_dict.get("component") or {}
    material_field = classification_dict.get("material_family") or {}
    component = str(component_field.get("value") or "").strip()
    material_family = material_field.get("value")

    cited_standards = []
    for raw in classification_dict.get("cited_standards", []) or []:
        row = _dump(raw)
        standard_raw = str(row.get("standard", "")).strip()
        evidence = str(row.get("evidence", "")).strip()
        if not standard_raw:
            continue
        cited_standards.append(
            {
                "standard": normalize_standard(standard_raw),
                "standard_raw": standard_raw,
                "note_number": extract_note_number(evidence),
                "source_text": evidence,
            }
        )

    engine = applicability_engine or StandardsApplicabilityEngine(normas_path)
    applicability = engine.get_applicable_standards(
        component=component,
        compressor_series=str(compressor_series_context),
        material_family=str(material_family) if material_family else None,
    )
    applicability_dict = _dump(applicability)
    applicable_rows = applicability_dict.get("applicable_standards", []) or []

    comparison = compare_standards(
        applicable_standards=[str(row.get("standard")) for row in applicable_rows],
        cited_standards=[str(row.get("standard")) for row in cited_standards],
        unresolved_fields=applicability_dict.get("unresolved_fields", []),
    )

    return {
        "classification": classification_dict,
        "cited_standards": cited_standards,
        "applicable_standards": applicable_rows,
        "standards_comparison": _dump(comparison),
        "review_context": {
            "compressor_series": str(compressor_series_context),
            "compressor_series_source": str(compressor_series_source),
        },
        "provenance": {
            "classification_source": "existing_llm_part_classification",
            "text_source": "pymupdf_vector_text",
            "standards_applicability_source": str(normas_path),
            "compressor_series_source": str(compressor_series_source),
            "llm_metadata": _dump(metadata),
        },
    }


__all__ = [
    "DEFAULT_COMPRESSOR_SERIES_CONTEXT",
    "DEFAULT_COMPRESSOR_SERIES_SOURCE",
    "run_part_classification_branch",
]
