"""End-to-end CAD review using one original and one revised PDF.

The revised drawing is the only input to Part Classification and deterministic
GD&T/datum detection.  The original and revised drawings are both passed to the
OpenCV comparison pipeline, whose candidate regions are verified by the LLM.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

import cv2
import fitz
import numpy as np


@dataclass
class GdtPageResult:
    """Structured and visual GD&T output for one revised drawing page."""

    page_index: int
    report: dict[str, Any]
    annotated_image: np.ndarray | None = field(default=None, repr=False)


@dataclass
class IntegratedReviewResult:
    """All outputs needed by the UI, JSON artifact and unified PDF report."""

    original_name: str
    revised_name: str
    part_classification: dict[str, Any]
    inferred_standards: dict[str, Any]
    gdt_pages: list[GdtPageResult]
    comparison_pages: list[Any]
    paper_format_changes: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "inputs": {
                "original": self.original_name,
                "revised": self.revised_name,
            },
            "part_classification": self.part_classification,
            "standards": {
                "cited": self.part_classification.get("lista_normas", []),
                "suggested": self.inferred_standards.get("normas_sugeridas", []),
                "inference": self.inferred_standards,
            },
            "gdt": [page.report for page in self.gdt_pages],
            "comparison": {
                "paper_format_changes": self.paper_format_changes,
                "pages": [_comparison_to_dict(page) for page in self.comparison_pages],
            },
            "metadata": self.metadata,
        }


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    raise TypeError(f"Cannot serialize {type(value)!r} as a mapping")


def _metadata_dict(value: Any) -> dict[str, Any]:
    return _as_dict(value) if value is not None else {}


def _comparison_to_dict(result: Any) -> dict[str, Any]:
    if hasattr(result, "report_json"):
        return result.report_json()
    return _as_dict(result)


def _default_format_checker(original_pdf: bytes, revised_pdf: bytes) -> list[dict[str, Any]]:
    from src.utils.paper_format import check_all_pages_format

    changes = check_all_pages_format(original_pdf, revised_pdf)
    return [
        {
            "page_index": page_index,
            "page": page_index + 1,
            "description": change.description,
            "format_changed": change.format_changed,
            "orientation_changed": change.orientation_changed,
            "original": _as_dict(change.original),
            "revised": _as_dict(change.revised),
        }
        for page_index, change in sorted(changes.items())
    ]


def extract_all_pdf_text(pdf_bytes: bytes) -> str:
    """Extract vector text from all revised drawing pages."""

    with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
        return "\n\n".join(page.get_text() for page in document)


def _default_classifier(text: str, prompt: str, model: str | None) -> tuple[Any, Any]:
    from src.modeling.llm_models import classify_and_extract_norms

    kwargs = {"model": model} if model else {}
    return classify_and_extract_norms(texto_notas=text, system_prompt=prompt, **kwargs)


def _default_inferer(
    classification: str,
    cited_standards: list[str],
    prompt: str,
    model: str | None,
) -> tuple[Any, Any]:
    from src.modeling.llm_models import infer_missing_norms

    kwargs = {"model": model} if model else {}
    return infer_missing_norms(
        classificacao=classification,
        lista_normas_atuais=cited_standards,
        system_prompt=prompt,
        **kwargs,
    )


def _default_gdt_analyzer(
    revised_pdf: bytes,
    page_index: int,
    *,
    template_root: Path,
    dpi: int,
    score_threshold: float,
    max_workers: int,
) -> GdtPageResult:
    from src.gdt.gdt_report import analyze_page, render_annotated_page

    report, detections, frames, extractions, datum_definitions = analyze_page(
        revised_pdf,
        page_index=page_index,
        template_root=str(template_root),
        dpi=dpi,
        score_threshold=score_threshold,
        max_workers=max_workers,
    )
    image = render_annotated_page(
        revised_pdf,
        detections,
        frames,
        extractions,
        datum_definitions,
        page_index=page_index,
        dpi=min(dpi, 200),
        verified_datum_defs=report.datum_definitions,
    )
    return GdtPageResult(page_index=page_index, report=report.to_dict(), annotated_image=image)


def _default_comparator(
    original_pdf: bytes,
    revised_pdf: bytes,
    *,
    model: str | None,
    opencv_config: Any,
) -> list[Any]:
    from src.modeling.llm_verify_changes import run_verification_pipeline_all_pages

    kwargs: dict[str, Any] = {"opencv_config": opencv_config}
    if model:
        kwargs["model"] = model
    return run_verification_pipeline_all_pages(original_pdf, revised_pdf, **kwargs)


def run_integrated_review(
    original_pdf: bytes,
    revised_pdf: bytes,
    *,
    original_name: str = "original.pdf",
    revised_name: str = "revised.pdf",
    classification_prompt: str | None = None,
    classification_model: str | None = None,
    comparison_model: str | None = None,
    gdt_dpi: int = 150,
    gdt_threshold: float = 0.74,
    gdt_workers: int = 1,
    template_root: str | Path = "assets/gdt/templates",
    opencv_config: Any = None,
    classifier: Callable[[str, str, str | None], tuple[Any, Any]] | None = None,
    standards_inferer: Callable[[str, list[str], str, str | None], tuple[Any, Any]] | None = None,
    gdt_analyzer: Callable[..., GdtPageResult] | None = None,
    comparator: Callable[..., list[Any]] | None = None,
    format_checker: Callable[[bytes, bytes], list[dict[str, Any]]] | None = None,
) -> IntegratedReviewResult:
    """Run classification, GD&T/datum detection and comparison in one flow."""

    if not original_pdf or not revised_pdf:
        raise ValueError("Both original and revised PDF bytes are required")
    if gdt_workers < 1:
        raise ValueError("gdt_workers must be at least 1")

    from prompts import classificacao_e_normas_prompt

    revised_text = extract_all_pdf_text(revised_pdf)
    prompt_template = classification_prompt or classificacao_e_normas_prompt
    prompt = prompt_template.replace("{{texto_extraido}}", revised_text)

    classify = classifier or _default_classifier
    classification_value, classification_metadata = classify(
        revised_text,
        prompt,
        classification_model,
    )
    classification = _as_dict(classification_value)

    cited_standards = [str(value) for value in classification.get("lista_normas", [])]
    inference_prompt = (
        "Você é especialista em normas técnicas de engenharia. Com base na classificação "
        "da peça e nas normas explicitamente citadas, indique somente normas adicionais que "
        "merecem validação humana. Não apresente uma recomendação como obrigação normativa."
    )
    infer = standards_inferer or _default_inferer
    inferred_value, inference_metadata = infer(
        str(classification.get("classificacao", "Não encontrado")),
        cited_standards,
        inference_prompt,
        classification_model,
    )
    inferred = _as_dict(inferred_value)

    with fitz.open(stream=revised_pdf, filetype="pdf") as document:
        revised_page_count = len(document)

    analyze_gdt = gdt_analyzer or _default_gdt_analyzer
    resolved_template_root = Path(template_root)
    gdt_pages = [
        analyze_gdt(
            revised_pdf,
            page_index,
            template_root=resolved_template_root,
            dpi=gdt_dpi,
            score_threshold=gdt_threshold,
            max_workers=gdt_workers,
        )
        for page_index in range(revised_page_count)
    ]

    compare = comparator or _default_comparator
    comparison_pages = compare(
        original_pdf,
        revised_pdf,
        model=comparison_model,
        opencv_config=opencv_config,
    )
    paper_format_changes = (format_checker or _default_format_checker)(original_pdf, revised_pdf)

    return IntegratedReviewResult(
        original_name=original_name,
        revised_name=revised_name,
        part_classification=classification,
        inferred_standards=inferred,
        gdt_pages=gdt_pages,
        comparison_pages=comparison_pages,
        paper_format_changes=paper_format_changes,
        metadata={
            "classification": _metadata_dict(classification_metadata),
            "standards_inference": _metadata_dict(inference_metadata),
            "revised_text_characters": len(revised_text),
            "revised_pages": revised_page_count,
            "compared_pages": len(comparison_pages),
            "gdt_mode": "deterministic_template_and_geometry",
            "gdt_workers": gdt_workers,
            "comparison_mode": "opencv_candidates_then_llm_verification",
        },
    )


def save_integrated_review(result: IntegratedReviewResult, output_dir: str | Path) -> dict[str, Path]:
    """Persist JSON, annotated images, comparison artifacts and final PDF."""

    from src.reporting.unified_cad_report import build_unified_report

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    gdt_dir = output / "gdt"
    comparison_dir = output / "comparison"
    gdt_dir.mkdir(exist_ok=True)
    comparison_dir.mkdir(exist_ok=True)

    for page in result.gdt_pages:
        if page.annotated_image is not None:
            cv2.imwrite(str(gdt_dir / f"page_{page.page_index + 1:03d}_annotated.png"), page.annotated_image)

    from src.modeling.llm_verify_changes import save_verification_result

    for page in result.comparison_pages:
        page_index = int(getattr(page, "page_index", 0))
        save_verification_result(page, comparison_dir / f"page_{page_index + 1:03d}")

    result_json = output / "integrated_review.json"
    result_json.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    report_pdf = output / "integrated_review_report.pdf"
    report_pdf.write_bytes(build_unified_report(result))
    return {"json": result_json, "report": report_pdf, "gdt": gdt_dir, "comparison": comparison_dir}


__all__ = [
    "GdtPageResult",
    "IntegratedReviewResult",
    "extract_all_pdf_text",
    "run_integrated_review",
    "save_integrated_review",
]
