from __future__ import annotations

from dataclasses import dataclass

import fitz
import numpy as np

from src.cad_review.integrated_review import GdtPageResult, run_integrated_review
from src.modeling.llm_verify_changes import (
    VerificationResult,
    VerifiedChange,
    render_verified_highlights,
    save_verification_result,
)
from src.reporting.unified_cad_report import _image_flowable, build_unified_report


def _pdf_with_text(text: str, pages: int = 1) -> bytes:
    document = fitz.open()
    for index in range(pages):
        page = document.new_page(width=400, height=300)
        page.insert_text((30, 50), f"{text} page {index + 1}")
    payload = document.tobytes()
    document.close()
    return payload


@dataclass
class _Metadata:
    total_tokens: int = 10


def test_integrated_review_uses_revised_for_classification_and_gdt() -> None:
    original = _pdf_with_text("ORIGINAL ONLY")
    revised = _pdf_with_text("REVISED CONNECTING ROD ISO 1101", pages=2)
    calls: dict[str, object] = {"gdt_pages": []}

    def classifier(text, prompt, model):
        assert "REVISED CONNECTING ROD ISO 1101" in text
        assert "ORIGINAL ONLY" not in text
        assert "REVISED CONNECTING ROD ISO 1101" in prompt
        calls["classification_model"] = model
        return {
            "classificacao": "Connecting Rod",
            "justificativa_classificacao": "CONNECTING ROD",
            "lista_normas": ["ISO 1101"],
            "justificativas_normas": ["ISO 1101"],
        }, _Metadata()

    def inferer(classification, cited, prompt, model):
        assert classification == "Connecting Rod"
        assert cited == ["ISO 1101"]
        return {"normas_sugeridas": [], "reasoning": "none", "confianca": 0.9}, _Metadata()

    def gdt_analyzer(pdf_bytes, page_index, **kwargs):
        assert pdf_bytes == revised
        assert kwargs["max_workers"] == 1
        calls["gdt_pages"].append(page_index)
        return GdtPageResult(
            page_index=page_index,
            report={
                "page": page_index + 1,
                "summary": {
                    "total_detections": 1,
                    "resolved_datum_refs": 1,
                    "datum_definitions_found": 1,
                },
            },
            annotated_image=np.full((80, 120, 3), 255, dtype=np.uint8),
        )

    def comparator(original_bytes, revised_bytes, **kwargs):
        assert original_bytes == original
        assert revised_bytes == revised
        calls["comparison_model"] = kwargs["model"]
        return [
            VerificationResult(
                page_index=0,
                true_changes=[
                    VerifiedChange(
                        index=1,
                        original_id="page_01_diff_001",
                        x=10,
                        y=12,
                        width=30,
                        height=20,
                        divergence_pct=42.0,
                        description="Dimension changed from 10 to 12 mm",
                    )
                ],
                false_positive_ids=[],
                image_highlighted=np.full((80, 240, 3), 255, dtype=np.uint8),
            )
        ]

    def format_checker(original_bytes, revised_bytes):
        assert original_bytes == original
        assert revised_bytes == revised
        return []

    result = run_integrated_review(
        original,
        revised,
        original_name="old.pdf",
        revised_name="new.pdf",
        classification_model="classification-model",
        comparison_model="comparison-model",
        classifier=classifier,
        standards_inferer=inferer,
        gdt_analyzer=gdt_analyzer,
        comparator=comparator,
        format_checker=format_checker,
    )

    assert calls == {
        "gdt_pages": [0, 1],
        "classification_model": "classification-model",
        "comparison_model": "comparison-model",
    }
    assert result.to_dict()["inputs"] == {"original": "old.pdf", "revised": "new.pdf"}
    assert result.to_dict()["comparison"]["pages"][0]["num_true_changes"] == 1

    report = build_unified_report(result)
    assert report.startswith(b"%PDF")
    assert len(report) > 2_000


def test_integrated_review_requires_both_pdfs() -> None:
    try:
        run_integrated_review(b"", b"revised")
    except ValueError as exc:
        assert "Both original and revised" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_integrated_review_rejects_invalid_gdt_workers() -> None:
    try:
        run_integrated_review(b"original", b"revised", gdt_workers=0)
    except ValueError as exc:
        assert "gdt_workers" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_comparison_artifacts_are_written_as_utf8(tmp_path) -> None:
    result = VerificationResult(
        page_index=0,
        true_changes=[
            VerifiedChange(
                index=1,
                original_id="page_01_diff_001",
                x=0,
                y=0,
                width=10,
                height=10,
                divergence_pct=50.0,
                description="Símbolo φ alterado",
            )
        ],
        false_positive_ids=[],
    )

    save_verification_result(result, tmp_path)

    assert "φ" in (tmp_path / "report.json").read_text(encoding="utf-8")
    assert "Símbolo φ alterado" in (tmp_path / "report.txt").read_text(encoding="utf-8")


def test_large_report_image_is_downsampled_before_reportlab_decodes_it() -> None:
    source = np.full((2500, 5000, 3), 255, dtype=np.uint8)

    flowable = _image_flowable(source, max_width=720, max_height=360)
    flowable._setup_inner()

    assert flowable.imageWidth <= 1440
    assert flowable.imageHeight <= 720
    assert flowable.drawWidth == 720
    assert flowable.drawHeight == 360


def test_comparison_panels_are_resized_before_side_by_side_allocation() -> None:
    original = np.full((1200, 2400, 3), 255, dtype=np.uint8)
    revised = original.copy()
    changes = [
        VerifiedChange(
            index=1,
            original_id="page_01_diff_001",
            x=1200,
            y=600,
            width=200,
            height=100,
            divergence_pct=50.0,
            description="Changed dimension",
        )
    ]

    combined = render_verified_highlights(
        original,
        revised,
        changes,
        max_output_height=400,
    )

    assert combined.shape == (400, 1600, 3)
    assert np.any(combined[:, 800:, 2] > combined[:, 800:, 1])
