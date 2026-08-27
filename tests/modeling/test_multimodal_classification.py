from __future__ import annotations

import json
from types import SimpleNamespace

from src.modeling import llm_models


def test_classification_sends_pdf_and_returns_structured_drawing_metadata(monkeypatch) -> None:
    captured = {}
    payload = {
        "header": {
            "drawing_number": "13358002",
            "title": "GASKET - VALVE PLATE",
            "compressor_series_code": None,
            "cr": "26177",
            "classification": "GASKET - VALVE PLATE",
            "last_revision_date": "2025.12.18",
        },
        "drawing_block": {
            "materials": ["NI-2085-G", "GZL-2002"],
            "material_code": "TSS 002259",
            "drawn_by": "ALEXANDRE PEREIRA",
            "approved_by": "EMILIO HULSE",
            "drawing_code_ecm": "26177",
            "date": "2012.NOV.13",
            "name_and_document_type": "GASKET - VALVE PLATE",
            "general_tolerance": None,
            "angular_tolerance": None,
            "scale": "5:2",
            "unit": "mm",
            "replace": None,
            "number": "13358002",
        },
        "classificacao": "GASKET - VALVE PLATE",
        "justificativa_classificacao": "TITLE, DOCUMENT TYPE",
        "lista_normas": ["ISO STANDARDS", "TSS 002513"],
        "justificativas_normas": ["Note 1", "Note 1"],
        "quantidade_revisoes": 7,
        "quantidade_notas": 15,
        "quantidade_codigos": 7,
    }

    class FakeModels:
        def generate_content(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                text=json.dumps(payload),
                usage_metadata=SimpleNamespace(
                    total_token_count=100,
                    prompt_token_count=80,
                    candidates_token_count=20,
                ),
            )

    monkeypatch.setattr(llm_models, "client", SimpleNamespace(models=FakeModels()))

    result, _ = llm_models.classify_and_extract_norms(
        texto_notas="vector text",
        system_prompt="structured prompt",
        pdf_bytes=b"%PDF-test",
        model="test-model",
    )

    assert captured["model"] == "test-model"
    assert captured["contents"][0].inline_data.mime_type == "application/pdf"
    assert captured["contents"][0].inline_data.data == b"%PDF-test"
    assert result.header.drawing_number == "13358002"
    assert result.drawing_block.scale == "5:2"
    assert result.lista_normas == ["ISO STANDARDS", "TSS 002513"]


def test_missing_standards_requires_english_reasoning(monkeypatch) -> None:
    captured = {}
    payload = {
        "normas_sugeridas": ["ASTM F104"],
        "reasoning": "ASTM F104 provides a material classification system for gasket sheets.",
        "confianca": 0.9,
    }

    class FakeModels:
        def generate_content(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                text=json.dumps(payload),
                usage_metadata=SimpleNamespace(
                    total_token_count=50,
                    prompt_token_count=35,
                    candidates_token_count=15,
                ),
            )

    monkeypatch.setattr(llm_models, "client", SimpleNamespace(models=FakeModels()))

    result, _ = llm_models.infer_missing_norms(
        classificacao="GASKET - VALVE PLATE",
        lista_normas_atuais=["TSS 002513"],
        system_prompt="Return JSON. Write every user-visible text value in English.",
        model="test-model",
    )

    prompt_text = captured["contents"][0].text
    assert "Component: GASKET - VALVE PLATE" in prompt_text
    assert "Current standards: TSS 002513" in prompt_text
    assert "Write the reasoning exclusively in English" in prompt_text
    assert result.reasoning.startswith("ASTM F104")
