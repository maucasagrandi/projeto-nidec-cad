from src.cad_review.orchestrator import run_part_classification_branch


class FakeApplicabilityEngine:
    def __init__(self):
        self.calls = []

    def get_applicable_standards(self, *, component, compressor_series, material_family):
        self.calls.append(
            {
                "component": component,
                "compressor_series": compressor_series,
                "material_family": material_family,
            }
        )
        return {
            "component": component,
            "compressor_series": compressor_series,
            "material_family": material_family,
            "applicable_standards": [
                {
                    "standard": "TSS 002513",
                    "reason": "General / All",
                    "source": "component_match",
                },
                {
                    "standard": "TSS 002420",
                    "reason": "Material / All",
                    "source": "component_match",
                },
            ],
            "unresolved_fields": [],
        }


def fake_classifier(_text, _prompt):
    return {
        "document_type": {"value": "product_drawing", "evidence": "TITLE", "confidence": 0.9},
        "component": {
            "value": "GASKET - VALVE PLATE",
            "evidence": "GASKET - VALVE PLATE",
            "confidence": 0.99,
        },
        "material_family": {
            "value": None,
            "evidence": None,
            "confidence": 0.0,
        },
        "compressor_series": {
            "value": None,
            "evidence": None,
            "confidence": 0.0,
        },
        "cited_standards": [
            {
                "standard": "TSS002513",
                "evidence": "1 - DRAWING ACCORDING TO ISO STANDARDS, SEE TSS002513.",
            },
            {
                "standard": "TSS 002420",
                "evidence": "3 - REQUIREMENTS ... ACCORDING TO TSS 002420.",
            },
        ],
    }


def test_all_series_is_external_context_and_does_not_rewrite_llm_classification():
    engine = FakeApplicabilityEngine()
    result = run_part_classification_branch(
        b"unused",
        classification_prompt="unused",
        extracted_text="vector text",
        classifier_fn=fake_classifier,
        applicability_engine=engine,
    )

    assert result["classification"]["compressor_series"]["value"] is None
    assert result["review_context"] == {
        "compressor_series": "ALL",
        "compressor_series_source": "temporary_default_until_windchill",
    }
    assert engine.calls[0]["compressor_series"] == "ALL"
    assert engine.calls[0]["component"] == "GASKET - VALVE PLATE"


def test_cited_norms_are_normalized_before_deterministic_comparison():
    engine = FakeApplicabilityEngine()
    result = run_part_classification_branch(
        b"unused",
        classification_prompt="unused",
        extracted_text="vector text",
        classifier_fn=fake_classifier,
        applicability_engine=engine,
    )

    assert [row["standard"] for row in result["cited_standards"]] == [
        "TSS 002513",
        "TSS 002420",
    ]
    assert result["standards_comparison"]["matching"] == ["TSS 002420", "TSS 002513"]
    assert result["standards_comparison"]["missing"] == []
    assert result["standards_comparison"]["applicability_status"] == "RESOLVED"
