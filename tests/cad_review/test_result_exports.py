import zipfile
import xml.etree.ElementTree as ET

from src.cad_review.result_exports import (
    ENGINEERING_HEADERS,
    TECHNICAL_HEADERS,
    engineering_row,
    technical_row,
    write_batch_workbooks,
)


def _sample_result():
    return {
        "drawing": {"name": "sample.pdf"},
        "review_context": {
            "compressor_series": "ALL",
            "compressor_series_source": "temporary_default_until_windchill",
        },
        "part_classification": {
            "component": {"value": "CRANKSHAFT", "confidence": 0.98},
            "compressor_series": {"value": None, "confidence": 0.0},
        },
        "cited_standards": [
            {"standard": "TSS 002513", "source_text": "NOTE 1 - SEE TSS 002513"},
        ],
        "applicable_standards": [
            {"standard": "TSS 002513", "source": "component_match"},
            {"standard": "TSS 002420", "source": "component_match"},
        ],
        "standards_comparison": {
            "matching": ["TSS 002513"],
            "missing": ["TSS 002420"],
            "unexpected": [],
        },
        "gdt_frames": [
            {
                "candidate_id": "GDT-CAND-P01-001",
                "characteristic": "position",
                "referenced_datums": ["A", "B"],
            }
        ],
        "datum_definitions": [{"label": "A"}, {"label": "B"}],
        "findings": [
            {"domain": "iso1101", "status": "NEEDS_CONTEXT"},
            {"domain": "iso5459", "status": "PASS"},
        ],
        "summary": {"PASS": 2, "WARNING": 1, "NEEDS_CONTEXT": 1, "NOT_EVALUATED": 0},
        "artifacts": {
            "visual_evidence": {"pages": [{"page": 1, "annotated_image": "page_001_annotated.png"}]}
        },
        "validation_status": "BATCH_VALIDATION_ONLY",
        "provenance": {
            "part_classification": {
                "llm_metadata": {
                    "prompt_tokens": 100,
                    "completion_tokens": 25,
                    "latency_ms": 500.0,
                }
            }
        },
    }


def test_engineering_layout_matches_existing_validation_sheet_and_uses_deterministic_missing_standards():
    row = engineering_row(_sample_result())
    assert ENGINEERING_HEADERS == [
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
    assert row["Classificação_Ground_Truth"] == ""
    assert row["Match_Classif"] == ""
    assert row["Normas_Sugeridas_LLM"] == "TSS 002420"
    assert "free-form" in row["Observações"]


def test_technical_row_contains_artifact_and_gdt_counts():
    row = technical_row(_sample_result(), result_json_path="sample/result.json")
    assert row["GDT_Candidate_Count"] == 1
    assert row["GDT_Classified_Count"] == 1
    assert row["Datum_Reference_Count"] == 2
    assert row["Datum_Definition_Count"] == 2
    assert row["ISO1101_NEEDS_CONTEXT"] == 1
    assert row["ISO5459_PASS"] == 1
    assert row["Annotated_Images"] == "page_001_annotated.png"
    assert row["Result_JSON"] == "sample/result.json"


def test_generated_xlsx_is_valid_openxml_zip(tmp_path):
    result = _sample_result()
    paths = write_batch_workbooks(
        tmp_path,
        engineering_rows=[engineering_row(result)],
        technical_rows=[technical_row(result, result_json_path="sample/result.json")],
    )
    for name in paths.values():
        path = tmp_path / name
        assert path.exists()
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
            assert "xl/workbook.xml" in names
            assert "xl/worksheets/sheet1.xml" in names
            assert "xl/styles.xml" in names
            ET.fromstring(zf.read("[Content_Types].xml"))
            ET.fromstring(zf.read("xl/workbook.xml"))
            sheet = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
            assert sheet is not None


def test_headers_have_expected_column_counts():
    assert len(ENGINEERING_HEADERS) == 15
    assert len(TECHNICAL_HEADERS) == 28
