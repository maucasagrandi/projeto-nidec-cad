import fitz
from PIL import Image

from src.cad_review.visual_output import render_visual_evidence


def _synthetic_pdf() -> bytes:
    doc = fitz.open()
    page = doc.new_page(width=300, height=200)
    page.draw_rect(fitz.Rect(30, 40, 120, 65), color=(0, 0, 0), width=1)
    page.draw_rect(fitz.Rect(200, 90, 220, 110), color=(0, 0, 0), width=1)
    page.insert_text((205, 105), "A", fontsize=9)
    data = doc.tobytes()
    doc.close()
    return data


def test_visual_evidence_renders_page_and_crops(tmp_path):
    result = render_visual_evidence(
        _synthetic_pdf(),
        output_dir=tmp_path,
        gdt_candidates=[
            {
                "candidate_id": "GDT-CAND-P01-001",
                "page": 1,
                "frame_bbox": [30, 40, 120, 65],
                "characteristic": "position",
                "referenced_datums": ["A"],
            }
        ],
        datum_definitions=[
            {
                "label": "A",
                "page": 1,
                "box_bbox": [200, 90, 220, 110],
            }
        ],
        findings=[
            {
                "candidate_id": "GDT-CAND-P01-001",
                "datum": "A",
                "status": "PASS",
            }
        ],
        dpi=144,
    )

    assert result["label_policy"] == "GDT-CAND until independently validated"
    assert result["pages"][0]["gdt_candidate_count"] == 1
    assert result["pages"][0]["datum_definition_count"] == 1
    page_path = tmp_path / result["pages"][0]["annotated_image"]
    assert page_path.exists()
    with Image.open(page_path) as image:
        assert image.width == 600
        assert image.height == 400
    assert any(path.endswith("_frame.png") for path in result["crops"])
    assert any("DATUM-A" in path for path in result["crops"])
    assert all((tmp_path / path).exists() for path in result["crops"])
