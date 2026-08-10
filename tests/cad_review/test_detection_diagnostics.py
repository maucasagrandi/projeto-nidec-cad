import csv

import fitz
from PIL import Image

from src.cad_review.detection_diagnostics import render_detection_diagnostics


def _synthetic_pdf() -> bytes:
    doc = fitz.open()
    page = doc.new_page(width=320, height=220)
    page.draw_rect(fitz.Rect(30, 50, 125, 75), color=(0, 0, 0), width=1)
    page.draw_line(fitz.Point(55, 50), fitz.Point(55, 75), color=(0, 0, 0), width=1)
    page.draw_line(fitz.Point(95, 50), fitz.Point(95, 75), color=(0, 0, 0), width=1)
    data = doc.tobytes()
    doc.close()
    return data


def _candidate():
    return {
        "candidate_id": "GDT-CAND-P01-001",
        "page": 1,
        "frame_bbox": [30, 50, 125, 75],
        "symbol_bbox": [30, 50, 55, 75],
        "cell_bboxes": [[30, 50, 55, 75], [55, 50, 95, 75], [95, 50, 125, 75]],
        "detection_status": "candidate_unvalidated",
        "referenced_datums": [],
        "unresolved_fields": [],
        "symbol_scoring": {
            "class_scores": {
                "parallelism": 0.81,
                "straightness": 0.62,
                "position": 0.31,
            },
            "best_class": "parallelism",
            "best_score": 0.81,
            "second_best_class": "straightness",
            "second_best_score": 0.62,
            "margin": 0.19,
            "decision_policy": "ranking_only_no_global_threshold",
            "catalog_complete": True,
        },
    }


def test_detection_diagnostics_separates_candidate_and_classifier_artifacts(tmp_path):
    result = render_detection_diagnostics(
        _synthetic_pdf(),
        output_dir=tmp_path,
        gdt_candidates=[_candidate()],
        dpi=144,
        top_k=3,
    )

    assert result["ground_truth_used"] is False
    assert result["candidate_semantics"] == "unvalidated detector proposals"
    assert result["pages"][0]["candidate_count"] == 1

    candidates_path = tmp_path / result["pages"][0]["candidates_image"]
    contact_path = tmp_path / result["pages"][0]["symbol_contact_sheet"]
    csv_path = tmp_path / result["candidate_csv"]

    assert candidates_path.exists()
    assert contact_path.exists()
    assert csv_path.exists()

    with Image.open(candidates_path) as image:
        assert image.width == 640
        assert image.height == 440

    with Image.open(contact_path) as image:
        assert image.width > 0
        assert image.height > 0

    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    assert rows[0]["candidate_id"] == "GDT-CAND-P01-001"
    assert rows[0]["top1_class"] == "parallelism"
    assert rows[0]["top2_class"] == "straightness"
    assert rows[0]["human_is_real_gdt"] == ""
    assert rows[0]["human_true_characteristic"] == ""
    assert rows[0]["human_notes"] == ""
