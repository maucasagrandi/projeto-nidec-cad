from __future__ import annotations

import fitz

from src.cad_review.detection_diagnostics_v2 import render_v2_detection_diagnostics


def test_v2_diagnostics_write_raw_accepted_and_rejected_images(tmp_path):
    doc = fitz.open()
    doc.new_page(width=240, height=160)
    pdf_bytes = doc.tobytes()
    doc.close()

    raw = [
        {
            "proposal_id": "GDT-V2-P01-001",
            "page": 1,
            "frame_bbox": [20, 20, 70, 32],
            "cell_bboxes": [[20, 20, 32, 32], [32, 20, 55, 32], [55, 20, 70, 32]],
            "symbol_bbox": [20, 20, 32, 32],
            "sources": ["vector_normalized_v2"],
            "validation_status": "accepted_for_downstream",
            "rejection_reasons": [],
        },
        {
            "proposal_id": "GDT-V2-P01-002",
            "page": 1,
            "frame_bbox": [20, 70, 90, 84],
            "cell_bboxes": [[20, 70, 40, 84], [40, 70, 65, 84], [65, 70, 90, 84]],
            "symbol_bbox": [20, 70, 40, 84],
            "sources": ["raster_morphology_v2"],
            "validation_status": "rejected_by_validator",
            "rejection_reasons": ["FIRST_CELL_NOT_GDT_LIKE"],
        },
    ]

    result = render_v2_detection_diagnostics(
        pdf_bytes,
        output_dir=tmp_path,
        page_results=[
            {
                "page": 1,
                "primitive_audit": {"primitive_counts": {"l": 8}},
                "raw_proposals": raw,
                "accepted_candidate_ids": ["GDT-V2-P01-001"],
                "rejected_proposal_ids": ["GDT-V2-P01-002"],
            }
        ],
        dpi=100,
    )

    page = result["pages"][0]
    assert (tmp_path / page["raw_proposals_image"]).exists()
    assert (tmp_path / page["accepted_image"]).exists()
    assert (tmp_path / page["rejected_image"]).exists()
    assert page["primitive_audit"]["primitive_counts"]["l"] == 8
