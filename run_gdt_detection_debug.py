"""Run isolated GD&T detector diagnostics for one CAD PDF.

Default mode is Candidate Detector V2.1 in diagnostic/shadow mode. It does NOT
call Gemini, Normas.xlsx, ISO rules, datum consistency, or the Compliance Engine.

V2.1 keeps V2 proposal generation, then improves the rectangle/cell-chain
geometry before symbol ranking. Use ``--detector-version v2`` to reproduce the
unrefined hybrid proposal detector or ``--detector-version v1`` for legacy Phase 1.
Nothing is called TP/FP without independent ground truth.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import fitz

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cad_review.detection_diagnostics import render_detection_diagnostics
from src.cad_review.detection_diagnostics_v2 import render_v2_detection_diagnostics
from src.gdt.candidate_detector_v2 import GdtCandidateDetectorV2, validate_proposals
from src.gdt.candidate_detector_v21 import GdtCandidateDetectorV21
from src.gdt.detector import GdtFrameDetector
from src.gdt.symbol_classifier import load_template_catalog, render_page_gray, score_candidates


def _safe_stem(path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("._")
    return stem or "cad"


def _bbox(value: Any) -> list[float]:
    return [round(float(v), 3) for v in value.to_list()]


def _sync_phase4_templates() -> None:
    script = PROJECT_ROOT / "validation" / "gdt" / "scripts" / "sync_phase4_templates.py"
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.stdout.strip():
        print(completed.stdout.strip())
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"Phase 4 template sync failed (code={completed.returncode}): {detail}")


def _score_payload(score: Any, *, catalog_complete: Any = None) -> dict:
    if score is None:
        return {
            "best_class": None,
            "decision_policy": "not_evaluated",
            "reason": "no score",
        }
    return {
        **score.to_dict(),
        "decision_policy": "ranking_only_no_global_threshold",
        "catalog_complete": catalog_complete,
    }


def _candidate_row(candidate: Any, score: Any = None, *, sources: list[str] | None = None, validator: dict | None = None) -> dict:
    return {
        "candidate_id": candidate.candidate_id,
        "page": candidate.page,
        "frame_bbox": _bbox(candidate.frame_bbox),
        "symbol_bbox": _bbox(candidate.symbol_bbox),
        "cell_bboxes": [_bbox(cell.bbox) for cell in candidate.cells],
        "cell_texts": [list(cell.texts) for cell in candidate.cells],
        "detection_status": "candidate_unvalidated",
        "referenced_datums": [],
        "unresolved_fields": [],
        "sources": list(sources or []),
        "validator": dict(validator or {}),
        "symbol_scoring": _score_payload(score),
    }


def _run_v1(pdf_bytes: bytes, page_count: int, templates: list, symbol_dpi: int) -> tuple[list[dict], list[dict]]:
    detector = GdtFrameDetector()
    rows: list[dict] = []
    pages: list[dict] = []
    for page_index in range(page_count):
        candidates = detector.detect_frames(pdf_bytes, page_index=page_index)
        scores = []
        if templates and candidates:
            page_gray, zoom = render_page_gray(pdf_bytes, page_index=page_index, dpi=symbol_dpi)
            scores = [score for score, _crop in score_candidates(candidates, page_gray, zoom, templates)]
        else:
            scores = [None] * len(candidates)
        rows.extend(_candidate_row(c, s, sources=["v1_legacy"]) for c, s in zip(candidates, scores))
        pages.append({
            "page": page_index + 1,
            "primitive_audit": {},
            "raw_proposals": [],
            "accepted_candidate_ids": [c.candidate_id for c in candidates],
            "rejected_proposal_ids": [],
        })
    return rows, pages


def _run_hybrid(
    pdf_bytes: bytes,
    page_count: int,
    templates: list,
    *,
    symbol_dpi: int,
    detector_only: bool,
    detector_version: str,
) -> tuple[list[dict], list[dict], list[dict]]:
    detector = GdtCandidateDetectorV21() if detector_version == "v21" else GdtCandidateDetectorV2()
    accepted_rows: list[dict] = []
    raw_rows: list[dict] = []
    page_results: list[dict] = []

    for page_index in range(page_count):
        proposals, audit = detector.propose(pdf_bytes, page_index=page_index)
        materialized = detector.materialize(pdf_bytes, proposals, page_index=page_index, dpi=symbol_dpi)

        scores: list[Any] = [None] * len(materialized)
        if templates and materialized and not detector_only:
            page_gray, zoom = render_page_gray(pdf_bytes, page_index=page_index, dpi=symbol_dpi)
            scores = [score for score, _crop in score_candidates(materialized, page_gray, zoom, templates)]

        if detector_only:
            accepted = []
            rejected = []
        else:
            accepted, rejected = validate_proposals(proposals, materialized, scores)

        score_by_id = {c.candidate_id: s for c, s in zip(materialized, scores)}
        candidate_by_id = {c.candidate_id: c for c in materialized}
        proposal_by_id = {p.proposal_id: p for p in proposals}

        for proposal in proposals:
            candidate = candidate_by_id[proposal.proposal_id]
            row = _candidate_row(
                candidate,
                score_by_id.get(proposal.proposal_id),
                sources=proposal.sources,
                validator={
                    "validation_status": proposal.validation_status,
                    "rejection_reasons": list(proposal.rejection_reasons),
                    "evidence": dict(proposal.validator_evidence),
                    "rectangle_geometry": dict(proposal.primitive_evidence.get("rectangle_geometry") or {}),
                },
            )
            row["proposal_id"] = proposal.proposal_id
            raw_rows.append(row)

        for candidate in accepted:
            proposal = proposal_by_id[candidate.candidate_id]
            accepted_rows.append(
                _candidate_row(
                    candidate,
                    score_by_id.get(candidate.candidate_id),
                    sources=proposal.sources,
                    validator={
                        "validation_status": proposal.validation_status,
                        "rejection_reasons": [],
                        "evidence": dict(proposal.validator_evidence),
                        "rectangle_geometry": dict(proposal.primitive_evidence.get("rectangle_geometry") or {}),
                    },
                )
            )

        page_results.append({
            "page": page_index + 1,
            "primitive_audit": audit,
            "raw_proposals": [p.to_dict() for p in proposals],
            "accepted_candidate_ids": [c.candidate_id for c in accepted],
            "rejected_proposal_ids": [p.proposal_id for p in rejected],
        })

    return accepted_rows, raw_rows, page_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolated GD&T detection diagnostics")
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--output-folder", type=Path, default=PROJECT_ROOT / "DEBUG_RESULTS")
    parser.add_argument("--templates", type=Path, default=PROJECT_ROOT / "assets" / "gdt" / "templates")
    parser.add_argument("--visual-dpi", type=int, default=180)
    parser.add_argument("--symbol-dpi", type=int, default=300)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--skip-template-sync", action="store_true")
    parser.add_argument("--detector-version", choices=("v1", "v2", "v21"), default="v21")
    parser.add_argument(
        "--detector-only",
        action="store_true",
        help="Generate geometry proposals only. Structural symbol/content validation is skipped.",
    )
    args = parser.parse_args()

    pdf = args.pdf.resolve()
    if not pdf.exists() or not pdf.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf}")
    pdf_bytes = pdf.read_bytes()

    if not args.detector_only and not args.skip_template_sync:
        print("template_sync=START")
        _sync_phase4_templates()
        print("template_sync=PASS")

    templates = []
    template_classes: list[str] = []
    classification_error = None
    if not args.detector_only:
        try:
            templates = load_template_catalog(args.templates)
            template_classes = sorted({row.class_name for row in templates})
        except Exception as exc:
            classification_error = f"{type(exc).__name__}: {exc}"
            print(f"symbol_ranking=DISABLED reason={classification_error}")

    output = args.output_folder.resolve() / _safe_stem(pdf)
    output.mkdir(parents=True, exist_ok=True)

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        page_count = len(doc)

    if args.detector_version == "v1":
        rows, page_results = _run_v1(pdf_bytes, page_count, templates, args.symbol_dpi)
        raw_rows = rows
    else:
        rows, raw_rows, page_results = _run_hybrid(
            pdf_bytes,
            page_count,
            templates,
            symbol_dpi=args.symbol_dpi,
            detector_only=args.detector_only,
            detector_version=args.detector_version,
        )

    diagnostics = render_detection_diagnostics(
        pdf_bytes,
        output_dir=output,
        gdt_candidates=raw_rows,
        dpi=args.visual_dpi,
        top_k=args.top_k,
    )

    v2_visual = None
    if args.detector_version != "v1":
        v2_visual = render_v2_detection_diagnostics(
            pdf_bytes,
            output_dir=output,
            page_results=page_results,
            dpi=args.visual_dpi,
        )

    payload = {
        "phase": "gdt_detection_debug",
        "detector_version": args.detector_version,
        "validation_status": "DIAGNOSTIC_ONLY",
        "production_claim": False,
        "drawing": {"name": pdf.name, "source_path": str(pdf)},
        "page_count": page_count,
        "raw_proposal_count": len(raw_rows),
        "accepted_candidate_count": len(rows) if args.detector_version != "v1" else len(raw_rows),
        "ground_truth_used": False,
        "candidate_semantics": "unvalidated detector proposals",
        "template_classes": template_classes,
        "symbol_ranking_enabled": bool(templates),
        "symbol_ranking_error": classification_error,
        "accepted_candidates": rows,
        "raw_proposals_materialized": raw_rows,
        "v2_pages": page_results if args.detector_version != "v1" else [],
        "artifacts": {
            "candidate_contact_diagnostics": diagnostics,
            "v2_stage_diagnostics": v2_visual,
        },
        "next_validation_step": (
            "Annotate real FCF ground truth independently, then match GT against geometry-refined proposals and accepted candidates. "
            "Measure rectangle proposal recall first, validator precision/recall second, and symbol accuracy only on matched real FCFs."
        ),
    }
    result_path = output / "debug_result.json"
    result_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("phase=gdt_detection_debug")
    print(f"detector_version={args.detector_version}")
    print(f"pdf={pdf}")
    print(f"pages={page_count}")
    print(f"raw_proposals={len(raw_rows)}")
    print(f"accepted_candidates={payload['accepted_candidate_count']}")
    print(f"symbol_ranking_enabled={bool(templates)}")
    print(f"output={output}")
    print(f"result={result_path}")
    if args.detector_version != "v1":
        for page in page_results:
            audit = page.get("primitive_audit") or {}
            print(
                f"page={page['page']} v1={audit.get('v1_proposals', 0)} "
                f"vector={audit.get('normalized_vector_proposals', 0)} "
                f"raster={audit.get('raster_proposals', 0)} "
                f"dedup={audit.get('combined_after_dedup', 0)} "
                f"rectangle_in={audit.get('v21_pre_rectangle_refinement', audit.get('combined_after_dedup', 0))} "
                f"rectangle_kept={audit.get('v21_post_rectangle_refinement', audit.get('combined_after_dedup', 0))} "
                f"rectangle_rejected={audit.get('v21_geometry_rejected', 0)} "
                f"accepted={len(page.get('accepted_candidate_ids') or [])} "
                f"rejected={len(page.get('rejected_proposal_ids') or [])}"
            )


if __name__ == "__main__":
    main()
