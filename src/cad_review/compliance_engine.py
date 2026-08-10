"""Phase 8: aggregate part classification, standards and GD&T findings.

The engine is intentionally deterministic. It does not call an LLM and does
not reinterpret upstream results. Its job is to normalize provenance-rich
outputs from the existing part-classification branch and GD&T Phases 5-7 into
one stable CAD Review response.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

from src.cad_review.types import (
    CadReviewContext,
    CadReviewFinding,
    CadReviewIntegratedResult,
    CadReviewSummary,
)


def _dict(value: Any) -> dict:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    raise TypeError(f"cannot convert {type(value)!r} to dict")


def _severity(status: str) -> str:
    if status == "WARNING":
        return "WARNING"
    return "INFO"


def _append(
    findings: list[CadReviewFinding],
    *,
    domain: str,
    status: str,
    code: str,
    finding: str,
    recommended_action: Optional[str] = None,
    standard: Optional[str] = None,
    source_ref: Optional[str] = None,
    candidate_id: Optional[str] = None,
    datum: Optional[str] = None,
    normative_claim: bool = False,
    evidence: Optional[dict] = None,
) -> None:
    findings.append(
        CadReviewFinding(
            finding_id=f"F-{len(findings) + 1:03d}",
            domain=domain,
            status=status,
            severity=_severity(status),
            code=code,
            finding=finding,
            recommended_action=recommended_action,
            standard=standard,
            source_ref=source_ref,
            candidate_id=candidate_id,
            datum=datum,
            normative_claim=normative_claim,
            evidence=evidence or {},
        )
    )


def _standards_findings(comparison: Mapping[str, Any], findings: list[CadReviewFinding]) -> None:
    for standard in comparison.get("matching", []) or []:
        _append(
            findings,
            domain="standards",
            status="PASS",
            code="APPLICABLE_STANDARD_PRESENT",
            standard=str(standard),
            finding=f"Applicable standard {standard} is cited in the drawing.",
            evidence={"comparison_source": "applicable_vs_cited"},
        )

    for standard in comparison.get("missing", []) or []:
        _append(
            findings,
            domain="standards",
            status="WARNING",
            code="APPLICABLE_STANDARD_MISSING",
            standard=str(standard),
            finding=f"Applicable standard {standard} was not identified among the standards cited in the drawing.",
            recommended_action=f"Verify whether {standard} must be referenced/applied in the drawing.",
            evidence={"comparison_source": "applicable_vs_cited"},
        )

    for standard in comparison.get("unexpected", []) or []:
        _append(
            findings,
            domain="standards",
            status="NEEDS_CONTEXT",
            code="CITED_STANDARD_NOT_IN_APPLICABILITY_RESULT",
            standard=str(standard),
            finding=(
                f"Standard {standard} is cited in the drawing but was not returned by the current "
                "applicability context. Its presence is not treated as a violation automatically."
            ),
            recommended_action="Review the applicability context before classifying this reference as incorrect.",
            evidence={"comparison_source": "applicable_vs_cited"},
        )


def _phase6_findings(payload: Mapping[str, Any], findings: list[CadReviewFinding]) -> None:
    for frame in payload.get("frames", []) or []:
        finding = _dict(frame.get("finding"))
        if not finding:
            continue
        status = str(finding.get("status", "NOT_EVALUATED"))
        standard = finding.get("standard_label") or finding.get("standard")
        _append(
            findings,
            domain="iso1101",
            status=status,
            code=str(finding.get("code", "ISO1101_UNSPECIFIED")),
            finding=str(finding.get("finding", "ISO 1101 assessment produced no explanation.")),
            recommended_action=finding.get("recommended_action"),
            standard=str(standard) if standard else None,
            source_ref=finding.get("source_ref"),
            candidate_id=frame.get("candidate_id"),
            normative_claim=bool(finding.get("normative_claim", False)),
            evidence={
                "characteristic": finding.get("characteristic"),
                "referenced_datums": finding.get("referenced_datums", []),
                "datum_requirement": finding.get("datum_requirement"),
            },
        )


def _phase7_findings(payload: Mapping[str, Any], findings: list[CadReviewFinding]) -> None:
    for frame in payload.get("frames", []) or []:
        candidate_id = frame.get("candidate_id")
        for raw in frame.get("findings", []) or []:
            finding = _dict(raw)
            status = str(finding.get("status", "NOT_EVALUATED"))
            _append(
                findings,
                domain="iso5459",
                status=status,
                code=str(finding.get("code", "ISO5459_UNSPECIFIED")),
                finding=str(finding.get("finding", "ISO 5459 datum assessment produced no explanation.")),
                recommended_action=finding.get("recommended_action"),
                standard=finding.get("standard") or payload.get("standard"),
                source_ref=finding.get("source_ref") or payload.get("source_ref"),
                candidate_id=candidate_id,
                datum=finding.get("datum"),
                normative_claim=bool(finding.get("normative_claim", False)),
                evidence={
                    "definition_count": finding.get("definition_count"),
                    "definition_evidence": finding.get("definition_evidence", []),
                },
            )


def _summary(findings: Iterable[CadReviewFinding]) -> CadReviewSummary:
    counts = {"PASS": 0, "WARNING": 0, "NEEDS_CONTEXT": 0, "NOT_EVALUATED": 0}
    for finding in findings:
        counts[finding.status] += 1
    return CadReviewSummary(**counts)


def build_cad_review_result(
    *,
    drawing: Optional[Mapping[str, Any]] = None,
    part_branch: Optional[Mapping[str, Any]] = None,
    phase5_payload: Optional[Mapping[str, Any]] = None,
    phase6_payload: Optional[Mapping[str, Any]] = None,
    phase7_payload: Optional[Mapping[str, Any]] = None,
    compressor_series: str = "ALL",
    compressor_series_source: str = "temporary_default_until_windchill",
) -> CadReviewIntegratedResult:
    """Build one CAD Review result without re-running or reinterpreting upstream models."""

    part = dict(part_branch or {})
    phase5 = dict(phase5_payload or {})
    phase6 = dict(phase6_payload or {})
    phase7 = dict(phase7_payload or {})

    findings: list[CadReviewFinding] = []
    comparison = _dict(part.get("standards_comparison"))
    if comparison:
        _standards_findings(comparison, findings)
    if phase6:
        _phase6_findings(phase6, findings)
    if phase7:
        _phase7_findings(phase7, findings)

    gdt_frames = []
    for row in phase5.get("frames", []) or []:
        parsed = row.get("parsed") or {}
        gdt_frames.append(
            {
                "candidate_id": row.get("candidate_id"),
                "characteristic": parsed.get("characteristic"),
                "tolerance_raw": parsed.get("tolerance_raw"),
                "tolerance_value": parsed.get("tolerance_value"),
                "referenced_datums": parsed.get("referenced_datums", []),
                "unresolved_fields": parsed.get("unresolved_fields", []),
                "field_sources": parsed.get("field_sources", {}),
            }
        )

    result = CadReviewIntegratedResult(
        drawing=dict(drawing or {}),
        review_context=CadReviewContext(
            compressor_series=str(compressor_series),
            compressor_series_source=str(compressor_series_source),
        ),
        part_classification=_dict(part.get("classification")),
        cited_standards=[_dict(row) for row in part.get("cited_standards", []) or []],
        applicable_standards=[_dict(row) for row in part.get("applicable_standards", []) or []],
        standards_comparison=comparison,
        gdt_frames=gdt_frames,
        datum_definitions=[_dict(row) for row in phase7.get("definitions", []) or []],
        findings=findings,
        summary=_summary(findings),
        provenance={
            "part_classification": part.get("provenance", {}),
            "phase5": {
                "phase": phase5.get("phase"),
                "validation_status": phase5.get("validation_status"),
            },
            "phase6": {
                "phase": phase6.get("phase"),
                "validation_status": phase6.get("validation_status"),
                "mode": phase6.get("mode"),
                "normative_applicability_established": phase6.get("normative_applicability_established"),
            },
            "phase7": {
                "phase": phase7.get("phase"),
                "validation_status": phase7.get("validation_status"),
                "mode": phase7.get("mode"),
                "normative_applicability_established": phase7.get("normative_applicability_established"),
            },
        },
    )
    return result


__all__ = ["build_cad_review_result"]
