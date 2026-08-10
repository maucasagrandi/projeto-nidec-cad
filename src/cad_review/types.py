"""Typed contracts for the integrated CAD Review result.

These contracts deliberately preserve provenance. In particular, compressor
series may come from Windchill/task context and must never be rewritten as if
it had been extracted from the CAD by the LLM.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


ReviewStatus = Literal["PASS", "WARNING", "NEEDS_CONTEXT", "NOT_EVALUATED"]
ReviewSeverity = Literal["INFO", "WARNING", "ERROR"]


class CadReviewContext(BaseModel):
    compressor_series: str = "ALL"
    compressor_series_source: str = "temporary_default_until_windchill"
    note: str = (
        "Temporary Phase 8 context: compressor series is not extracted from the CAD; "
        "ALL is used until Windchill provides the real series."
    )


class CadReviewFinding(BaseModel):
    finding_id: str
    domain: Literal["standards", "iso1101", "iso5459", "pipeline"]
    status: ReviewStatus
    severity: ReviewSeverity
    code: str
    finding: str
    recommended_action: Optional[str] = None
    standard: Optional[str] = None
    source_ref: Optional[str] = None
    candidate_id: Optional[str] = None
    datum: Optional[str] = None
    normative_claim: bool = False
    evidence: Dict[str, Any] = Field(default_factory=dict)


class CadReviewSummary(BaseModel):
    PASS: int = 0
    WARNING: int = 0
    NEEDS_CONTEXT: int = 0
    NOT_EVALUATED: int = 0


class CadReviewIntegratedResult(BaseModel):
    schema_version: int = 1
    phase: str = "phase8_integrated_cad_review"
    drawing: Dict[str, Any] = Field(default_factory=dict)
    review_context: CadReviewContext = Field(default_factory=CadReviewContext)
    part_classification: Dict[str, Any] = Field(default_factory=dict)
    cited_standards: List[Dict[str, Any]] = Field(default_factory=list)
    applicable_standards: List[Dict[str, Any]] = Field(default_factory=list)
    standards_comparison: Dict[str, Any] = Field(default_factory=dict)
    gdt_frames: List[Dict[str, Any]] = Field(default_factory=list)
    datum_definitions: List[Dict[str, Any]] = Field(default_factory=list)
    findings: List[CadReviewFinding] = Field(default_factory=list)
    summary: CadReviewSummary = Field(default_factory=CadReviewSummary)
    provenance: Dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "CadReviewContext",
    "CadReviewFinding",
    "CadReviewSummary",
    "CadReviewIntegratedResult",
]
