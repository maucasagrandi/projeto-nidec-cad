"""Deterministic assessment of GD&T tolerance-cell evidence.

Phase 5 needs to distinguish two very different situations:

1. a tolerance value was actually recovered (for example from a textual PDF
   token); and
2. the visual cell contains only frame / leader geometry and no plausible text
   glyph from which a numeric tolerance can be read.

This module deliberately does **not** perform OCR, digit classification or ISO
validation.  It only converts the evidence already produced by the visual-cell
filter into an explicit status.  Missing evidence stays unresolved instead of
being guessed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional


STATUS_RESOLVED_NUMERIC = "resolved_numeric"
STATUS_UNRESOLVED_NO_TEXT = "unresolved_no_text_candidate"
STATUS_UNRESOLVED_UNRECOGNIZED = "unresolved_unrecognized_text_candidate"
STATUS_INVALID_EVIDENCE = "invalid_evidence"


@dataclass(frozen=True)
class ToleranceCellAssessment:
    """Evidence state for a single GD&T tolerance cell."""

    cell_index: int
    status: str
    tolerance_raw: Optional[str] = None
    tolerance_value: Optional[float] = None
    diameter_zone: Optional[bool] = None
    selected_text_candidate_count: int = 0
    structural_count: int = 0
    arrow_like_count: int = 0
    other_count: int = 0
    reason: str = ""
    source: str = "phase5_cell_content_filter_diagnostic"

    @property
    def resolved(self) -> bool:
        return self.status == STATUS_RESOLVED_NUMERIC and self.tolerance_value is not None

    def to_dict(self) -> dict:
        return {
            "cell_index": self.cell_index,
            "status": self.status,
            "resolved": self.resolved,
            "tolerance_raw": self.tolerance_raw,
            "tolerance_value": self.tolerance_value,
            "diameter_zone": self.diameter_zone,
            "selected_text_candidate_count": self.selected_text_candidate_count,
            "structural_count": self.structural_count,
            "arrow_like_count": self.arrow_like_count,
            "other_count": self.other_count,
            "reason": self.reason,
            "source": self.source,
        }


def assessment_from_filter_row(row: Mapping[str, Any]) -> ToleranceCellAssessment:
    """Interpret one row from ``phase5_cell_content_filter_diagnostic``.

    The visual filter currently isolates plausible glyph components but does
    not classify digits. Therefore:

    - zero selected text candidates -> the numeric value is unresolved because
      no plausible text glyph exists inside the logical tolerance cell;
    - one or more selected text candidates -> still unresolved until a numeric
      recognizer explicitly decodes them;
    - a row that is not cell[1] or is not marked as ``tolerance`` is invalid
      evidence for this function.
    """

    try:
        cell_index = int(row.get("cell_index", -1))
    except (TypeError, ValueError):
        cell_index = -1

    role = str(row.get("expected_role", ""))
    selected = int(row.get("text_candidate_count", 0) or 0)
    structural = int(row.get("structural_count", 0) or 0)
    arrows = int(row.get("arrow_like_count", 0) or 0)
    other = int(row.get("other_count", 0) or 0)

    if cell_index != 1 or role != "tolerance":
        return ToleranceCellAssessment(
            cell_index=cell_index,
            status=STATUS_INVALID_EVIDENCE,
            selected_text_candidate_count=selected,
            structural_count=structural,
            arrow_like_count=arrows,
            other_count=other,
            reason="row is not the logical tolerance cell[1]",
        )

    if selected <= 0:
        geometry_parts = []
        if structural:
            geometry_parts.append(f"structural={structural}")
        if arrows:
            geometry_parts.append(f"arrows={arrows}")
        if other:
            geometry_parts.append(f"other={other}")
        suffix = ", ".join(geometry_parts) or "no retained visual components"
        return ToleranceCellAssessment(
            cell_index=cell_index,
            status=STATUS_UNRESOLVED_NO_TEXT,
            selected_text_candidate_count=0,
            structural_count=structural,
            arrow_like_count=arrows,
            other_count=other,
            reason=f"no selected text candidate in tolerance cell; {suffix}",
        )

    return ToleranceCellAssessment(
        cell_index=cell_index,
        status=STATUS_UNRESOLVED_UNRECOGNIZED,
        selected_text_candidate_count=selected,
        structural_count=structural,
        arrow_like_count=arrows,
        other_count=other,
        reason=(
            f"{selected} plausible text candidate(s) isolated, but no deterministic "
            "numeric recognizer has decoded them"
        ),
    )


def resolved_numeric_assessment(
    *,
    raw: str,
    value: float,
    diameter_zone: Optional[bool] = None,
    source: str = "text_parser",
) -> ToleranceCellAssessment:
    """Create explicit resolved evidence when a numeric value is known."""

    return ToleranceCellAssessment(
        cell_index=1,
        status=STATUS_RESOLVED_NUMERIC,
        tolerance_raw=str(raw),
        tolerance_value=float(value),
        diameter_zone=diameter_zone,
        reason="numeric tolerance recovered deterministically",
        source=source,
    )


__all__ = [
    "STATUS_INVALID_EVIDENCE",
    "STATUS_RESOLVED_NUMERIC",
    "STATUS_UNRESOLVED_NO_TEXT",
    "STATUS_UNRESOLVED_UNRECOGNIZED",
    "ToleranceCellAssessment",
    "assessment_from_filter_row",
    "resolved_numeric_assessment",
]
