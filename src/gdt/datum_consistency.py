"""Phase 7 consistency checks between referenced and defined datum labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from src.gdt.datum_feature import DatumFeatureIndicatorCandidate

STATUS_PASS = "PASS"
STATUS_WARNING = "WARNING"
MODE_REFERENCE = "reference"
MODE_NORMATIVE = "normative"


@dataclass(frozen=True)
class DatumDefinitionFinding:
    datum: str
    status: str
    code: str
    standard: str
    mode: str
    finding: str
    recommended_action: str
    normative_claim: bool
    definition_count: int
    definition_evidence: tuple[dict, ...]
    source_ref: str

    def to_dict(self) -> dict:
        return {
            "datum": self.datum,
            "status": self.status,
            "code": self.code,
            "standard": self.standard,
            "mode": self.mode,
            "finding": self.finding,
            "recommended_action": self.recommended_action,
            "normative_claim": self.normative_claim,
            "definition_count": self.definition_count,
            "definition_evidence": list(self.definition_evidence),
            "source_ref": self.source_ref,
        }


def _indicator_dict(value: DatumFeatureIndicatorCandidate | Mapping[str, object]) -> dict:
    if isinstance(value, DatumFeatureIndicatorCandidate):
        return value.to_dict()
    return dict(value)


def assess_referenced_datum_definitions(
    *,
    referenced_datums: Sequence[str],
    defined_indicators: Iterable[DatumFeatureIndicatorCandidate | Mapping[str, object]],
    mode: str = MODE_REFERENCE,
    standard: str = "ISO 5459",
    source_ref: str = "Datum related symbols table: Datum Feature Indicator -> ISO 5459",
) -> list[DatumDefinitionFinding]:
    """Check whether each referenced datum has a detected datum feature indicator."""

    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in {MODE_REFERENCE, MODE_NORMATIVE}:
        raise ValueError(f"unsupported mode: {mode}")

    definitions: dict[str, list[dict]] = {}
    for raw in defined_indicators:
        row = _indicator_dict(raw)
        label = str(row.get("label", "")).strip().upper()
        if len(label) == 1 and label.isalpha():
            definitions.setdefault(label, []).append(row)

    findings: list[DatumDefinitionFinding] = []
    seen: set[str] = set()
    for raw in referenced_datums:
        datum = str(raw).strip().upper()
        if len(datum) != 1 or not datum.isalpha() or datum in seen:
            continue
        seen.add(datum)
        evidence = tuple(definitions.get(datum, []))
        if evidence:
            findings.append(
                DatumDefinitionFinding(
                    datum=datum,
                    status=STATUS_PASS,
                    code="ISO5459_DATUM_DEFINITION_FOUND",
                    standard=standard,
                    mode=normalized_mode,
                    finding=f"Datum {datum} is referenced and a corresponding datum feature indicator was identified in the drawing.",
                    recommended_action="No action required for datum-definition presence.",
                    normative_claim=False,
                    definition_count=len(evidence),
                    definition_evidence=evidence,
                    source_ref=source_ref,
                )
            )
            continue

        prefix = "Violation" if normalized_mode == MODE_NORMATIVE else "Potential violation"
        findings.append(
            DatumDefinitionFinding(
                datum=datum,
                status=STATUS_WARNING,
                code="ISO5459_REFERENCED_DATUM_NOT_DEFINED",
                standard=standard,
                mode=normalized_mode,
                finding=(
                    f"{prefix} of {standard}: datum {datum} is referenced by a GD&T specification, "
                    "but no corresponding datum feature indicator was identified in the drawing."
                ),
                recommended_action=(
                    f"Verify that datum {datum} is correctly defined in the drawing or correct the GD&T datum reference."
                ),
                normative_claim=normalized_mode == MODE_NORMATIVE,
                definition_count=0,
                definition_evidence=(),
                source_ref=source_ref,
            )
        )

    return findings


__all__ = [
    "MODE_NORMATIVE",
    "MODE_REFERENCE",
    "STATUS_PASS",
    "STATUS_WARNING",
    "DatumDefinitionFinding",
    "assess_referenced_datum_definitions",
]
