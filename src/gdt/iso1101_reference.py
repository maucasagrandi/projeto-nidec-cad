"""ISO 1101 reference-baseline findings for CAD Review.

This module is intentionally separate from normative applicability resolution.
The customer use case can evaluate GD&T consistency against ISO 1101:2017 even
when the drawing/TSS chain does not explicitly cite that edition.

Two claim modes are supported:

``reference``
    The standard is used as a technical baseline. Findings are worded as
    *potential* violations and do not claim contractual applicability.

``normative``
    Use only when upstream applicability logic has established that the
    standard/edition governs the drawing. A failing rule can then be reported
    as a violation.

The rule table itself remains external/configured and carries the exact
``source_ref`` used in the finding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

from src.gdt.iso1101 import (
    CharacteristicSemanticRule,
    Iso1101EditionResolution,
    STATUS_RESOLVED,
    assess_datum_reference_semantics,
)

MODE_REFERENCE = "reference"
MODE_NORMATIVE = "normative"

FINDING_PASS = "PASS"
FINDING_WARNING = "WARNING"
FINDING_NEEDS_CONTEXT = "NEEDS_CONTEXT"
FINDING_NOT_EVALUATED = "NOT_EVALUATED"


@dataclass(frozen=True)
class Iso1101RuleFinding:
    status: str
    code: str
    standard: str
    edition: int
    mode: str
    source_ref: Optional[str]
    characteristic: Optional[str]
    referenced_datums: tuple[str, ...]
    datum_requirement: Optional[str]
    finding: str
    recommended_action: Optional[str]
    normative_claim: bool

    @property
    def standard_label(self) -> str:
        return f"{self.standard}:{self.edition}"

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "code": self.code,
            "standard": self.standard,
            "edition": self.edition,
            "standard_label": self.standard_label,
            "mode": self.mode,
            "source_ref": self.source_ref,
            "characteristic": self.characteristic,
            "referenced_datums": list(self.referenced_datums),
            "datum_requirement": self.datum_requirement,
            "finding": self.finding,
            "recommended_action": self.recommended_action,
            "normative_claim": self.normative_claim,
        }


def reference_baseline_resolution(
    *,
    edition: int = 2017,
    source_ref: str = "configured ISO 1101 reference baseline",
) -> Iso1101EditionResolution:
    """Create an explicit non-drawing ISO resolution for reference checking."""

    return Iso1101EditionResolution(
        status=STATUS_RESOLVED,
        edition=int(edition),
        source="reference_baseline",
        reason=f"ISO 1101:{int(edition)} selected as CAD Review reference baseline ({source_ref})",
    )


def _prefix(mode: str, standard_label: str) -> str:
    if mode == MODE_NORMATIVE:
        return f"Violation of {standard_label}"
    return f"Potential violation of {standard_label}"


def assess_iso1101_datum_rule(
    *,
    characteristic: Optional[str],
    referenced_datums: Sequence[str],
    rules: Iterable[CharacteristicSemanticRule | Mapping[str, object]],
    edition: int = 2017,
    mode: str = MODE_REFERENCE,
) -> Iso1101RuleFinding:
    """Convert ISO datum semantics into a user-facing CAD Review finding."""

    if mode not in {MODE_REFERENCE, MODE_NORMATIVE}:
        raise ValueError(f"unsupported ISO finding mode: {mode}")

    resolution = reference_baseline_resolution(edition=edition)
    assessment = assess_datum_reference_semantics(
        characteristic=characteristic,
        referenced_datums=referenced_datums,
        edition_resolution=resolution,
        rules=rules,
    )
    refs = assessment.referenced_datums
    standard = "ISO 1101"
    standard_label = f"{standard}:{edition}"
    normalized_characteristic = assessment.characteristic or characteristic

    if assessment.result == "compatible":
        if assessment.datum_requirement == "required":
            detail = (
                f"{normalized_characteristic} requires a datum reference and the GD&T frame "
                f"contains reference datum(s): {', '.join(refs)}."
            )
        elif assessment.datum_requirement == "none":
            detail = f"{normalized_characteristic} does not use a datum reference and none was identified."
        else:
            detail = f"The GD&T datum-reference structure is compatible with the configured {standard_label} rule."
        return Iso1101RuleFinding(
            status=FINDING_PASS,
            code="ISO1101_DATUM_RULE_OK",
            standard=standard,
            edition=edition,
            mode=mode,
            source_ref=assessment.source_ref,
            characteristic=normalized_characteristic,
            referenced_datums=refs,
            datum_requirement=assessment.datum_requirement,
            finding=detail,
            recommended_action=None,
            normative_claim=mode == MODE_NORMATIVE,
        )

    if assessment.result == "missing_required_datum_reference":
        return Iso1101RuleFinding(
            status=FINDING_WARNING,
            code="ISO1101_REQUIRED_DATUM_MISSING",
            standard=standard,
            edition=edition,
            mode=mode,
            source_ref=assessment.source_ref,
            characteristic=normalized_characteristic,
            referenced_datums=refs,
            datum_requirement=assessment.datum_requirement,
            finding=(
                f"{_prefix(mode, standard_label)}: {normalized_characteristic} was defined without a "
                "reference datum, while the configured ISO rule requires a datum reference."
            ),
            recommended_action="Verify the GD&T specification and add/correct the required reference datum.",
            normative_claim=mode == MODE_NORMATIVE,
        )

    if assessment.result == "datum_reference_not_permitted_by_rule":
        return Iso1101RuleFinding(
            status=FINDING_WARNING,
            code="ISO1101_DATUM_NOT_USED_BY_CHARACTERISTIC",
            standard=standard,
            edition=edition,
            mode=mode,
            source_ref=assessment.source_ref,
            characteristic=normalized_characteristic,
            referenced_datums=refs,
            datum_requirement=assessment.datum_requirement,
            finding=(
                f"{_prefix(mode, standard_label)}: {normalized_characteristic} contains reference datum(s) "
                f"{', '.join(refs)}, while the configured ISO rule lists datum needed = no."
            ),
            recommended_action="Verify whether the datum reference should be removed or the GD&T characteristic corrected.",
            normative_claim=mode == MODE_NORMATIVE,
        )

    if assessment.result == "condition_context_required":
        return Iso1101RuleFinding(
            status=FINDING_NEEDS_CONTEXT,
            code="ISO1101_DATUM_REQUIREMENT_CONDITIONAL",
            standard=standard,
            edition=edition,
            mode=mode,
            source_ref=assessment.source_ref,
            characteristic=normalized_characteristic,
            referenced_datums=refs,
            datum_requirement=assessment.datum_requirement,
            finding=(
                f"{standard_label} allows context-dependent datum usage for {normalized_characteristic}; "
                "datum presence/absence alone is insufficient to declare a violation."
            ),
            recommended_action="Review the feature function/specification context before classifying this GD&T as compliant or non-compliant.",
            normative_claim=False,
        )

    return Iso1101RuleFinding(
        status=FINDING_NOT_EVALUATED,
        code="ISO1101_RULE_NOT_EVALUATED",
        standard=standard,
        edition=edition,
        mode=mode,
        source_ref=assessment.source_ref,
        characteristic=normalized_characteristic,
        referenced_datums=refs,
        datum_requirement=assessment.datum_requirement,
        finding=f"ISO datum-reference rule could not be evaluated: {assessment.reason}",
        recommended_action="Review the configured ISO rule table and the resolved GD&T characteristic.",
        normative_claim=False,
    )


__all__ = [
    "MODE_REFERENCE",
    "MODE_NORMATIVE",
    "FINDING_PASS",
    "FINDING_WARNING",
    "FINDING_NEEDS_CONTEXT",
    "FINDING_NOT_EVALUATED",
    "Iso1101RuleFinding",
    "reference_baseline_resolution",
    "assess_iso1101_datum_rule",
]
