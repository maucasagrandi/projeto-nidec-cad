"""Fail-closed ISO 1101 edition resolution and datum-semantics scaffolding.

Phase 6 must never guess an ISO 1101 edition.  This module therefore separates
three concerns:

1. parse an ISO 1101 citation and detect an explicit edition year when present;
2. if the drawing cites ISO 1101 without a year, resolve the edition only from
   explicit customer applicability/configuration context;
3. evaluate characteristic/datum semantics only from an explicit rule table.

No ISO semantic rule is hard-coded here.  The caller must supply edition-
specific rules sourced from the applicable customer/normative configuration.
That keeps the implementation deterministic while avoiding unsupported claims
about a standard edition that has not been resolved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Iterable, Mapping, Optional, Sequence

SUPPORTED_EDITIONS = (2012, 2017)

STATUS_RESOLVED = "resolved"
STATUS_NO_CITATION = "unresolved_no_iso1101_citation"
STATUS_UNSUPPORTED_EDITION = "unresolved_unsupported_edition"
STATUS_NO_APPLICABILITY = "unresolved_no_applicability_rule"
STATUS_MISSING_CONTEXT = "unresolved_missing_context"
STATUS_AMBIGUOUS = "unresolved_ambiguous_edition"

SEMANTIC_EVALUATED = "evaluated"
SEMANTIC_NOT_EVALUATED = "not_evaluated"
SEMANTIC_NEEDS_CONTEXT = "needs_condition_context"

_CITATION_RE = re.compile(
    r"\bISO\s*1101(?:\s*[:\-]\s*|\s+)?(?P<year>20\d{2})?\b",
    flags=re.IGNORECASE,
)


def _norm(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).upper()


def _norm_series(value: object) -> Optional[str]:
    text = _norm(value)
    return text or None


def _is_applicable(value: object) -> bool:
    text = _norm(value)
    return text in {"1", "TRUE", "YES", "Y", "APPLICABLE", "APLICAVEL", "APLICÁVEL", "ACTIVE", "ATIVO"}


@dataclass(frozen=True)
class Iso1101Citation:
    raw: str
    year: Optional[int] = None

    def to_dict(self) -> dict:
        return {"raw": self.raw, "year": self.year}


@dataclass(frozen=True)
class Iso1101ApplicabilityRule:
    edition: int
    compressor_series: Optional[str] = None
    applicability: bool = True
    standard: str = "ISO 1101"
    content: Optional[str] = None
    category: Optional[str] = None
    source_id: Optional[str] = None

    @classmethod
    def from_mapping(cls, row: Mapping[str, object]) -> "Iso1101ApplicabilityRule":
        edition_raw = row.get("edition", row.get("year"))
        if edition_raw is None:
            raise ValueError("applicability rule requires edition/year")
        edition = int(edition_raw)
        applicability_raw = row.get("applicability", True)
        applicability = (
            applicability_raw
            if isinstance(applicability_raw, bool)
            else _is_applicable(applicability_raw)
        )
        return cls(
            edition=edition,
            compressor_series=_norm_series(row.get("compressor_series", row.get("Compressor Series"))),
            applicability=bool(applicability),
            standard=str(row.get("standard", row.get("Standard", "ISO 1101"))),
            content=(str(row.get("content", row.get("Content"))).strip() if row.get("content", row.get("Content")) is not None else None),
            category=(str(row.get("category", row.get("Category"))).strip() if row.get("category", row.get("Category")) is not None else None),
            source_id=(str(row.get("source_id")).strip() if row.get("source_id") is not None else None),
        )

    def to_dict(self) -> dict:
        return {
            "standard": self.standard,
            "edition": self.edition,
            "compressor_series": self.compressor_series,
            "applicability": self.applicability,
            "content": self.content,
            "category": self.category,
            "source_id": self.source_id,
        }


@dataclass(frozen=True)
class Iso1101EditionResolution:
    status: str
    edition: Optional[int]
    source: Optional[str]
    reason: str
    citation: Optional[Iso1101Citation] = None
    matching_rules: tuple[Iso1101ApplicabilityRule, ...] = field(default_factory=tuple)

    @property
    def resolved(self) -> bool:
        return self.status == STATUS_RESOLVED and self.edition is not None

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "resolved": self.resolved,
            "edition": self.edition,
            "source": self.source,
            "reason": self.reason,
            "citation": self.citation.to_dict() if self.citation else None,
            "matching_rules": [row.to_dict() for row in self.matching_rules],
        }


def parse_iso1101_citation(text: str) -> Optional[Iso1101Citation]:
    match = _CITATION_RE.search(str(text or ""))
    if not match:
        return None
    year = int(match.group("year")) if match.group("year") else None
    return Iso1101Citation(raw=match.group(0), year=year)


def _collect_citations(citations: Sequence[str | Iso1101Citation]) -> list[Iso1101Citation]:
    output: list[Iso1101Citation] = []
    for value in citations:
        if isinstance(value, Iso1101Citation):
            output.append(value)
            continue
        parsed = parse_iso1101_citation(str(value))
        if parsed is not None:
            output.append(parsed)
    return output


def resolve_iso1101_edition(
    citations: Sequence[str | Iso1101Citation],
    *,
    applicability_rules: Iterable[Iso1101ApplicabilityRule | Mapping[str, object]] = (),
    compressor_series: Optional[str] = None,
) -> Iso1101EditionResolution:
    """Resolve the applicable ISO 1101 edition without guessing.

    Priority:
    1. one explicit supported year in the drawing citation;
    2. otherwise an exact compressor-series applicability rule;
    3. otherwise a generic applicability rule.

    Multiple conflicting years/rules remain unresolved.
    """

    parsed = _collect_citations(citations)
    if not parsed:
        return Iso1101EditionResolution(
            status=STATUS_NO_CITATION,
            edition=None,
            source=None,
            reason="drawing/context does not contain an ISO 1101 citation",
        )

    explicit_years = sorted({row.year for row in parsed if row.year is not None})
    if explicit_years:
        if len(explicit_years) > 1:
            return Iso1101EditionResolution(
                status=STATUS_AMBIGUOUS,
                edition=None,
                source="drawing_citation",
                reason=f"conflicting explicit ISO 1101 years: {explicit_years}",
                citation=parsed[0],
            )
        year = explicit_years[0]
        if year not in SUPPORTED_EDITIONS:
            return Iso1101EditionResolution(
                status=STATUS_UNSUPPORTED_EDITION,
                edition=None,
                source="drawing_citation",
                reason=f"ISO 1101:{year} is not in supported editions {list(SUPPORTED_EDITIONS)}",
                citation=next(row for row in parsed if row.year == year),
            )
        return Iso1101EditionResolution(
            status=STATUS_RESOLVED,
            edition=year,
            source="drawing_citation",
            reason="explicit edition year in drawing citation",
            citation=next(row for row in parsed if row.year == year),
        )

    rules: list[Iso1101ApplicabilityRule] = []
    for raw in applicability_rules:
        rule = raw if isinstance(raw, Iso1101ApplicabilityRule) else Iso1101ApplicabilityRule.from_mapping(raw)
        if _norm(rule.standard).replace(" ", "") != "ISO1101":
            continue
        if not rule.applicability:
            continue
        rules.append(rule)

    series = _norm_series(compressor_series)
    exact = [row for row in rules if series is not None and _norm_series(row.compressor_series) == series]
    generic = [row for row in rules if _norm_series(row.compressor_series) in {None, "*", "ALL", "TODAS", "TODOS"}]

    if exact:
        eligible = exact
        source = "customer_applicability_exact_series"
    elif generic:
        eligible = generic
        source = "customer_applicability_generic"
    else:
        series_specific = [row for row in rules if _norm_series(row.compressor_series) not in {None, "*", "ALL", "TODAS", "TODOS"}]
        if series is None and series_specific:
            return Iso1101EditionResolution(
                status=STATUS_MISSING_CONTEXT,
                edition=None,
                source="customer_applicability",
                reason="ISO 1101 citation has no year and applicability is compressor-series-specific, but compressor_series is missing",
                citation=parsed[0],
                matching_rules=tuple(series_specific),
            )
        return Iso1101EditionResolution(
            status=STATUS_NO_APPLICABILITY,
            edition=None,
            source="customer_applicability",
            reason="ISO 1101 citation has no year and no applicable edition rule matched the provided context",
            citation=parsed[0],
        )

    editions = sorted({row.edition for row in eligible})
    unsupported = [year for year in editions if year not in SUPPORTED_EDITIONS]
    if unsupported:
        return Iso1101EditionResolution(
            status=STATUS_UNSUPPORTED_EDITION,
            edition=None,
            source=source,
            reason=f"matching applicability rule references unsupported edition(s): {unsupported}",
            citation=parsed[0],
            matching_rules=tuple(eligible),
        )
    if len(editions) != 1:
        return Iso1101EditionResolution(
            status=STATUS_AMBIGUOUS,
            edition=None,
            source=source,
            reason=f"multiple applicable ISO 1101 editions matched: {editions}",
            citation=parsed[0],
            matching_rules=tuple(eligible),
        )

    return Iso1101EditionResolution(
        status=STATUS_RESOLVED,
        edition=editions[0],
        source=source,
        reason="edition resolved from explicit customer applicability rule",
        citation=parsed[0],
        matching_rules=tuple(eligible),
    )


@dataclass(frozen=True)
class CharacteristicSemanticRule:
    characteristic: str
    edition: int
    datum_requirement: str
    source_ref: str
    notes: Optional[str] = None

    @classmethod
    def from_mapping(cls, row: Mapping[str, object]) -> "CharacteristicSemanticRule":
        return cls(
            characteristic=_norm(row.get("characteristic")).lower(),
            edition=int(row.get("edition")),
            datum_requirement=str(row.get("datum_requirement", "unknown")).strip().lower(),
            source_ref=str(row.get("source_ref", "unspecified")).strip(),
            notes=(str(row.get("notes")).strip() if row.get("notes") is not None else None),
        )


@dataclass(frozen=True)
class DatumSemanticAssessment:
    status: str
    result: str
    characteristic: Optional[str]
    edition: Optional[int]
    referenced_datums: tuple[str, ...]
    datum_requirement: Optional[str]
    source_ref: Optional[str]
    reason: str

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "result": self.result,
            "characteristic": self.characteristic,
            "edition": self.edition,
            "referenced_datums": list(self.referenced_datums),
            "datum_requirement": self.datum_requirement,
            "source_ref": self.source_ref,
            "reason": self.reason,
        }


def assess_datum_reference_semantics(
    *,
    characteristic: Optional[str],
    referenced_datums: Sequence[str],
    edition_resolution: Iso1101EditionResolution,
    rules: Iterable[CharacteristicSemanticRule | Mapping[str, object]],
) -> DatumSemanticAssessment:
    """Apply only an explicitly supplied edition-specific semantic rule."""

    refs = tuple(str(value).strip().upper() for value in referenced_datums if str(value).strip())
    if not edition_resolution.resolved:
        return DatumSemanticAssessment(
            status=SEMANTIC_NOT_EVALUATED,
            result="edition_unresolved",
            characteristic=characteristic,
            edition=None,
            referenced_datums=refs,
            datum_requirement=None,
            source_ref=None,
            reason="ISO 1101 edition must be resolved before semantic evaluation",
        )
    if not characteristic:
        return DatumSemanticAssessment(
            status=SEMANTIC_NOT_EVALUATED,
            result="characteristic_unresolved",
            characteristic=None,
            edition=edition_resolution.edition,
            referenced_datums=refs,
            datum_requirement=None,
            source_ref=None,
            reason="GD&T characteristic is unresolved",
        )

    normalized = _norm(characteristic).lower()
    rule_rows = [
        row if isinstance(row, CharacteristicSemanticRule) else CharacteristicSemanticRule.from_mapping(row)
        for row in rules
    ]
    matches = [row for row in rule_rows if row.edition == edition_resolution.edition and row.characteristic == normalized]
    if len(matches) != 1:
        return DatumSemanticAssessment(
            status=SEMANTIC_NOT_EVALUATED,
            result="semantic_rule_missing_or_ambiguous",
            characteristic=normalized,
            edition=edition_resolution.edition,
            referenced_datums=refs,
            datum_requirement=None,
            source_ref=None,
            reason=f"expected exactly one semantic rule, found {len(matches)}",
        )

    rule = matches[0]
    requirement = rule.datum_requirement
    if requirement == "required":
        result = "compatible" if refs else "missing_required_datum_reference"
        status = SEMANTIC_EVALUATED
    elif requirement == "none":
        result = "compatible" if not refs else "datum_reference_not_permitted_by_rule"
        status = SEMANTIC_EVALUATED
    elif requirement == "optional":
        result = "compatible"
        status = SEMANTIC_EVALUATED
    elif requirement == "conditional":
        result = "condition_context_required"
        status = SEMANTIC_NEEDS_CONTEXT
    else:
        result = "unknown_requirement"
        status = SEMANTIC_NOT_EVALUATED

    return DatumSemanticAssessment(
        status=status,
        result=result,
        characteristic=normalized,
        edition=edition_resolution.edition,
        referenced_datums=refs,
        datum_requirement=requirement,
        source_ref=rule.source_ref,
        reason=(rule.notes or f"applied explicit semantic rule for ISO 1101:{rule.edition}"),
    )


__all__ = [
    "SUPPORTED_EDITIONS",
    "Iso1101Citation",
    "Iso1101ApplicabilityRule",
    "Iso1101EditionResolution",
    "CharacteristicSemanticRule",
    "DatumSemanticAssessment",
    "parse_iso1101_citation",
    "resolve_iso1101_edition",
    "assess_datum_reference_semantics",
]
