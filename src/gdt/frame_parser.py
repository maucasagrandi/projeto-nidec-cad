"""Parser determinístico de conteúdo estruturado de quadros GD&T.

Escopo da Fase 5:
- NÃO detecta o quadro (recebe ``GdtFrameCandidate`` pronto);
- NÃO classifica o símbolo da primeira célula;
- NÃO decide conformidade ISO;
- NÃO valida se um datum referenciado está realmente definido no desenho.

O parser combina duas fontes, com proveniência explícita:

1. texto PDF por célula, quando existe e é inequívoco;
2. evidência visual determinística já produzida pela Fase 5, usada somente para
   preencher campos que o texto não resolveu.

A ordem é intencional: texto confiável tem prioridade. Se texto e visão
conflitarem, o campo fica ``unresolved`` em vez de uma das fontes ser escolhida
silenciosamente.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Dict, List, Optional, Sequence

from src.gdt.detector import GdtFrameCandidate
from src.gdt.tolerance_cell import ToleranceCellAssessment

_DIAMETER_GLYPHS = {"⌀", "Ø", "∅"}
_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9])([+-]?(?:\d+(?:[.,]\d+)?|[.,]\d+))(?![A-Za-z0-9])"
)
_SINGLE_LETTER_RE = re.compile(r"^[A-Z]$")


@dataclass(frozen=True)
class FrameVisualEvidence:
    """Structured visual evidence supplied to the frame parser.

    ``datum_by_cell`` maps the real cell index (2+) to one recognized uppercase
    datum letter. Acceptance / threshold policy belongs to the caller; this
    object only transports the already-selected result.
    """

    datum_by_cell: Dict[int, str] = field(default_factory=dict)
    tolerance_assessment: Optional[ToleranceCellAssessment] = None
    source: str = "phase5_visual_pipeline"
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "datum_by_cell": {str(key): value for key, value in sorted(self.datum_by_cell.items())},
            "tolerance_assessment": (
                self.tolerance_assessment.to_dict() if self.tolerance_assessment is not None else None
            ),
            "source": self.source,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class ParsedGdtFrame:
    """Leitura estrutural de um feature control frame, antes das regras ISO."""

    candidate_id: str
    page: int
    characteristic: Optional[str]
    cell_texts: List[List[str]]
    tolerance_cell_index: Optional[int]
    tolerance_raw: Optional[str]
    tolerance_value: Optional[float]
    diameter_zone: Optional[bool]
    referenced_datums: List[str] = field(default_factory=list)
    unresolved_tokens: List[str] = field(default_factory=list)
    unresolved_fields: List[str] = field(default_factory=list)
    field_sources: dict = field(default_factory=dict)
    evidence_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "page": self.page,
            "characteristic": self.characteristic,
            "cell_texts": self.cell_texts,
            "tolerance_cell_index": self.tolerance_cell_index,
            "tolerance_raw": self.tolerance_raw,
            "tolerance_value": self.tolerance_value,
            "diameter_zone": self.diameter_zone,
            "referenced_datums": self.referenced_datums,
            "unresolved_tokens": self.unresolved_tokens,
            "unresolved_fields": self.unresolved_fields,
            "field_sources": self.field_sources,
            "evidence_notes": self.evidence_notes,
        }


def _clean_tokens(tokens: Sequence[str]) -> List[str]:
    return [str(token).strip() for token in tokens if str(token).strip()]


def _extract_first_number(tokens: Sequence[str]) -> tuple[Optional[str], Optional[float]]:
    """Extrai o primeiro número decimal preservando a representação original."""

    for token in _clean_tokens(tokens):
        match = _NUMBER_RE.search(token)
        if not match:
            continue
        raw = match.group(1)
        normalized = raw.replace(",", ".")
        if normalized.startswith("."):
            normalized = "0" + normalized
        elif normalized.startswith("+."):
            normalized = "+0" + normalized[1:]
        elif normalized.startswith("-."):
            normalized = "-0" + normalized[1:]
        try:
            value = float(normalized)
        except ValueError:
            continue
        return raw, value
    return None, None


def _detect_textual_diameter(tokens: Sequence[str]) -> Optional[bool]:
    """Retorna True quando o símbolo de diâmetro existe textualmente.

    ``None`` significa "não resolvido", e não ``False``: a ausência no texto
    não prova ausência no desenho, pois o símbolo pode ser puramente gráfico.
    """

    cleaned = _clean_tokens(tokens)
    for token in cleaned:
        if any(glyph in token for glyph in _DIAMETER_GLYPHS):
            return True
    return None


def _extract_structural_datum(cell_tokens: Sequence[str]) -> Optional[str]:
    """Reconhece uma célula textual inequivocamente composta por uma letra."""

    cleaned = [token.upper() for token in _clean_tokens(cell_tokens)]
    if len(cleaned) != 1:
        return None
    return cleaned[0] if _SINGLE_LETTER_RE.fullmatch(cleaned[0]) else None


def _normalized_visual_datum(value: object) -> Optional[str]:
    text = str(value).strip().upper()
    return text if _SINGLE_LETTER_RE.fullmatch(text) else None


def parse_feature_control_frame(
    candidate: GdtFrameCandidate,
    *,
    characteristic: Optional[str] = None,
    visual_evidence: Optional[FrameVisualEvidence] = None,
) -> ParsedGdtFrame:
    """Interpreta o frame sem aplicar qualquer regra de conformidade ISO."""

    cell_texts = [_clean_tokens(cell.texts) for cell in candidate.cells]
    unresolved_fields: List[str] = []
    unresolved_tokens: List[str] = []
    evidence_notes: List[str] = list(visual_evidence.notes) if visual_evidence is not None else []
    field_sources: dict = {
        "characteristic": "external_classifier" if characteristic is not None else None,
        "tolerance": None,
        "diameter_zone": None,
        "datum_cells": {},
    }

    if characteristic is None:
        unresolved_fields.append("characteristic")

    if len(cell_texts) < 2:
        unresolved_fields.append("tolerance_cell")
        return ParsedGdtFrame(
            candidate_id=candidate.candidate_id,
            page=candidate.page,
            characteristic=characteristic,
            cell_texts=cell_texts,
            tolerance_cell_index=None,
            tolerance_raw=None,
            tolerance_value=None,
            diameter_zone=None,
            referenced_datums=[],
            unresolved_tokens=[],
            unresolved_fields=unresolved_fields,
            field_sources=field_sources,
            evidence_notes=evidence_notes,
        )

    tolerance_tokens = cell_texts[1]
    tolerance_raw, tolerance_value = _extract_first_number(tolerance_tokens)
    diameter_zone = _detect_textual_diameter(tolerance_tokens)

    if tolerance_value is not None:
        field_sources["tolerance"] = "pdf_cell_text"
    elif visual_evidence is not None and visual_evidence.tolerance_assessment is not None:
        assessment = visual_evidence.tolerance_assessment
        if assessment.resolved:
            tolerance_raw = assessment.tolerance_raw
            tolerance_value = assessment.tolerance_value
            field_sources["tolerance"] = assessment.source
        else:
            evidence_notes.append(
                f"tolerance_cell_status={assessment.status}: {assessment.reason}"
            )

    if diameter_zone is True:
        field_sources["diameter_zone"] = "pdf_cell_text"
    elif visual_evidence is not None and visual_evidence.tolerance_assessment is not None:
        assessment = visual_evidence.tolerance_assessment
        if assessment.diameter_zone is not None:
            diameter_zone = assessment.diameter_zone
            field_sources["diameter_zone"] = assessment.source

    if tolerance_value is None:
        unresolved_fields.append("tolerance_value")

    # Qualquer token da célula de tolerância que não seja número/diâmetro é
    # preservado; ele pode representar modificador ou ruído ainda não resolvido.
    for token in tolerance_tokens:
        without_diameter = token
        for glyph in _DIAMETER_GLYPHS:
            without_diameter = without_diameter.replace(glyph, "")
        without_number = _NUMBER_RE.sub("", without_diameter).strip()
        if without_number:
            unresolved_tokens.append(token)

    visual_datums = visual_evidence.datum_by_cell if visual_evidence is not None else {}
    referenced_datums: List[str] = []
    for cell_index, tokens in enumerate(cell_texts[2:], start=2):
        text_datum = _extract_structural_datum(tokens)
        visual_datum = _normalized_visual_datum(visual_datums.get(cell_index)) if cell_index in visual_datums else None

        if text_datum is not None and visual_datum is not None and text_datum != visual_datum:
            unresolved_fields.append(f"datum_cell_{cell_index}_conflict")
            evidence_notes.append(
                f"datum cell[{cell_index}] conflict: text={text_datum} visual={visual_datum}"
            )
            continue

        datum = text_datum or visual_datum
        if datum is not None:
            if datum not in referenced_datums:
                referenced_datums.append(datum)
            field_sources["datum_cells"][str(cell_index)] = (
                "pdf_cell_text" if text_datum is not None else visual_evidence.source
            )
            continue

        if tokens:
            unresolved_tokens.extend(tokens)

    return ParsedGdtFrame(
        candidate_id=candidate.candidate_id,
        page=candidate.page,
        characteristic=characteristic,
        cell_texts=cell_texts,
        tolerance_cell_index=1,
        tolerance_raw=tolerance_raw,
        tolerance_value=tolerance_value,
        diameter_zone=diameter_zone,
        referenced_datums=referenced_datums,
        unresolved_tokens=unresolved_tokens,
        unresolved_fields=unresolved_fields,
        field_sources=field_sources,
        evidence_notes=evidence_notes,
    )


__all__ = [
    "FrameVisualEvidence",
    "ParsedGdtFrame",
    "parse_feature_control_frame",
]
