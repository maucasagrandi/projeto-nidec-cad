"""Parser determinístico do conteúdo textual de quadros GD&T.

Escopo da Fase 5A:
- NÃO detecta o quadro (recebe ``GdtFrameCandidate`` pronto);
- NÃO classifica o símbolo da primeira célula;
- NÃO decide conformidade ISO;
- NÃO valida se um datum referenciado está realmente definido no desenho.

O detector estável já preenche ``cell.texts`` via PyMuPDF. Este módulo somente
interpreta essa informação por posição de célula:

- cell[0] = característica visual (fornecida externamente pelo classificador);
- cell[1] = célula de tolerância;
- cell[2:] = células de referência de datum, quando textualmente inequívocas.

Símbolos/modificadores que não aparecem como texto continuam ``unresolved`` e
serão tratados por visão em uma etapa posterior da Fase 5.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import List, Optional, Sequence

from src.gdt.detector import GdtFrameCandidate

_DIAMETER_GLYPHS = {"⌀", "Ø", "∅"}
_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9])([+-]?(?:\d+(?:[.,]\d+)?|[.,]\d+))(?![A-Za-z0-9])"
)
_SINGLE_LETTER_RE = re.compile(r"^[A-Z]$")


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
    """Reconhece uma célula textual inequivocamente composta por uma letra.

    Se houver mais conteúdo na célula (por exemplo, um modificador gráfico que
    virou texto), ela fica para resolução posterior em vez de ser forçada como
    datum.
    """

    cleaned = [token.upper() for token in _clean_tokens(cell_tokens)]
    if len(cleaned) != 1:
        return None
    return cleaned[0] if _SINGLE_LETTER_RE.fullmatch(cleaned[0]) else None


def parse_feature_control_frame(
    candidate: GdtFrameCandidate,
    *,
    characteristic: Optional[str] = None,
) -> ParsedGdtFrame:
    """Interpreta textos por célula sem aplicar qualquer regra de conformidade."""

    cell_texts = [_clean_tokens(cell.texts) for cell in candidate.cells]
    unresolved_fields: List[str] = []
    unresolved_tokens: List[str] = []

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
        )

    tolerance_tokens = cell_texts[1]
    tolerance_raw, tolerance_value = _extract_first_number(tolerance_tokens)
    diameter_zone = _detect_textual_diameter(tolerance_tokens)

    if tolerance_value is None:
        unresolved_fields.append("tolerance_value")

    # Qualquer token da célula de tolerância que não seja número/diâmetro é
    # preservado para uma futura etapa visual de modificadores.
    for token in tolerance_tokens:
        without_diameter = token
        for glyph in _DIAMETER_GLYPHS:
            without_diameter = without_diameter.replace(glyph, "")
        without_number = _NUMBER_RE.sub("", without_diameter).strip()
        if without_number:
            unresolved_tokens.append(token)

    referenced_datums: List[str] = []
    for tokens in cell_texts[2:]:
        datum = _extract_structural_datum(tokens)
        if datum is not None:
            if datum not in referenced_datums:
                referenced_datums.append(datum)
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
    )


__all__ = ["ParsedGdtFrame", "parse_feature_control_frame"]
