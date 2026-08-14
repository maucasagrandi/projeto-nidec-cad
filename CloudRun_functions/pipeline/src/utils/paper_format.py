"""
Detecção e comparação de formato de papel ISO 216 entre dois PDFs.

Converte dimensões de página PDF (em pontos, 72pt = 1 polegada) para milímetros
e classifica contra os formatos ISO padrão (A0–A5). Tolerância de ±3mm para
absorver variações de scan, render e arredondamento de PDF.

Também detecta mudança de orientação (retrato ↔ paisagem) como parte do mesmo
check, já que a combinação formato+orientação define o "drawing format" completo.
"""

from dataclasses import dataclass
from typing import Optional

import fitz  # PyMuPDF


# ==============================================================================
# Tabela ISO 216 — dimensões em mm (largura x altura em retrato)
# ==============================================================================

ISO_PAPER_SIZES_MM: dict[str, tuple[float, float]] = {
    "A0": (841, 1189),
    "A1": (594, 841),
    "A2": (420, 594),
    "A3": (297, 420),
    "A4": (210, 297),
    "A5": (148, 210),
}

# Tolerância em mm para considerar match com formato ISO
TOLERANCE_MM: float = 3.0

# Fator de conversão: 1 ponto PDF = 25.4/72 mm
PT_TO_MM: float = 25.4 / 72.0


# ==============================================================================
# Dataclasses de resultado
# ==============================================================================

@dataclass(frozen=True)
class PageFormat:
    """Formato detectado de uma página."""
    iso_name: Optional[str]       # ex: "A3", None se não reconhecido
    orientation: str              # "paisagem" ou "retrato"
    width_mm: float
    height_mm: float

    @property
    def display_name(self) -> str:
        if self.iso_name:
            return f"{self.iso_name} ({self.orientation})"
        return f"{self.width_mm:.0f}×{self.height_mm:.0f}mm ({self.orientation})"


@dataclass(frozen=True)
class FormatChangeResult:
    """Resultado da comparação de formato entre original e revisado."""
    original: PageFormat
    revised: PageFormat
    format_changed: bool          # True se o tamanho ISO mudou
    orientation_changed: bool     # True se a orientação mudou

    @property
    def description(self) -> str:
        """Descrição textual da mudança para exibição na UI e no prompt."""
        parts = []
        if self.format_changed:
            parts.append(
                f"Formato do papel alterado de {self.original.iso_name or 'desconhecido'} "
                f"para {self.revised.iso_name or 'desconhecido'} "
                f"({self.original.width_mm:.0f}×{self.original.height_mm:.0f}mm → "
                f"{self.revised.width_mm:.0f}×{self.revised.height_mm:.0f}mm)"
            )
        if self.orientation_changed:
            parts.append(
                f"Orientação alterada de {self.original.orientation} "
                f"para {self.revised.orientation}"
            )
        return "; ".join(parts) if parts else "Sem alteração de formato"


# ==============================================================================
# Funções de detecção
# ==============================================================================

def _pt_to_mm(pt: float) -> float:
    """Converte pontos PDF para milímetros."""
    return pt * PT_TO_MM


def detect_iso_format(width_pt: float, height_pt: float) -> PageFormat:
    """
    Detecta o formato ISO de uma página a partir de suas dimensões em pontos PDF.

    Normaliza para retrato (menor dimensão = largura) antes de comparar com a
    tabela ISO. Retorna PageFormat com iso_name=None se não casar com nenhum
    formato padrão dentro da tolerância.
    """
    w_mm = _pt_to_mm(width_pt)
    h_mm = _pt_to_mm(height_pt)

    # Determina orientação baseada nas dimensões reais
    if w_mm > h_mm:
        orientation = "paisagem"
        # Normaliza para retrato para comparação com tabela ISO
        portrait_w, portrait_h = h_mm, w_mm
    else:
        orientation = "retrato"
        portrait_w, portrait_h = w_mm, h_mm

    # Tenta casar com formato ISO
    matched_name: Optional[str] = None
    for name, (iso_w, iso_h) in ISO_PAPER_SIZES_MM.items():
        if (abs(portrait_w - iso_w) <= TOLERANCE_MM and
                abs(portrait_h - iso_h) <= TOLERANCE_MM):
            matched_name = name
            break

    return PageFormat(
        iso_name=matched_name,
        orientation=orientation,
        width_mm=w_mm,
        height_mm=h_mm,
    )


def check_format_change(
    pdf1_bytes: bytes,
    pdf2_bytes: bytes,
    page_index: int = 0,
) -> Optional[FormatChangeResult]:
    """
    Compara o formato de papel de uma página entre dois PDFs.

    Retorna FormatChangeResult se houver mudança de formato e/ou orientação.
    Retorna None se os formatos forem idênticos (sem nada a reportar).
    """
    doc1 = fitz.open(stream=pdf1_bytes, filetype="pdf")
    doc2 = fitz.open(stream=pdf2_bytes, filetype="pdf")

    try:
        if page_index >= len(doc1) or page_index >= len(doc2):
            return None

        page1 = doc1[page_index]
        page2 = doc2[page_index]

        fmt1 = detect_iso_format(page1.rect.width, page1.rect.height)
        fmt2 = detect_iso_format(page2.rect.width, page2.rect.height)

        format_changed = fmt1.iso_name != fmt2.iso_name
        orientation_changed = fmt1.orientation != fmt2.orientation

        if not format_changed and not orientation_changed:
            return None

        return FormatChangeResult(
            original=fmt1,
            revised=fmt2,
            format_changed=format_changed,
            orientation_changed=orientation_changed,
        )
    finally:
        doc1.close()
        doc2.close()


def check_all_pages_format(
    pdf1_bytes: bytes,
    pdf2_bytes: bytes,
) -> dict[int, FormatChangeResult]:
    """
    Verifica mudança de formato em todas as páginas comuns entre os dois PDFs.

    Retorna um dict {page_index: FormatChangeResult} apenas para páginas onde
    houve mudança. Dict vazio = nenhuma mudança detectada.
    """
    doc1 = fitz.open(stream=pdf1_bytes, filetype="pdf")
    doc2 = fitz.open(stream=pdf2_bytes, filetype="pdf")

    results: dict[int, FormatChangeResult] = {}

    try:
        n_pages = min(len(doc1), len(doc2))
        for i in range(n_pages):
            page1 = doc1[i]
            page2 = doc2[i]

            fmt1 = detect_iso_format(page1.rect.width, page1.rect.height)
            fmt2 = detect_iso_format(page2.rect.width, page2.rect.height)

            format_changed = fmt1.iso_name != fmt2.iso_name
            orientation_changed = fmt1.orientation != fmt2.orientation

            if format_changed or orientation_changed:
                results[i] = FormatChangeResult(
                    original=fmt1,
                    revised=fmt2,
                    format_changed=format_changed,
                    orientation_changed=orientation_changed,
                )
    finally:
        doc1.close()
        doc2.close()

    return results
