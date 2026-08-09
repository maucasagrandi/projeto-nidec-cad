"""Extração determinística da camada textual de baixo nível de PDFs CAD.

Este módulo existe para um caso específico observado nos desenhos do projeto:
o conteúdo visual do CAD é majoritariamente vetorial, mas o PDF também contém
uma camada textual invisível (``3 Tr``) que permite seleção/cópia no leitor.

O caminho comum do PyMuPDF (``get_text('words')`` / ``rawdict``) não associa
essa camada às células GD&T. Aqui reconstruímos os operadores do content stream
(``Tm``, ``Td``, ``Tj`` etc.) e calculamos a origem de cada trecho textual em
coordenadas PyMuPDF.

Limites deliberados:
- não é um interpretador PDF completo;
- não usa OCR nem LLM;
- o posicionamento usa a origem/baseline do ``Tj`` (não estima bbox do texto);
- suporta os operadores de texto e gráficos necessários para o CAD atual,
  preservando diagnóstico explícito para evolução futura.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Iterator, List, Optional, Sequence, Tuple

import fitz


PdfMatrix = Tuple[float, float, float, float, float, float]
Point = Tuple[float, float]


@dataclass(frozen=True)
class PdfTextEvent:
    """Um ``Tj``/``TJ`` reconstruído do content stream."""

    xref: int
    sequence: int
    operator: str
    text: str
    raw_hex: str
    font_resource: Optional[str]
    font_size: Optional[float]
    rendering_mode: int
    pdf_origin: Point
    page_origin: Point

    @property
    def invisible(self) -> bool:
        return self.rendering_mode == 3

    def to_dict(self) -> dict:
        return {
            "xref": self.xref,
            "sequence": self.sequence,
            "operator": self.operator,
            "text": self.text,
            "raw_hex": self.raw_hex,
            "font_resource": self.font_resource,
            "font_size": self.font_size,
            "rendering_mode": self.rendering_mode,
            "invisible": self.invisible,
            "pdf_origin": [round(v, 4) for v in self.pdf_origin],
            "page_origin": [round(v, 4) for v in self.page_origin],
        }


@dataclass(frozen=True)
class _Name:
    value: str


_ARRAY_MARKER = object()


def _is_ws(value: int) -> bool:
    return value in b"\x00\t\n\x0c\r "


def _is_delimiter(value: int) -> bool:
    return value in b"()<>[]{}/%"


def _parse_literal_string(data: bytes, start: int) -> tuple[bytes, int]:
    """Lê string literal PDF começando logo após ``(``."""

    out = bytearray()
    depth = 1
    i = start

    escape_map = {
        ord("n"): b"\n",
        ord("r"): b"\r",
        ord("t"): b"\t",
        ord("b"): b"\b",
        ord("f"): b"\f",
    }

    while i < len(data) and depth > 0:
        ch = data[i]
        i += 1

        if ch == ord("\\"):
            if i >= len(data):
                break
            nxt = data[i]
            i += 1

            if nxt in escape_map:
                out.extend(escape_map[nxt])
                continue
            if nxt in (ord("("), ord(")"), ord("\\")):
                out.append(nxt)
                continue
            if nxt == ord("\r"):
                if i < len(data) and data[i] == ord("\n"):
                    i += 1
                continue
            if nxt == ord("\n"):
                continue
            if ord("0") <= nxt <= ord("7"):
                digits = bytearray([nxt])
                for _ in range(2):
                    if i < len(data) and ord("0") <= data[i] <= ord("7"):
                        digits.append(data[i])
                        i += 1
                    else:
                        break
                out.append(int(digits.decode("ascii"), 8) & 0xFF)
                continue

            # Escape desconhecido: a especificação permite tratar o caractere
            # literalmente. Isso é preferível a descartá-lo.
            out.append(nxt)
            continue

        if ch == ord("("):
            depth += 1
            out.append(ch)
            continue
        if ch == ord(")"):
            depth -= 1
            if depth > 0:
                out.append(ch)
            continue

        out.append(ch)

    return bytes(out), i


def _parse_hex_string(data: bytes, start: int) -> tuple[bytes, int]:
    digits = bytearray()
    i = start
    while i < len(data):
        ch = data[i]
        i += 1
        if ch == ord(">"):
            break
        if _is_ws(ch):
            continue
        digits.append(ch)

    if len(digits) % 2:
        digits.append(ord("0"))
    try:
        return bytes.fromhex(digits.decode("ascii")), i
    except (ValueError, UnicodeDecodeError):
        return bytes(digits), i


def _tokens(data: bytes) -> Iterator[Any]:
    """Tokenizador enxuto para content streams PDF."""

    i = 0
    n = len(data)
    while i < n:
        ch = data[i]

        if _is_ws(ch):
            i += 1
            continue

        if ch == ord("%"):
            while i < n and data[i] not in (ord("\r"), ord("\n")):
                i += 1
            continue

        if ch == ord("("):
            value, i = _parse_literal_string(data, i + 1)
            yield value
            continue

        if ch == ord("<"):
            if i + 1 < n and data[i + 1] == ord("<"):
                yield "<<"
                i += 2
            else:
                value, i = _parse_hex_string(data, i + 1)
                yield value
            continue

        if ch == ord(">") and i + 1 < n and data[i + 1] == ord(">"):
            yield ">>"
            i += 2
            continue

        if ch == ord("["):
            yield _ARRAY_MARKER
            i += 1
            continue

        if ch == ord("]"):
            yield "]"
            i += 1
            continue

        if ch == ord("/"):
            i += 1
            start = i
            while i < n and not _is_ws(data[i]) and not _is_delimiter(data[i]):
                i += 1
            yield _Name(data[start:i].decode("latin-1", errors="replace"))
            continue

        start = i
        while i < n and not _is_ws(data[i]) and not _is_delimiter(data[i]):
            i += 1
        raw = data[start:i]
        if not raw:
            # Delimitador não tratado; evita loop infinito.
            i += 1
            continue

        text = raw.decode("latin-1", errors="replace")
        try:
            if any(c in text for c in ".eE"):
                yield float(text)
            else:
                yield int(text)
            continue
        except ValueError:
            yield text


def _decode_pdf_bytes(raw: bytes, encoding: str = "cp1252") -> str:
    try:
        return raw.decode(encoding, errors="replace")
    except LookupError:
        return raw.decode("latin-1", errors="replace")


def _as_matrix(values: Sequence[Any]) -> Optional[fitz.Matrix]:
    if len(values) < 6:
        return None
    try:
        a, b, c, d, e, f = (float(v) for v in values[-6:])
    except (TypeError, ValueError):
        return None
    return fitz.Matrix(a, b, c, d, e, f)


def _number(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class PdfContentTextParser:
    """Reconstrói estado textual ao percorrer streams concatenados da página."""

    def __init__(
        self,
        *,
        pdf_to_page: Optional[fitz.Matrix] = None,
        text_encoding: str = "cp1252",
    ) -> None:
        self.pdf_to_page = pdf_to_page or fitz.Matrix(1, 0, 0, 1, 0, 0)
        self.text_encoding = text_encoding

        self.ctm = fitz.Matrix(1, 0, 0, 1, 0, 0)
        self.graphics_stack: list[fitz.Matrix] = []

        self.in_text = False
        self.text_matrix = fitz.Matrix(1, 0, 0, 1, 0, 0)
        self.line_matrix = fitz.Matrix(1, 0, 0, 1, 0, 0)
        self.leading = 0.0
        self.font_resource: Optional[str] = None
        self.font_size: Optional[float] = None
        self.rendering_mode = 0

        self.sequence = 0

    def _origin(self) -> tuple[Point, Point]:
        pdf_point = fitz.Point(0, 0) * self.text_matrix * self.ctm
        page_point = pdf_point * self.pdf_to_page
        return (
            (float(pdf_point.x), float(pdf_point.y)),
            (float(page_point.x), float(page_point.y)),
        )

    def _emit(self, xref: int, operator: str, raw_parts: Iterable[bytes]) -> Optional[PdfTextEvent]:
        raw_list = list(raw_parts)
        if not raw_list:
            return None
        raw = b"".join(raw_list)
        text = "".join(_decode_pdf_bytes(part, self.text_encoding) for part in raw_list)
        pdf_origin, page_origin = self._origin()
        event = PdfTextEvent(
            xref=int(xref),
            sequence=self.sequence,
            operator=operator,
            text=text,
            raw_hex=raw.hex(),
            font_resource=self.font_resource,
            font_size=self.font_size,
            rendering_mode=int(self.rendering_mode),
            pdf_origin=pdf_origin,
            page_origin=page_origin,
        )
        self.sequence += 1
        return event

    def _apply_td(self, tx: float, ty: float) -> None:
        translation = fitz.Matrix(1, 0, 0, 1, tx, ty)
        self.line_matrix = translation * self.line_matrix
        self.text_matrix = fitz.Matrix(self.line_matrix)

    def _handle_operator(self, xref: int, operator: str, operands: list[Any]) -> Optional[PdfTextEvent]:
        if operator == "q":
            self.graphics_stack.append(fitz.Matrix(self.ctm))
            return None
        if operator == "Q":
            if self.graphics_stack:
                self.ctm = self.graphics_stack.pop()
            return None
        if operator == "cm":
            matrix = _as_matrix(operands)
            if matrix is not None:
                self.ctm = matrix * self.ctm
            return None

        if operator == "BT":
            self.in_text = True
            self.text_matrix = fitz.Matrix(1, 0, 0, 1, 0, 0)
            self.line_matrix = fitz.Matrix(1, 0, 0, 1, 0, 0)
            return None
        if operator == "ET":
            self.in_text = False
            return None

        if operator == "Tf" and len(operands) >= 2:
            name = operands[-2]
            size = _number(operands[-1])
            if isinstance(name, _Name):
                self.font_resource = name.value
            if size is not None:
                self.font_size = size
            return None

        if operator == "Tr" and operands:
            mode = _number(operands[-1])
            if mode is not None:
                self.rendering_mode = int(mode)
            return None

        if operator == "TL" and operands:
            value = _number(operands[-1])
            if value is not None:
                self.leading = value
            return None

        if operator == "Tm":
            matrix = _as_matrix(operands)
            if matrix is not None:
                self.text_matrix = matrix
                self.line_matrix = fitz.Matrix(matrix)
            return None

        if operator in ("Td", "TD") and len(operands) >= 2:
            tx = _number(operands[-2])
            ty = _number(operands[-1])
            if tx is not None and ty is not None:
                if operator == "TD":
                    self.leading = -ty
                self._apply_td(tx, ty)
            return None

        if operator == "T*":
            self._apply_td(0.0, -self.leading)
            return None

        if operator == "Tj" and operands and isinstance(operands[-1], bytes):
            return self._emit(xref, operator, [operands[-1]])

        if operator == "TJ" and operands and isinstance(operands[-1], list):
            parts = [item for item in operands[-1] if isinstance(item, bytes)]
            return self._emit(xref, operator, parts)

        if operator == "'" and operands and isinstance(operands[-1], bytes):
            self._apply_td(0.0, -self.leading)
            return self._emit(xref, operator, [operands[-1]])

        if operator == '"' and len(operands) >= 3 and isinstance(operands[-1], bytes):
            # aw ac string "  -> word spacing / char spacing não afetam a origem.
            self._apply_td(0.0, -self.leading)
            return self._emit(xref, operator, [operands[-1]])

        return None

    def parse_stream(self, stream: bytes, *, xref: int) -> list[PdfTextEvent]:
        """Processa um stream mantendo o estado para o próximo stream da página."""

        events: list[PdfTextEvent] = []
        operands: list[Any] = []

        for token in _tokens(stream):
            if token is _ARRAY_MARKER:
                operands.append(_ARRAY_MARKER)
                continue

            if token == "]":
                try:
                    marker_index = len(operands) - 1 - operands[::-1].index(_ARRAY_MARKER)
                except ValueError:
                    continue
                array = operands[marker_index + 1 :]
                operands = operands[:marker_index]
                operands.append(array)
                continue

            # Tipos de operando válidos.
            if isinstance(token, (_Name, bytes, int, float, list)):
                operands.append(token)
                continue

            # Bare word = operador de content stream.
            if isinstance(token, str):
                event = self._handle_operator(int(xref), token, operands)
                if event is not None:
                    events.append(event)
                operands = []

        return events


def extract_page_text_events(
    doc: fitz.Document,
    page_index: int = 0,
    *,
    text_encoding: str = "cp1252",
) -> list[PdfTextEvent]:
    """Extrai eventos textuais dos content streams na ordem da página.

    Os streams em ``/Contents`` são tratados como concatenados: estado gráfico e
    estado textual persistem entre xrefs, como exige a semântica do PDF.
    """

    page = doc[page_index]
    parser = PdfContentTextParser(
        pdf_to_page=page.transformation_matrix,
        text_encoding=text_encoding,
    )

    contents = page.get_contents()
    if isinstance(contents, int):
        xrefs = [contents]
    else:
        xrefs = list(contents or [])

    events: list[PdfTextEvent] = []
    for xref in xrefs:
        stream = doc.xref_stream(int(xref))
        events.extend(parser.parse_stream(stream, xref=int(xref)))
    return events


__all__ = [
    "PdfContentTextParser",
    "PdfTextEvent",
    "extract_page_text_events",
]
