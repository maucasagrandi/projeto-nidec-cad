"""Pinta no desenho CAD as regiões de quadrante que a LLM já reporta na tabela.

Não faz nenhuma chamada à LLM. Trabalha inteiramente sobre texto que já existe na
saída atual do system_prompt (coluna "Localização (Quadrante)"), como:

    "Central (D4 a E7)"
    "Canto Inferior Direito (A1, B1)"
    "Vários Quadrantes (F4-F7, C9-E9, B8-D8)"
    "Central / Vistas de Detalhe"                (sem quadrante localizável)

Duas etapas independentes:

1. extract_grid(): lê a grade de zoneamento vetorial impressa nas bordas do PDF
   (colunas numéricas, linhas em letra) e converte rótulo -> coordenada real.
2. parse_quadrant_text(): extrai da string livre da LLM os tokens de quadrante
   (célula única, intervalo com "-"/"a", ou lista separada por vírgula) e resolve
   cada um em um retângulo usando a grade.

paint_quadrants() combina as duas para desenhar as regiões sobre a imagem revisada.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont


# ==============================================================================
# Grade de zoneamento
# ==============================================================================

EDGE_FRACTION = 0.13  # fração da borda onde se procuram os rótulos da grade
STEP_TOLERANCE = 0.06
MIN_LABELS = 3


@dataclass(frozen=True)
class GridInfo:
    """Grade de zoneamento de uma página, em pontos PDF (72 pt = 1 polegada)."""

    columns: dict[str, float]  # rótulo (dígito) -> x do centro
    rows: dict[str, float]  # rótulo (letra) -> y do centro
    column_step: float
    row_step: float
    page_width: float
    page_height: float


def _edge_label_candidates(page: "fitz.Page"):
    """Candidatos a rótulo de grade nas bordas: dígito -> coluna, letra -> linha."""
    width, height = page.rect.width, page.rect.height
    columns, rows = [], []

    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                text = span["text"].strip()
                if not text or len(text) > 2 or not text.isalnum():
                    continue
                x0, y0, x1, y1 = span["bbox"]
                cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
                near_h = cy < height * EDGE_FRACTION or cy > height * (1 - EDGE_FRACTION)
                near_v = cx < width * EDGE_FRACTION or cx > width * (1 - EDGE_FRACTION)
                if near_h and text.isdigit():
                    columns.append((text, cx))
                if near_v and text.isalpha():
                    rows.append((text.upper(), cy))
    return columns, rows


def _best_progression(candidates: list[tuple[str, float]]):
    """Maior subconjunto de rótulos regularmente espaçados (descarta ruído)."""
    unique = sorted({(l, round(c, 1)) for l, c in candidates}, key=lambda p: p[1])
    n = len(unique)
    if n < 2:
        return {}, None

    best_chain, best_step = [], None
    for i in range(n):
        for j in range(i + 1, n):
            step = unique[j][1] - unique[i][1]
            if step <= 1.0:
                continue
            chain = [unique[i], unique[j]]
            used = {unique[i][0], unique[j][0]}
            cursor = unique[j][1]
            tol = step * STEP_TOLERANCE
            extended = True
            while extended:
                extended = False
                for label, coord in unique:
                    if label in used:
                        continue
                    if abs(coord - (cursor + step)) <= tol:
                        chain.append((label, coord))
                        used.add(label)
                        cursor = coord
                        extended = True
                        break
            if len(chain) > len(best_chain):
                best_chain, best_step = chain, step

    if len(best_chain) < 2:
        return {}, None
    return {l: c for l, c in best_chain}, best_step


def extract_grid_from_page(page: "fitz.Page") -> Optional[GridInfo]:
    """Extrai a grade de zoneamento de uma página. None se não detectada."""
    col_cands, row_cands = _edge_label_candidates(page)
    columns, col_step = _best_progression(col_cands)
    rows, row_step = _best_progression(row_cands)

    if len(columns) < MIN_LABELS or len(rows) < MIN_LABELS:
        return None

    return GridInfo(
        columns=columns,
        rows=rows,
        column_step=col_step or 0.0,
        row_step=row_step or 0.0,
        page_width=page.rect.width,
        page_height=page.rect.height,
    )


def extract_grid(pdf_bytes: bytes, page_index: int = 0) -> Optional[GridInfo]:
    """Abre o PDF em memória e extrai a grade da página indicada."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        if page_index >= len(doc):
            return None
        return extract_grid_from_page(doc[page_index])
    finally:
        doc.close()


def cell_rect_pt(grid: GridInfo, row: str, col: str):
    """Retângulo (x0,y0,x1,y1) em pt da célula, ou None se rótulo não existe."""
    if row not in grid.rows or col not in grid.columns:
        return None
    cx, cy = grid.columns[col], grid.rows[row]
    hw, hh = grid.column_step / 2.0, grid.row_step / 2.0
    return (
        max(0.0, cx - hw),
        max(0.0, cy - hh),
        min(grid.page_width, cx + hw),
        min(grid.page_height, cy + hh),
    )


# ==============================================================================
# Parsing do texto livre de quadrante
# ==============================================================================

_TOKEN = r"(?:[A-Za-z]\d{1,2}|\d{1,2}[A-Za-z])"
_RANGE_RE = re.compile(rf"\b({_TOKEN})\s*(?:-|to|a)\s*({_TOKEN})\b", re.IGNORECASE)
_TOKEN_RE = re.compile(rf"\b({_TOKEN})\b")


def _normalize_token(token: str) -> tuple[str, str]:
    """Token 'D4' ou '4D' -> (linha='D', coluna='4'), independente da ordem."""
    m = re.match(r"^([A-Za-z])(\d{1,2})$", token)
    if m:
        return m.group(1).upper(), m.group(2)
    m = re.match(r"^(\d{1,2})([A-Za-z])$", token)
    return m.group(2).upper(), m.group(1)


def parse_quadrant_text(texto: str) -> list[tuple[tuple[str, str], Optional[tuple[str, str]]]]:
    """Extrai grupos de células do texto livre de localização.

    Cada grupo é (célula_inicial, célula_final_ou_None). célula_final presente
    significa intervalo (conectado por '-', 'to' ou 'a'); None significa célula única.
    Tokens separados por vírgula são grupos independentes de célula única.

    Retorna lista vazia quando nenhum token de quadrante é reconhecível no texto
    (ex: "Central / Vistas de Detalhe").
    """
    grupos: list[tuple[tuple[str, str], Optional[tuple[str, str]]]] = []
    spans_usados: list[tuple[int, int]] = []

    for m in _RANGE_RE.finditer(texto):
        grupos.append((_normalize_token(m.group(1)), _normalize_token(m.group(2))))
        spans_usados.append(m.span())

    restante = list(texto)
    for s, e in spans_usados:
        for i in range(s, e):
            restante[i] = " "
    restante_str = "".join(restante)

    for m in _TOKEN_RE.finditer(restante_str):
        grupos.append((_normalize_token(m.group(1)), None))

    return grupos


def group_bbox_pt(grid: GridInfo, grupo) -> Optional[tuple[float, float, float, float]]:
    """Bbox em pt do grupo (célula única ou intervalo). None se rótulo inexistente."""
    inicio, fim = grupo
    rect_a = cell_rect_pt(grid, *inicio)
    rect_b = cell_rect_pt(grid, *fim) if fim else rect_a
    if rect_a is None or rect_b is None:
        return None
    return (
        min(rect_a[0], rect_b[0]),
        min(rect_a[1], rect_b[1]),
        max(rect_a[2], rect_b[2]),
        max(rect_a[3], rect_b[3]),
    )


def text_bboxes_pt(texto: str, grid: GridInfo) -> list[tuple[float, float, float, float]]:
    """Todos os retângulos (em pt) que o texto de localização resolve na grade."""
    bboxes = []
    for grupo in parse_quadrant_text(texto):
        bbox = group_bbox_pt(grid, grupo)
        if bbox is not None:
            bboxes.append(bbox)
    return bboxes


def bbox_pt_to_px(bbox_pt, dpi: int) -> tuple[int, int, int, int]:
    """Converte um retângulo de pontos PDF para pixels na resolução informada."""
    scale = dpi / 72.0
    x0, y0, x1, y1 = bbox_pt
    return int(x0 * scale), int(y0 * scale), int(x1 * scale), int(y1 * scale)


# ==============================================================================
# Parsing da tabela Markdown já produzida pela LLM
# ==============================================================================

def parse_markdown_table(markdown_text: str) -> list[dict[str, str]]:
    """Extrai a tabela Markdown em lista de dicts {cabeçalho: valor}.

    Mesma lógica de detecção de linha de tabela já usada em front.py: linha começa
    e termina com '|'; a linha separadora (só '-', '|' e espaço) é descartada; a
    primeira linha restante é o cabeçalho.
    """
    linhas_tabela = []
    for linha in markdown_text.split("\n"):
        s = linha.strip()
        if s.startswith("|") and s.endswith("|"):
            if all(c in "-|: " for c in s):
                continue
            linhas_tabela.append(s)

    if not linhas_tabela:
        return []

    def celulas(linha: str) -> list[str]:
        return [c.strip() for c in linha.split("|")[1:-1]]

    cabecalho = celulas(linhas_tabela[0])
    registros = []
    for linha in linhas_tabela[1:]:
        valores = celulas(linha)
        registro = {}
        for i, chave in enumerate(cabecalho):
            registro[chave] = valores[i] if i < len(valores) else ""
        registros.append(registro)
    return registros


def encontrar_coluna(cabecalho_dict: dict[str, str], *pistas: str) -> Optional[str]:
    """Acha a chave do dict cujo texto contém alguma das pistas (case-insensitive)."""
    for chave in cabecalho_dict:
        baixo = chave.lower()
        if any(p.lower() in baixo for p in pistas):
            return chave
    return None


# ==============================================================================
# Pintura sobre a imagem
# ==============================================================================

@dataclass(frozen=True)
class PaintedRegion:
    """Uma região pintada, para montar a legenda ao lado da imagem."""

    item: str
    quadrante_texto: str
    bboxes_px: list[tuple[int, int, int, int]]
    resolvido: bool  # True se ao menos um bbox foi encontrado na grade


_CORES = [
    (230, 60, 60), (60, 140, 230), (60, 190, 90),
    (230, 160, 40), (170, 70, 220), (40, 190, 190),
]


def paint_quadrants(
    imagem: Image.Image,
    itens: list[tuple[str, str]],
    grid: GridInfo,
    dpi: int,
    status_list: Optional[list[str]] = None,
) -> tuple[Image.Image, list[PaintedRegion]]:
    """Desenha as regiões de cada item (item, texto_localização) sobre a imagem.

    Args:
        imagem: imagem revisada rasterizada na resolução `dpi`.
        itens: lista de (rótulo_do_item, texto_de_localização) vindos da tabela.
        grid: grade extraída do PDF.
        dpi: resolução em que `imagem` foi rasterizada.
        status_list: lista opcional de strings de status para cada item da tabela.
            Quando fornecida, a cor de cada região reflete o status semântico:
            - "Aprovado"               → verde
            - "Aprovado com Observação"→ laranja/âmbar
            - "Requer Correção"        → vermelho
            Quando ausente, usa o esquema de cores por índice.

    Returns:
        (imagem anotada, lista de PaintedRegion na mesma ordem de `itens`).
    """
    anotada = imagem.convert("RGB").copy()
    try:
        fonte = ImageFont.truetype("arialbd.ttf", size=max(40, imagem.width // 45))
    except OSError:
        try:
            fonte = ImageFont.truetype("arial.ttf", size=max(40, imagem.width // 45))
        except OSError:
            fonte = ImageFont.load_default()

    regioes: list[PaintedRegion] = []
    pendentes: list[tuple[tuple[int, int, int, int], tuple[int, int, int], str]] = []

    for idx, (item, texto) in enumerate(itens):
        cor = _CORES[idx % len(_CORES)]
        bboxes_pt = text_bboxes_pt(texto, grid)
        bboxes_px = [bbox_pt_to_px(b, dpi) for b in bboxes_pt]

        for bbox in bboxes_px:
            pendentes.append((bbox, cor, str(item)))

        regioes.append(
            PaintedRegion(item=item, quadrante_texto=texto, bboxes_px=bboxes_px, resolvido=bool(bboxes_px))
        )

    # Pinta cada retângulo apenas na sua própria região recortada, evitando alocar
    # um overlay RGBA do tamanho da página inteira (causava MemoryError em páginas
    # grandes rasterizadas em alta resolução, ex: 300 dpi em folhas A0/A1).
    desenho_base = ImageDraw.Draw(anotada, "RGBA")
    for (x0, y0, x1, y1), cor, item in pendentes:
        largura, altura = x1 - x0, y1 - y0
        if largura <= 0 or altura <= 0:
            continue
        recorte = anotada.crop((x0, y0, x1, y1)).convert("RGBA")
        preenchimento = Image.new("RGBA", recorte.size, (*cor, 70))
        recorte = Image.alpha_composite(recorte, preenchimento).convert("RGB")
        anotada.paste(recorte, (x0, y0))
        desenho_base.rectangle((x0, y0, x1, y1), outline=(*cor, 255), width=4)

        # Rótulo grande, centrado no meio do quadrante (não no canto), para ser
        # legível de longe. Se o item tiver vários quadrantes, cada um recebe seu
        # próprio rótulo centrado, já que a mudança pode estar espalhada.
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        tx0, ty0, tx1, ty1 = desenho_base.textbbox((0, 0), item, font=fonte)
        text_w, text_h = tx1 - tx0, ty1 - ty0
        pos = (cx - text_w / 2.0, cy - text_h / 2.0)
        # Contorno branco fino para o número se destacar sobre linhas do CAD.
        for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            desenho_base.text((pos[0] + dx, pos[1] + dy), item, fill=(255, 255, 255, 230), font=fonte)
        desenho_base.text(pos, item, fill=(*cor, 255), font=fonte)

    return anotada, regioes


def paint_single_item(
    imagem: Image.Image,
    item: str,
    texto_localizacao: str,
    grid: GridInfo,
    dpi: int,
    status: str = "",
) -> Image.Image:
    """Pinta apenas um item (ID) na imagem. Usado nos blocos por ID do relatório.

    Mesma lógica de paint_quadrants, mas para um único item e sem retornar
    metadados — só a imagem anotada. Se o texto não resolver nenhum quadrante
    na grade, retorna a imagem original sem modificação.

    Args:
        imagem: imagem do CAD (original ou revisado) rasterizada em `dpi`.
        item: rótulo do item (ex: "1", "2").
        texto_localizacao: texto da coluna Localização (Quadrante) para este item.
        grid: grade extraída do PDF correspondente a esta imagem.
        dpi: resolução em que `imagem` foi rasterizada.
        status: string de status do item para coloração semântica (opcional).

    Returns:
        Imagem RGB com o item pintado (ou inalterada se quadrante não localizável).
    """
    anotada, _ = paint_quadrants(
        imagem,
        [(item, texto_localizacao)],
        grid,
        dpi,
        status_list=[status] if status else None,
    )
    return anotada
