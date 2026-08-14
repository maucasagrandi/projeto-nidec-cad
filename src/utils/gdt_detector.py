"""
gdt_detector.py
---------------
Tópicos 7 e 8 do prompt_classification.md — detecção de quadros GD&T e geração de crops.

Abordagem (V1):
    O PDF do cliente NÃO desenha quadros GD&T como comandos "re" (retângulo).
    Cada quadro é composto por múltiplos segmentos de linha ("l") independentes.
    Portanto, precisamos RECONSTRUIR os retângulos a partir dos segmentos.

Fluxo:
    1. Renderizar a página uma única vez em alta resolução (para crops e debug).
    2. Extrair segmentos horizontais e verticais de page.get_drawings().
    3. Unir segmentos colineares próximos (mesma y para H, mesma x para V).
    4. Para cada par de linhas horizontais próximas verticalmente, encontrar
       linhas verticais que as conectem — isto define um retângulo composto
       por N células.
    5. Filtrar candidatos por tamanho e número de células.
    6. SÓ ENTÃO extrair textos internos, verificando se o CENTRO da palavra
       está dentro da célula. Texto é evidência de confiança, não define
       posição do quadro.
    7. Gerar crops (frame e primeira célula) e imagem de debug.

Observações importantes:
    - Coordenadas PyMuPDF: origem no canto SUPERIOR ESQUERDO (y cresce para baixo).
    - Não classificar semanticamente as células nesta etapa (tópico 9).
    - Falsos positivos são mantidos para calibração — não filtrar por conteúdo textual.
"""

import logging
import re
from dataclasses import dataclass, field
from io import BytesIO
from typing import List, Optional, Tuple

import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


# ==============================================================================
# Estruturas geométricas básicas
# ==============================================================================

@dataclass
class BBox:
    """
    Bounding box em coordenadas PDF.

    PyMuPDF usa sistema com origem no CANTO SUPERIOR ESQUERDO da página.
    Portanto y cresce para BAIXO (y0 é o topo, y1 é a base).
    """
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return abs(self.x1 - self.x0)

    @property
    def height(self) -> float:
        return abs(self.y1 - self.y0)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)

    def to_list(self) -> List[float]:
        return [self.x0, self.y0, self.x1, self.y1]

    def contains_point(self, x: float, y: float) -> bool:
        """Verifica se um ponto está dentro do bbox."""
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1


@dataclass
class HSegment:
    """Segmento horizontal (mesmo y)."""
    y: float
    x0: float
    x1: float

    @property
    def length(self) -> float:
        return self.x1 - self.x0


@dataclass
class VSegment:
    """Segmento vertical (mesmo x)."""
    x: float
    y0: float
    y1: float

    @property
    def length(self) -> float:
        return self.y1 - self.y0


# ==============================================================================
# Modelos de saída
# ==============================================================================

@dataclass
class GdtCell:
    """Célula de um quadro GD&T candidato."""
    bbox: BBox
    texts: List[str] = field(default_factory=list)
    role: str = "unclassified"  # "unclassified" nesta fase — tópico 9 classifica

    def has_numeric_content(self) -> bool:
        if not self.texts:
            return False
        pattern = r"\d+[.,]?\d*"
        return any(re.search(pattern, t) for t in self.texts)

    def has_letter_content(self) -> bool:
        if not self.texts:
            return False
        pattern = r"^[A-Z]$"
        return any(re.match(pattern, t.strip()) for t in self.texts)


@dataclass
class GdtFrameCandidate:
    """
    Candidato a quadro de tolerância geométrica (Tópico 7).

    Reconstruído geometricamente a partir de segmentos de linha.
    Não classificado semanticamente — Tópico 9 fará isso via LLM visual.
    """
    candidate_id: str
    page: int  # 1-indexed
    frame_bbox: BBox
    symbol_bbox: BBox  # primeira célula (a mais à esquerda)
    cells: List[GdtCell]
    frame_crop: Optional[Image.Image] = None
    symbol_crop: Optional[Image.Image] = None
    confidence_score: float = 0.0

    def extract_tolerance_values(self) -> List[str]:
        """Valores numéricos encontrados em qualquer célula (exceto símbolo)."""
        values = []
        for cell in self.cells[1:]:  # pula a primeira (símbolo)
            for t in cell.texts:
                if re.search(r"\d+[.,]?\d+", t):
                    values.append(t)
        return values

    def extract_datum_references(self) -> List[str]:
        """Letras isoladas em qualquer célula (exceto símbolo)."""
        datums = []
        for cell in self.cells[1:]:
            for t in cell.texts:
                if re.match(r"^[A-Z]$", t.strip()):
                    datums.append(t.strip())
        return datums


# ==============================================================================
# Detector
# ==============================================================================

class GdtFrameDetector:
    """
    Reconstrói quadros GD&T a partir de segmentos de linha vetoriais.

    Parâmetros geométricos:
        min_cells / max_cells: número aceitável de células por quadro
        min_frame_height / max_frame_height: altura em pontos PDF
        min_frame_width / max_frame_width: largura total em pontos PDF
        symbol_aspect_min / symbol_aspect_max: proporção largura/altura da primeira célula
        line_tolerance: tolerância para considerar segmentos colineares
        merge_gap: gap máximo para unir segmentos colineares
        endpoint_tolerance: quanto uma linha vertical pode "faltar" para tocar a horizontal
            (padrão 4.5pt — em alguns quadros do cliente as verticais não
            tocam exatamente a horizontal do topo/base, com gaps reais de
            até ~4.3pt observados; 2.0pt deixava esses quadros sem nenhum
            candidato reconstruído)
        page_border_margin: margem da borda da página a ignorar (grade de quadrante)
    """

    def __init__(
        self,
        min_cells: int = 2,
        max_cells: int = 6,
        min_frame_height: float = 5.0,
        max_frame_height: float = 40.0,
        min_frame_width: float = 15.0,
        max_frame_width: float = 300.0,
        min_cell_width: float = 4.0,
        symbol_aspect_min: float = 0.5,
        symbol_aspect_max: float = 2.5,
        line_tolerance: float = 1.0,
        merge_gap: float = 2.0,
        endpoint_tolerance: float = 4.5,
        page_border_margin: float = 30.0,
        crop_dpi: int = 300,
        crop_padding: float = 4.0,
    ):
        self.min_cells = min_cells
        self.max_cells = max_cells
        self.min_frame_height = min_frame_height
        self.max_frame_height = max_frame_height
        self.min_frame_width = min_frame_width
        self.max_frame_width = max_frame_width
        self.min_cell_width = min_cell_width
        self.symbol_aspect_min = symbol_aspect_min
        self.symbol_aspect_max = symbol_aspect_max
        self.line_tolerance = line_tolerance
        self.merge_gap = merge_gap
        self.endpoint_tolerance = endpoint_tolerance
        self.page_border_margin = page_border_margin
        self.crop_dpi = crop_dpi
        self.crop_padding = crop_padding

        # Cache da última detecção (para debug image)
        self._last_page_image: Optional[Image.Image] = None
        self._last_page_rect: Optional[fitz.Rect] = None
        self._last_zoom: float = 1.0

    # --------------------------------------------------------------------------
    # API pública
    # --------------------------------------------------------------------------

    def detect_frames(self, pdf_bytes: bytes, page_index: int = 0) -> List[GdtFrameCandidate]:
        """Detecta quadros GD&T em uma página do PDF."""
        logger.info(f"Detectando quadros GD&T na pagina {page_index + 1}...")

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            if page_index >= len(doc):
                logger.warning(f"Pagina {page_index + 1} nao existe no PDF")
                return []

            page = doc[page_index]

            # 1. Renderizar página uma única vez em alta resolução
            self._last_zoom = self.crop_dpi / 72.0
            mat = fitz.Matrix(self._last_zoom, self._last_zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            self._last_page_image = Image.open(BytesIO(pix.tobytes("png"))).convert("RGB")
            self._last_page_rect = page.rect

            # 2. Extrair segmentos H e V
            h_raw, v_raw = self._extract_line_segments(page)
            logger.info(f"   Segmentos brutos: {len(h_raw)} H, {len(v_raw)} V")

            # 3. Unir colineares
            h_merged = self._merge_horizontal(h_raw)
            v_merged = self._merge_vertical(v_raw)
            logger.info(f"   Apos merge: {len(h_merged)} H, {len(v_merged)} V")

            # 4. Reconstruir quadros retangulares
            raw_frames = self._reconstruct_frames(h_merged, v_merged)
            logger.info(f"   Retangulos com celulas reconstruidos: {len(raw_frames)}")

            # 5. Filtrar candidatos por geometria
            valid_frames = self._filter_by_geometry(raw_frames)
            logger.info(f"   Apos filtro geometrico: {len(valid_frames)}")

            # 5b. Deduplicar por IoU (frames muito sobrepostos que sobreviveram)
            valid_frames = self._dedup_frames_by_iou(valid_frames)
            logger.info(f"   Apos dedup por IoU: {len(valid_frames)}")

            # 6. Extrair textos dentro das células (só agora)
            words = page.get_text("words")
            for _frame_bbox, cells in valid_frames:
                for cell in cells:
                    cell.texts = self._words_in_cell(words, cell.bbox)

            # 7. Calcular confiança e gerar crops
            candidates: List[GdtFrameCandidate] = []
            for idx, (frame_bbox, cells) in enumerate(valid_frames, start=1):
                symbol_bbox = cells[0].bbox
                confidence = self._compute_confidence(cells)

                frame_crop = self._crop_from_page_image(frame_bbox)
                symbol_crop = self._crop_from_page_image(symbol_bbox)

                candidate = GdtFrameCandidate(
                    candidate_id=f"GDT-CAND-P{page_index + 1:02d}-{idx:03d}",
                    page=page_index + 1,
                    frame_bbox=frame_bbox,
                    symbol_bbox=symbol_bbox,
                    cells=cells,
                    frame_crop=frame_crop,
                    symbol_crop=symbol_crop,
                    confidence_score=confidence,
                )
                candidates.append(candidate)

            logger.info(f"Detectados {len(candidates)} quadros GD&T candidatos")
            return candidates

        finally:
            doc.close()

    def render_debug_image(
        self,
        candidates: List[GdtFrameCandidate],
    ) -> Optional[Image.Image]:
        """
        Retorna uma cópia da página renderizada com todos os candidatos marcados.

        Cada quadro é desenhado com contorno vermelho e o candidate_id acima.
        Cada célula é marcada com contorno azul claro. A primeira célula (símbolo)
        recebe destaque adicional.
        """
        if self._last_page_image is None:
            return None

        img = self._last_page_image.copy()
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except (OSError, IOError):
            font = ImageFont.load_default()

        for cand in candidates:
            frame_px = self._bbox_to_pixels(cand.frame_bbox)
            symbol_px = self._bbox_to_pixels(cand.symbol_bbox)

            # Contorno do quadro completo
            draw.rectangle(frame_px, outline=(220, 20, 60), width=3)

            # Contorno de cada célula
            for cell in cand.cells:
                cell_px = self._bbox_to_pixels(cell.bbox)
                draw.rectangle(cell_px, outline=(30, 144, 255), width=1)

            # Destaque na célula do símbolo
            draw.rectangle(symbol_px, outline=(255, 165, 0), width=2)

            # Rótulo
            label = f"{cand.candidate_id} (conf {cand.confidence_score:.2f})"
            text_x = frame_px[0]
            text_y = max(0, frame_px[1] - 18)
            # fundo do texto
            try:
                tw, th = draw.textbbox((0, 0), label, font=font)[2:]
            except AttributeError:
                tw, th = draw.textsize(label, font=font)
            draw.rectangle([text_x, text_y, text_x + tw + 4, text_y + th + 2],
                           fill=(220, 20, 60))
            draw.text((text_x + 2, text_y + 1), label, fill="white", font=font)

        return img

    # --------------------------------------------------------------------------
    # Etapa 2: extração de segmentos
    # --------------------------------------------------------------------------

    def _extract_line_segments(
        self, page: fitz.Page
    ) -> Tuple[List[HSegment], List[VSegment]]:
        """Coleta segmentos horizontais e verticais do desenho vetorial."""
        h_list: List[HSegment] = []
        v_list: List[VSegment] = []

        page_rect = page.rect
        margin = self.page_border_margin

        drawings = page.get_drawings()
        for d in drawings:
            for item in d.get("items", []):
                if item[0] != "l":
                    continue
                p1, p2 = item[1], item[2]
                dx = abs(p2.x - p1.x)
                dy = abs(p2.y - p1.y)

                # Ignora diagonais
                if dx > self.line_tolerance and dy > self.line_tolerance:
                    continue

                if dy <= self.line_tolerance:
                    # Horizontal
                    y = (p1.y + p2.y) / 2
                    x0, x1 = sorted([p1.x, p2.x])
                    # Ignora bordas da folha
                    if y < margin or y > (page_rect.y1 - margin):
                        continue
                    if x1 - x0 <= 0:
                        continue
                    h_list.append(HSegment(y=y, x0=x0, x1=x1))
                elif dx <= self.line_tolerance:
                    # Vertical
                    x = (p1.x + p2.x) / 2
                    y0, y1 = sorted([p1.y, p2.y])
                    if x < margin or x > (page_rect.x1 - margin):
                        continue
                    if y1 - y0 <= 0:
                        continue
                    v_list.append(VSegment(x=x, y0=y0, y1=y1))

        return h_list, v_list

    # --------------------------------------------------------------------------
    # Etapa 3: unir colineares
    # --------------------------------------------------------------------------

    def _merge_horizontal(self, segments: List[HSegment]) -> List[HSegment]:
        """Une segmentos horizontais que estão na mesma y (com tolerância) e próximos."""
        if not segments:
            return []

        # Ordena por y depois x0
        sorted_segs = sorted(segments, key=lambda s: (s.y, s.x0))

        # Agrupa por y (com tolerância line_tolerance)
        groups: List[List[HSegment]] = []
        current = [sorted_segs[0]]
        for seg in sorted_segs[1:]:
            if abs(seg.y - current[-1].y) <= self.line_tolerance:
                current.append(seg)
            else:
                groups.append(current)
                current = [seg]
        groups.append(current)

        merged: List[HSegment] = []
        for group in groups:
            group.sort(key=lambda s: s.x0)
            y_avg = sum(s.y for s in group) / len(group)
            cur = HSegment(y=y_avg, x0=group[0].x0, x1=group[0].x1)
            for seg in group[1:]:
                if seg.x0 <= cur.x1 + self.merge_gap:
                    cur.x1 = max(cur.x1, seg.x1)
                else:
                    merged.append(cur)
                    cur = HSegment(y=y_avg, x0=seg.x0, x1=seg.x1)
            merged.append(cur)

        return merged

    def _merge_vertical(self, segments: List[VSegment]) -> List[VSegment]:
        """Análogo ao _merge_horizontal para verticais."""
        if not segments:
            return []

        sorted_segs = sorted(segments, key=lambda s: (s.x, s.y0))

        groups: List[List[VSegment]] = []
        current = [sorted_segs[0]]
        for seg in sorted_segs[1:]:
            if abs(seg.x - current[-1].x) <= self.line_tolerance:
                current.append(seg)
            else:
                groups.append(current)
                current = [seg]
        groups.append(current)

        merged: List[VSegment] = []
        for group in groups:
            group.sort(key=lambda s: s.y0)
            x_avg = sum(s.x for s in group) / len(group)
            cur = VSegment(x=x_avg, y0=group[0].y0, y1=group[0].y1)
            for seg in group[1:]:
                if seg.y0 <= cur.y1 + self.merge_gap:
                    cur.y1 = max(cur.y1, seg.y1)
                else:
                    merged.append(cur)
                    cur = VSegment(x=x_avg, y0=seg.y0, y1=seg.y1)
            merged.append(cur)

        return merged

    # --------------------------------------------------------------------------
    # Etapa 4: reconstruir retângulos com células
    # --------------------------------------------------------------------------

    def _reconstruct_frames(
        self,
        h_lines: List[HSegment],
        v_lines: List[VSegment],
    ) -> List[Tuple[BBox, List[GdtCell]]]:
        """
        Para cada par (topo, base) de linhas horizontais que estão numa
        distância vertical típica de quadro GD&T, procura as linhas verticais
        que as conectam. As verticais definem as fronteiras das células.

        Retorna lista de (frame_bbox, cells).
        """
        results: List[Tuple[BBox, List[GdtCell]]] = []

        # Ordena horizontais por y
        h_sorted = sorted(h_lines, key=lambda s: s.y)

        # Índice: para cada horizontal, quais verticais podem estar conectadas?
        # Uma vertical v está conectada a horizontais (top, bottom) se:
        #  - v.x está no range [max(top.x0, bottom.x0), min(top.x1, bottom.x1)]
        #  - v.y0 <= top.y + tol e v.y1 >= bottom.y - tol
        tol = self.endpoint_tolerance

        for i, top in enumerate(h_sorted):
            for j in range(i + 1, len(h_sorted)):
                bottom = h_sorted[j]
                dy = bottom.y - top.y
                if dy < self.min_frame_height:
                    continue
                if dy > self.max_frame_height:
                    break  # dy só cresce daqui, pode encerrar

                # Overlap horizontal entre topo e base
                x_left = max(top.x0, bottom.x0)
                x_right = min(top.x1, bottom.x1)
                if x_right - x_left < self.min_frame_width:
                    continue

                # Verticais que conectam topo e base dentro do overlap
                connectors: List[VSegment] = []
                for v in v_lines:
                    if v.x < x_left - tol or v.x > x_right + tol:
                        continue
                    if v.y0 > top.y + tol:
                        continue
                    if v.y1 < bottom.y - tol:
                        continue
                    connectors.append(v)

                if len(connectors) < self.min_cells + 1:
                    continue

                # Ordena verticais por x
                connectors.sort(key=lambda v: v.x)

                # Remove verticais muito próximas (duplicatas colineares residuais)
                dedup: List[VSegment] = [connectors[0]]
                for v in connectors[1:]:
                    if v.x - dedup[-1].x >= self.min_cell_width:
                        dedup.append(v)

                if len(dedup) < self.min_cells + 1:
                    continue
                if len(dedup) > self.max_cells + 1:
                    # Ainda pode ser um GD&T se restringirmos ao overlap,
                    # mas frames com >6 células são raros; deixa como falso positivo
                    # apenas se ficar dentro do limite após dedup
                    continue

                # Constrói bbox do frame usando primeiro e último vertical
                frame_bbox = BBox(
                    x0=dedup[0].x,
                    y0=top.y,
                    x1=dedup[-1].x,
                    y1=bottom.y,
                )

                # Constrói células
                cells: List[GdtCell] = []
                for k in range(len(dedup) - 1):
                    cell_bbox = BBox(
                        x0=dedup[k].x,
                        y0=top.y,
                        x1=dedup[k + 1].x,
                        y1=bottom.y,
                    )
                    cells.append(GdtCell(bbox=cell_bbox))

                results.append((frame_bbox, cells))

        return results

    # --------------------------------------------------------------------------
    # Etapa 5: filtro geométrico
    # --------------------------------------------------------------------------

    def _filter_by_geometry(
        self, frames: List[Tuple[BBox, List[GdtCell]]]
    ) -> List[Tuple[BBox, List[GdtCell]]]:
        """Aplica filtros de tamanho e proporção."""
        valid: List[Tuple[BBox, List[GdtCell]]] = []
        seen_signatures: set = set()

        for frame_bbox, cells in frames:
            w, h = frame_bbox.width, frame_bbox.height

            if not (self.min_frame_height <= h <= self.max_frame_height):
                continue
            if not (self.min_frame_width <= w <= self.max_frame_width):
                continue
            if not (self.min_cells <= len(cells) <= self.max_cells):
                continue

            # Primeira célula (símbolo) deve ter proporção razoável
            first = cells[0]
            aspect = first.bbox.width / max(first.bbox.height, 1e-6)
            if not (self.symbol_aspect_min <= aspect <= self.symbol_aspect_max):
                continue

            # Deduplica quadros exatamente sobrepostos
            sig = (round(frame_bbox.x0, 1), round(frame_bbox.y0, 1),
                   round(frame_bbox.x1, 1), round(frame_bbox.y1, 1),
                   len(cells))
            if sig in seen_signatures:
                continue
            seen_signatures.add(sig)

            valid.append((frame_bbox, cells))

        # Remove candidatos "encaixados" (menores dentro de maiores com mesmo topo/base)
        valid = self._remove_nested(valid)
        return valid

    def _dedup_frames_by_iou(
        self,
        frames: List[Tuple[BBox, List[GdtCell]]],
        iou_threshold: float = 0.5,
        ios_threshold: float = 0.9,
    ) -> List[Tuple[BBox, List[GdtCell]]]:
        """
        Remove frames com alta sobreposicao (IoU) ou contencao quase total.

        Complementa _remove_nested (que so pega contencao com pequena folga).
        Prioriza frames com mais celulas; empate resolvido por maior area.
        """
        if not frames:
            return []

        def _iou(a: BBox, b: BBox) -> float:
            ix0 = max(a.x0, b.x0)
            iy0 = max(a.y0, b.y0)
            ix1 = min(a.x1, b.x1)
            iy1 = min(a.y1, b.y1)
            iw = max(0.0, ix1 - ix0)
            ih = max(0.0, iy1 - iy0)
            inter = iw * ih
            if inter <= 0:
                return 0.0
            union = a.area + b.area - inter
            return inter / union if union > 0 else 0.0

        def _ios(a: BBox, b: BBox) -> float:
            ix0 = max(a.x0, b.x0)
            iy0 = max(a.y0, b.y0)
            ix1 = min(a.x1, b.x1)
            iy1 = min(a.y1, b.y1)
            iw = max(0.0, ix1 - ix0)
            ih = max(0.0, iy1 - iy0)
            inter = iw * ih
            smaller = min(a.area, b.area)
            return inter / smaller if (inter > 0 and smaller > 0) else 0.0

        # Ranking: mais celulas > maior area
        ranked = sorted(
            frames,
            key=lambda fc: (len(fc[1]), fc[0].area),
            reverse=True,
        )

        kept: List[Tuple[BBox, List[GdtCell]]] = []
        for cand in ranked:
            cb = cand[0]
            is_dup = False
            for k in kept:
                if _iou(cb, k[0]) >= iou_threshold:
                    is_dup = True
                    break
                if _ios(cb, k[0]) >= ios_threshold:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(cand)

        kept.sort(key=lambda fc: (fc[0].y0, fc[0].x0))
        return kept

    def _remove_nested(
        self, frames: List[Tuple[BBox, List[GdtCell]]]
    ) -> List[Tuple[BBox, List[GdtCell]]]:
        """Remove frames que são subconjuntos de outros frames maiores."""
        if not frames:
            return []
        # Ordena por área desc — preserva os maiores
        sorted_frames = sorted(frames, key=lambda f: -f[0].area)
        kept: List[Tuple[BBox, List[GdtCell]]] = []
        for cand in sorted_frames:
            cb = cand[0]
            is_nested = False
            for k in kept:
                kb = k[0]
                # Está totalmente dentro de outro?
                if (kb.x0 - 0.5 <= cb.x0 and kb.y0 - 0.5 <= cb.y0
                        and cb.x1 <= kb.x1 + 0.5 and cb.y1 <= kb.y1 + 0.5):
                    is_nested = True
                    break
            if not is_nested:
                kept.append(cand)
        # Restaura ordem original (esquerda-topo primeiro)
        kept.sort(key=lambda f: (f[0].y0, f[0].x0))
        return kept

    # --------------------------------------------------------------------------
    # Etapa 6: extração textual (centro da palavra na célula)
    # --------------------------------------------------------------------------

    def _words_in_cell(self, words: list, cell_bbox: BBox) -> List[str]:
        """
        Retorna textos cujo CENTRO está dentro da célula.

        words vem de page.get_text("words"):
            (x0, y0, x1, y1, "text", block_no, line_no, word_no)
        """
        texts: List[str] = []
        for w in words:
            wx = (w[0] + w[2]) / 2
            wy = (w[1] + w[3]) / 2
            if cell_bbox.contains_point(wx, wy):
                text = w[4].strip()
                if text:
                    texts.append(text)
        return texts

    # --------------------------------------------------------------------------
    # Etapa 7: confiança e crops
    # --------------------------------------------------------------------------

    def _compute_confidence(self, cells: List[GdtCell]) -> float:
        """
        Heurística de confiança combinando geometria e presença textual.

        Base 0.5 (geometria válida). Presença de valor numérico ou letra
        isolada dentro das células (não símbolo) aumenta a confiança.
        """
        score = 0.5
        if 3 <= len(cells) <= 4:
            score += 0.15
        has_numeric = any(c.has_numeric_content() for c in cells[1:])
        has_letter = any(c.has_letter_content() for c in cells[1:])
        if has_numeric:
            score += 0.25
        if has_letter:
            score += 0.10
        return min(score, 1.0)

    def _bbox_to_pixels(self, bbox: BBox) -> Tuple[int, int, int, int]:
        """Converte bbox PDF para coordenadas em pixels (usa self._last_zoom)."""
        z = self._last_zoom
        return (
            int(round(bbox.x0 * z)),
            int(round(bbox.y0 * z)),
            int(round(bbox.x1 * z)),
            int(round(bbox.y1 * z)),
        )

    def _crop_from_page_image(self, bbox: BBox) -> Optional[Image.Image]:
        """Recorta o bbox da imagem já renderizada (sem re-renderizar PDF)."""
        if self._last_page_image is None:
            return None
        pad = self.crop_padding
        padded = BBox(
            x0=bbox.x0 - pad,
            y0=bbox.y0 - pad,
            x1=bbox.x1 + pad,
            y1=bbox.y1 + pad,
        )
        left, top, right, bottom = self._bbox_to_pixels(padded)
        left = max(0, left)
        top = max(0, top)
        right = min(self._last_page_image.width, right)
        bottom = min(self._last_page_image.height, bottom)
        if right <= left or bottom <= top:
            return None
        return self._last_page_image.crop((left, top, right, bottom))


# ==============================================================================
# Função de conveniência
# ==============================================================================

def detect_gdt_frames(
    pdf_bytes: bytes,
    page_index: int = 0,
    **detector_kwargs,
) -> Tuple[List[GdtFrameCandidate], Optional[Image.Image]]:
    """
    Detecta quadros GD&T e retorna também a imagem de debug.

    Returns:
        (candidates, debug_image)
    """
    detector = GdtFrameDetector(**detector_kwargs)
    candidates = detector.detect_frames(pdf_bytes, page_index)
    debug = detector.render_debug_image(candidates)
    return candidates, debug
