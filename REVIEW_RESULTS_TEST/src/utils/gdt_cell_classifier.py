"""
gdt_cell_classifier.py
------------------------
Segunda geração do experimento de template matching GD&T.

Diferença fundamental em relação a gdt_template_matcher.py (v1):
    v1: varria a PÁGINA INTEIRA em multiescala procurando cada template,
        depois tentava validar estruturalmente cada match encontrado.
        Resultado: centenas de falsos positivos, porque qualquer traço
        pequeno parecido com o template "achava" uma célula por perto.

    v2 (este módulo): usa os RETÂNGULOS E CÉLULAS RECONSTRUÍDOS PELO
        DETECTOR GEOMÉTRICO ANTES DO FILTRO (_reconstruct_frames, que já
        produz células com bbox exato, sem aplicar min_cells/max_cells/
        aspect ratio/etc.) como regiões de interesse. Cada célula já É uma
        estrutura geometricamente válida (limitada por linhas reais) — não
        precisamos mais "validar estrutura" depois do match, porque a
        estrutura já existe antes do match. O problema deixa de ser
        DETECÇÃO (onde está o símbolo?) e passa a ser CLASSIFICAÇÃO
        (qual símbolo, se algum, está nesta célula já conhecida?).

Fluxo:
    1. Extrair todas as células reconstruídas (pré-filtro geométrico),
       deduplicadas por bbox.
    2. Para cada célula:
        a. recortar da imagem renderizada;
        b. remover as bordas do retângulo (inset);
        c. normalizar polaridade (traços pretos em fundo branco);
        d. canonicalizar (resize preservando proporção para tamanho fixo);
        e. derivar gray / binary / edges;
        f. comparar contra os 3 templates (idem canonicalizados) nas 3
           representações via cv2.matchTemplate com pequena janela de
           busca (compensa deslocamento de centralização);
        g. registrar melhor e segundo melhor score por representação;
        h. aceitar somente se score >= min_score E margem >= min_margin.
    3. Fallback: rodar o matching de página inteira (gdt_template_matcher)
       apenas nas regiões NÃO cobertas por nenhuma célula reconstruída.
    4. Gerar contact sheet das células classificadas + tabela comparativa
       gray/binary/edges + JSON.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import fitz  # PyMuPDF
import numpy as np

# ==============================================================================
# Configuração global de canonicalização (template matching)
# ==============================================================================
# Altere via variável de ambiente GDT_TARGET_SIZE ou GDT_CELL_INSET_FRACTION,
# ou modifique os valores padrão aqui

TEST_NUMBER = 3
GDT_TARGET_SIZE = int(os.getenv("GDT_TARGET_SIZE", "32"))
GDT_CELL_INSET_FRACTION = float(os.getenv("GDT_CELL_INSET_FRACTION", "0.05"))
from PIL import Image, ImageDraw, ImageFont

from src.utils.gdt_detector import BBox, GdtFrameDetector
from src.utils.gdt_template_matcher import (
    DEFAULT_SCALES,
    PreparedTemplate,
    RenderedPage,
    TemplateSpec,
    find_raw_matches,
    non_max_suppression,
    prepare_template,
    render_page,
)

logger = logging.getLogger(__name__)

REPRESENTATIONS = ("gray", "binary", "edges")


# ==============================================================================
# Etapa 1: extração de células reconstruídas (pré-filtro geométrico)
# ==============================================================================

@dataclass
class CandidateCell:
    """Uma célula candidata extraída ANTES do filtro geométrico do detector."""
    cell_id: str
    bbox: BBox            # bbox da célula em coordenadas PDF (pt)
    frame_bbox: BBox       # bbox do quadro completo a que pertence
    cell_index: int        # posição da célula dentro do quadro (0 = mais à esquerda)
    num_cells_in_frame: int
    is_first_cell: bool    # convenção GD&T: primeira célula = símbolo


def extract_unique_raw_cells(
    pdf_bytes: bytes,
    page_index: int = 0,
    detector: Optional[GdtFrameDetector] = None,
    dedup_tolerance: float = 0.5,
) -> Tuple[List[CandidateCell], List[object], List[object]]:
    """
    Reconstrói quadros/células ANTES do filtro geométrico
    (_filter_by_geometry) e retorna a lista de células únicas.

    Retorna (cells, h_lines, v_lines) — h_lines/v_lines são devolvidos para
    reuso opcional (ex: fallback ou depuração adicional).
    """
    detector = detector or GdtFrameDetector()

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        page = doc[page_index]
        h_raw, v_raw = detector._extract_line_segments(page)
        h_lines = detector._merge_horizontal(h_raw)
        v_lines = detector._merge_vertical(v_raw)
        raw_frames = detector._reconstruct_frames(h_lines, v_lines)
    finally:
        doc.close()

    logger.info(f"Quadros reconstruidos (pre-filtro): {len(raw_frames)}")

    seen: set = set()
    cells: List[CandidateCell] = []
    counter = 0

    for frame_bbox, frame_cells in raw_frames:
        for idx, cell in enumerate(frame_cells):
            sig = (
                round(cell.bbox.x0 / dedup_tolerance) * dedup_tolerance,
                round(cell.bbox.y0 / dedup_tolerance) * dedup_tolerance,
                round(cell.bbox.x1 / dedup_tolerance) * dedup_tolerance,
                round(cell.bbox.y1 / dedup_tolerance) * dedup_tolerance,
            )
            if sig in seen:
                continue
            seen.add(sig)
            counter += 1
            cells.append(CandidateCell(
                cell_id=f"CELL-{page_index + 1:02d}-{counter:04d}",
                bbox=cell.bbox,
                frame_bbox=frame_bbox,
                cell_index=idx,
                num_cells_in_frame=len(frame_cells),
                is_first_cell=(idx == 0),
            ))

    logger.info(f"Celulas unicas (dedup): {len(cells)}")
    return cells, h_lines, v_lines


# ==============================================================================
# Etapa 2a-2d: recorte, remoção de borda, normalização e canonicalização
# ==============================================================================

def crop_cell_interior(
    page_gray: np.ndarray,
    bbox: BBox,
    zoom: float,
    inset_fraction: float =0.05,
    min_inset_pt: float = 0.6,
    max_inset_pt: float = 3.0,
) -> Optional[np.ndarray]:
    """
    Recorta o INTERIOR de uma célula, removendo as linhas de borda do
    retângulo (que ficam exatamente sobre bbox.x0/x1/y0/y1, pois foi assim
    que a célula foi reconstruída a partir das linhas vetoriais).

    O inset é proporcional ao tamanho da célula (fração), com limites
    mínimo/máximo em pontos PDF, para não remover conteúdo demais em
    células pequenas nem deixar borda residual em células grandes.
    """
    inset_x = min(max(bbox.width * inset_fraction, min_inset_pt), max_inset_pt)
    inset_y = min(max(bbox.height * inset_fraction, min_inset_pt), max_inset_pt)

    x0 = (bbox.x0 + inset_x) * zoom
    y0 = (bbox.y0 + inset_y) * zoom
    x1 = (bbox.x1 - inset_x) * zoom
    y1 = (bbox.y1 - inset_y) * zoom

    px0, py0 = int(round(x0)), int(round(y0))
    px1, py1 = int(round(x1)), int(round(y1))

    px0 = max(0, px0)
    py0 = max(0, py0)
    px1 = min(page_gray.shape[1], px1)
    py1 = min(page_gray.shape[0], py1)

    if px1 - px0 < 3 or py1 - py0 < 3:
        return None

    return page_gray[py0:py1, px0:px1].copy()


def normalize_polarity(gray: np.ndarray) -> np.ndarray:
    """
    Garante traços pretos em fundo branco. Desenhos vetoriais renderizados
    já costumam vir assim, mas esta função corrige o caso oposto (fundo
    escuro) e aplica contrast stretch para uniformizar o contraste entre
    células de diferentes regiões do desenho.
    """
    if gray.size == 0:
        return gray

    # Se a maioria dos pixels for escura, a polaridade esta invertida
    if float(np.mean(gray)) < 127.0:
        gray = 255 - gray

    # Contrast stretch (min-max) para uniformizar
    g_min, g_max = float(gray.min()), float(gray.max())
    if g_max - g_min > 1e-6:
        gray = ((gray.astype(np.float32) - g_min) / (g_max - g_min) * 255.0).astype(np.uint8)

    return gray


def canonicalize(
    gray: np.ndarray,
    target_size: int = 48,
    margin: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Redimensiona preservando a proporção para que a maior dimensão seja
    `target_size`, e retorna duas versões centradas em fundo branco:

        search_canvas: tamanho (target_size + 2*margin) quadrado — usado
                        como IMAGEM DE BUSCA (permite pequeno deslocamento
                        de centralização entre formas de proporções distintas)
        tight_canvas:   tamanho target_size quadrado (sem margem) — usado
                        como TEMPLATE deslizante dentro do search_canvas

    Ambos preservam a proporção original do conteúdo (sem distorção).
    """
    h, w = gray.shape[:2]
    if h == 0 or w == 0:
        blank_search = np.full((target_size + 2 * margin,) * 2, 255, dtype=np.uint8)
        blank_tight = np.full((target_size, target_size), 255, dtype=np.uint8)
        return blank_search, blank_tight

    scale = target_size / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(gray, (new_w, new_h), interpolation=interp)

    tight_canvas = np.full((target_size, target_size), 255, dtype=np.uint8)
    ty0 = (target_size - new_h) // 2
    tx0 = (target_size - new_w) // 2
    tight_canvas[ty0:ty0 + new_h, tx0:tx0 + new_w] = resized

    search_size = target_size + 2 * margin
    search_canvas = np.full((search_size, search_size), 255, dtype=np.uint8)
    sy0 = margin + ty0
    sx0 = margin + tx0
    search_canvas[sy0:sy0 + new_h, sx0:sx0 + new_w] = resized

    return search_canvas, tight_canvas


def derive_representations(gray_canvas: np.ndarray) -> Dict[str, np.ndarray]:
    """Deriva gray/binary/edges de uma imagem já canonicalizada (branco=fundo)."""
    binary = cv2.threshold(gray_canvas, 200, 255, cv2.THRESH_BINARY)[1]
    edges = cv2.Canny(gray_canvas, 50, 150)
    return {"gray": gray_canvas, "binary": binary, "edges": edges}


@dataclass
class CanonicalForms:
    """Formas canônicas (search + tight) por representação, para uma imagem."""
    search: Dict[str, np.ndarray]
    tight: Dict[str, np.ndarray]


def build_canonical_forms(gray: np.ndarray, target_size: int = None, margin: int = 10) -> CanonicalForms:
    """Aplica normalize_polarity + canonicalize + derive_representations."""
    if target_size is None:
        target_size = GDT_TARGET_SIZE
    normalized = normalize_polarity(gray)
    search_gray, tight_gray = canonicalize(normalized, target_size=target_size, margin=margin)
    search_reps = derive_representations(search_gray)
    tight_reps = derive_representations(tight_gray)
    return CanonicalForms(search=search_reps, tight=tight_reps)


# ==============================================================================
# Etapa 2e-2h: comparação contra templates e decisão
# ==============================================================================

@dataclass
class RepresentationScore:
    template_name: str
    score: float


@dataclass
class CellClassification:
    cell: CandidateCell
    scores_by_representation: Dict[str, List[RepresentationScore]]  # ordenado desc
    primary_representation: str = "gray"
    min_score: float = 0.55
    min_margin: float = 0.12

    @property
    def best(self) -> Optional[RepresentationScore]:
        ranked = self.scores_by_representation.get(self.primary_representation)
        return ranked[0] if ranked else None

    @property
    def second_best(self) -> Optional[RepresentationScore]:
        ranked = self.scores_by_representation.get(self.primary_representation)
        return ranked[1] if ranked and len(ranked) > 1 else None

    @property
    def margin(self) -> float:
        if self.best is None or self.second_best is None:
            return self.best.score if self.best else 0.0
        return self.best.score - self.second_best.score

    @property
    def accepted(self) -> bool:
        if self.best is None:
            return False
        return self.best.score >= self.min_score and self.margin >= self.min_margin

    @property
    def agreement_across_representations(self) -> bool:
        """Se as 3 representações concordam sobre qual e o melhor template."""
        tops = set()
        for rep in REPRESENTATIONS:
            ranked = self.scores_by_representation.get(rep)
            if ranked:
                tops.add(ranked[0].template_name)
        return len(tops) == 1

    def to_json(self) -> dict:
        return {
            "cell_id": self.cell.cell_id,
            "cell_bbox": [round(v, 2) for v in self.cell.bbox.to_list()],
            "frame_bbox": [round(v, 2) for v in self.cell.frame_bbox.to_list()],
            "cell_index": self.cell.cell_index,
            "num_cells_in_frame": self.cell.num_cells_in_frame,
            "is_first_cell": self.cell.is_first_cell,
            "accepted": self.accepted,
            "predicted_class": self.best.template_name if self.accepted else None,
            "best_score": round(self.best.score, 4) if self.best else None,
            "second_best_class": self.second_best.template_name if self.second_best else None,
            "second_best_score": round(self.second_best.score, 4) if self.second_best else None,
            "margin": round(self.margin, 4),
            "agreement_across_representations": self.agreement_across_representations,
            "scores_by_representation": {
                rep: [{"template": s.template_name, "score": round(s.score, 4)} for s in scores]
                for rep, scores in self.scores_by_representation.items()
            },
        }


def _match_score(search_canvas: np.ndarray, tight_template: np.ndarray) -> float:
    """
    cv2.matchTemplate(search, template) — como search_canvas tem margem
    extra em relação a tight_template, isso compensa pequenos deslocamentos
    de centralização entre formas com proporções diferentes. Retorna o
    score maximo encontrado na janela de busca.
    """
    th, tw = tight_template.shape[:2]
    sh, sw = search_canvas.shape[:2]
    if th > sh or tw > sw:
        # Template maior que a busca (nao deveria ocorrer com margin>0, mas protege)
        tight_template = cv2.resize(tight_template, (min(tw, sw), min(th, sh)))
    result = cv2.matchTemplate(search_canvas, tight_template, cv2.TM_CCOEFF_NORMED)
    return float(result.max())


def classify_cell(
    cell: CandidateCell,
    cell_forms: CanonicalForms,
    templates_forms: Dict[str, CanonicalForms],
    min_score: float = 0.55,
    min_margin: float = 0.12,
    primary_representation: str = "gray",
) -> CellClassification:
    """
    Compara uma célula (já canonicalizada) contra todos os templates
    (também canonicalizados), nas 3 representações, e retorna o ranking.
    """
    scores_by_representation: Dict[str, List[RepresentationScore]] = {}

    for rep in REPRESENTATIONS:
        cell_search = cell_forms.search[rep]
        scored = []
        for template_name, template_forms in templates_forms.items():
            template_tight = template_forms.tight[rep]
            score = _match_score(cell_search, template_tight)
            scored.append(RepresentationScore(template_name=template_name, score=score))
        scored.sort(key=lambda s: -s.score)
        scores_by_representation[rep] = scored

    return CellClassification(
        cell=cell,
        scores_by_representation=scores_by_representation,
        primary_representation=primary_representation,
        min_score=min_score,
        min_margin=min_margin,
    )


# ==============================================================================
# Orquestração: classificar todas as células reconstruídas
# ==============================================================================

@dataclass
class CellClassificationExperiment:
    rendered: RenderedPage
    cells: List[CandidateCell]
    classifications: List[CellClassification]
    templates_forms: Dict[str, CanonicalForms]
    fallback_matches: List[dict] = field(default_factory=list)


def prepare_templates_canonical(
    template_specs: Sequence[TemplateSpec],
    target_size: int = None,
    margin: int = 10,
) -> Dict[str, CanonicalForms]:
    """Prepara e canonicaliza os templates uma única vez."""
    if target_size is None:
        target_size = GDT_TARGET_SIZE
    result: Dict[str, CanonicalForms] = {}
    for spec in template_specs:
        prepared = prepare_template(spec)
        result[spec.name] = build_canonical_forms(prepared.gray, target_size=target_size, margin=margin)
    return result


def run_cell_classification(
    pdf_bytes: bytes,
    template_specs: Sequence[TemplateSpec],
    page_index: int = 0,
    dpi: int = 300,
    target_size: int = None,
    margin: int = 10,
    min_score: float = 0.55,
    min_margin: float = 0.12,
    primary_representation: str = "gray",
) -> CellClassificationExperiment:
    """
    Pipeline completo: extrai células reconstruídas (pré-filtro), canonicaliza,
    classifica contra os templates, e retorna o experimento completo.
    """
    if target_size is None:
        target_size = GDT_TARGET_SIZE
    logger.info("=" * 70)
    logger.info("CLASSIFICACAO POR CELULA RECONSTRUIDA (pre-filtro geometrico)")
    logger.info("=" * 70)

    rendered = render_page(pdf_bytes, page_index, dpi=dpi)
    logger.info(f"Pagina renderizada: {rendered.gray.shape[1]}x{rendered.gray.shape[0]} px")

    cells, h_lines, v_lines = extract_unique_raw_cells(pdf_bytes, page_index)

    templates_forms = prepare_templates_canonical(template_specs, target_size=target_size, margin=margin)
    logger.info(f"Templates canonicalizados: {list(templates_forms.keys())}")

    classifications: List[CellClassification] = []
    for cell in cells:
        interior = crop_cell_interior(rendered.gray, cell.bbox, rendered.zoom)
        if interior is None:
            continue
        cell_forms = build_canonical_forms(interior, target_size=target_size, margin=margin)
        classification = classify_cell(
            cell, cell_forms, templates_forms,
            min_score=min_score, min_margin=min_margin,
            primary_representation=primary_representation,
        )
        classifications.append(classification)

    n_accepted = sum(1 for c in classifications if c.accepted)
    logger.info(f"Celulas classificadas: {len(classifications)} ({n_accepted} aceitas)")
    for spec in template_specs:
        count = sum(1 for c in classifications if c.accepted and c.best.template_name == spec.name)
        logger.info(f"   {spec.name}: {count} celulas aceitas")

    return CellClassificationExperiment(
        rendered=rendered,
        cells=cells,
        classifications=classifications,
        templates_forms=templates_forms,
    )


# ==============================================================================
# Fallback: matching de página inteira apenas em regiões não cobertas
# ==============================================================================

def run_fullpage_fallback(
    experiment: CellClassificationExperiment,
    template_specs: Sequence[TemplateSpec],
    coverage_margin_pt: float = 3.0,
    score_threshold: float = 0.75,
    scales: Sequence[float] = DEFAULT_SCALES,
) -> List[dict]:
    """
    Roda o matching de página inteira (multiescala) do módulo v1, mas
    DESCARTA qualquer match cujo centro caia dentro (ou próximo, com
    coverage_margin_pt) de alguma célula já reconstruída e classificada.

    Isso restringe o fallback a recuperar apenas símbolos em regiões que o
    detector geométrico não conseguiu reconstruir como célula nenhuma.
    """
    covered_boxes = [c.bbox for c in experiment.cells]

    def is_covered(cx: float, cy: float) -> bool:
        for b in covered_boxes:
            if (b.x0 - coverage_margin_pt <= cx <= b.x1 + coverage_margin_pt
                    and b.y0 - coverage_margin_pt <= cy <= b.y1 + coverage_margin_pt):
                return True
        return False

    fallback_results: List[dict] = []
    for spec in template_specs:
        prepared = prepare_template(spec)
        raw_matches = find_raw_matches(
            experiment.rendered, prepared, scales=scales,
            score_threshold=score_threshold, nms_iou_threshold=0.3,
        )
        uncovered = []
        for m in raw_matches:
            cx, cy = m.symbol_bbox_px.center
            if not is_covered(cx, cy):
                uncovered.append(m)

        logger.info(f"Fallback pagina inteira '{spec.name}': {len(raw_matches)} matches brutos, "
                    f"{len(uncovered)} fora de celulas ja reconstruidas")

        for m in uncovered:
            fallback_results.append({
                "template_class": spec.name,
                "raw_score": round(m.raw_score, 4),
                "scale": m.scale,
                "symbol_bbox": [round(v, 2) for v in m.symbol_bbox_px.to_list()],
            })

    experiment.fallback_matches = fallback_results
    return fallback_results


# ==============================================================================
# Artefatos: contact sheet + tabela comparativa + JSON
# ==============================================================================

def _get_font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except (OSError, IOError):
        return ImageFont.load_default()


def render_annotated_page(
    experiment: CellClassificationExperiment,
    show_rejected: bool = False,
    show_frame_bbox: bool = True,
) -> Image.Image:
    """
    Retorna uma cópia da página ORIGINAL do CAD (renderizada em alta
    resolução) com as células classificadas destacadas diretamente sobre o
    desenho, na posição real onde foram encontradas.

    Cores:
        - verde: célula aceita, rotulada com a classe predita e o score
        - cinza claro (opcional, show_rejected=True): células analisadas
          mas rejeitadas, sem rótulo de classe (apenas contorno fino)

    Isso é diferente da contact sheet (que mostra crops isolados lado a
    lado) — aqui o objetivo é visualizar a distribuição espacial dos
    símbolos encontrados dentro do contexto do desenho completo.
    """
    img = experiment.rendered.pil_image.copy()
    draw = ImageDraw.Draw(img)
    font_label = _get_font(20)
    z = experiment.rendered.zoom

    color_by_class = {
        "profile": (20, 150, 40),
        "position": (30, 110, 220),
        "generic_circle": (200, 30, 160),
    }
    default_accept_color = (20, 150, 40)
    rejected_color = (190, 190, 190)

    if show_rejected:
        for c in experiment.classifications:
            if c.accepted:
                continue
            bx0, by0, bx1, by1 = [v * z for v in c.cell.bbox.to_list()]
            draw.rectangle([bx0, by0, bx1, by1], outline=rejected_color, width=1)

    for c in experiment.classifications:
        if not c.accepted:
            continue

        color = color_by_class.get(c.best.template_name, default_accept_color)
        bx0, by0, bx1, by1 = [v * z for v in c.cell.bbox.to_list()]
        draw.rectangle([bx0, by0, bx1, by1], outline=color, width=3)

        if show_frame_bbox:
            fx0, fy0, fx1, fy1 = [v * z for v in c.cell.frame_bbox.to_list()]
            draw.rectangle([fx0, fy0, fx1, fy1], outline=color, width=1)

        label = f"{c.best.template_name} ({c.best.score:.2f})"
        try:
            tw, th = draw.textbbox((0, 0), label, font=font_label)[2:]
        except AttributeError:
            tw, th = draw.textsize(label, font=font_label)
        label_x = bx0
        label_y = max(0, by0 - th - 6)
        draw.rectangle([label_x, label_y, label_x + tw + 6, label_y + th + 4], fill=color)
        draw.text((label_x + 3, label_y + 2), label, fill="white", font=font_label)

    return img


def save_annotated_page(
    experiment: CellClassificationExperiment,
    output_path: Path,
    show_rejected: bool = False,
    show_frame_bbox: bool = True,
) -> Path:
    """Gera e salva a página do CAD anotada com as células classificadas."""
    img = render_annotated_page(experiment, show_rejected=show_rejected, show_frame_bbox=show_frame_bbox)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    logger.info(f"Pagina do CAD anotada salva: {output_path}")
    return output_path


def generate_classified_cells_contact_sheet(
    experiment: CellClassificationExperiment,
    output_path: Path,
    only_accepted: bool = False,
    cols: int = 6,
    thumb_size: Tuple[int, int] = (150, 110),
) -> Optional[Path]:
    """
    Contact sheet de todas as células (ou apenas as aceitas), mostrando o
    crop original, a classe predita (ou 'no match'), score e margem.
    """
    classifications = experiment.classifications
    if only_accepted:
        classifications = [c for c in classifications if c.accepted]
    if not classifications:
        logger.info("Sem celulas para contact sheet")
        return None

    # Ordena: aceitas primeiro (por classe), depois rejeitadas
    classifications = sorted(
        classifications,
        key=lambda c: (not c.accepted, c.best.template_name if c.best else "zzz", -c.best.score if c.best else 0),
    )

    tw, th = thumb_size
    padding = 8
    header_h = 40
    cell_w = tw + 2 * padding
    cell_h = th + header_h + 2 * padding
    rows = (len(classifications) + cols - 1) // cols

    sheet = Image.new("RGB", (cell_w * cols, cell_h * rows), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    font_id = _get_font(11)
    font_meta = _get_font(9)

    z = experiment.rendered.zoom
    page_img = experiment.rendered.pil_image

    for idx, c in enumerate(classifications):
        row, col = divmod(idx, cols)
        x0 = col * cell_w
        y0 = row * cell_h

        header_color = (20, 140, 40) if c.accepted else (150, 150, 150)
        draw.rectangle([x0, y0, x0 + cell_w, y0 + header_h], fill=header_color)
        draw.text((x0 + padding, y0 + 3), c.cell.cell_id, fill="white", font=font_id)

        label = f"{c.best.template_name}" if c.best else "no match"
        score_txt = f"s={c.best.score:.2f} m={c.margin:.2f}" if c.best else ""
        draw.text((x0 + padding, y0 + 17), f"{label} {score_txt}"[:28], fill="white", font=font_meta)
        pos = "1st" if c.cell.is_first_cell else f"#{c.cell.cell_index}"
        draw.text((x0 + padding, y0 + 28), f"cell {pos}/{c.cell.num_cells_in_frame}", fill="white", font=font_meta)

        margin_px = 10
        bx0, by0, bx1, by1 = [v * z for v in c.cell.bbox.to_list()]
        bx0, by0 = max(0, bx0 - margin_px), max(0, by0 - margin_px)
        bx1 = min(page_img.width, bx1 + margin_px)
        by1 = min(page_img.height, by1 + margin_px)

        thumb_x0 = x0 + padding
        thumb_y0 = y0 + header_h + padding
        if bx1 > bx0 and by1 > by0:
            crop = page_img.crop((int(bx0), int(by0), int(bx1), int(by1)))
            crop.thumbnail(thumb_size, Image.LANCZOS)
            offset_x = thumb_x0 + (tw - crop.width) // 2
            offset_y = thumb_y0 + (th - crop.height) // 2
            sheet.paste(crop, (offset_x, offset_y))

        draw.rectangle(
            [thumb_x0 - 1, thumb_y0 - 1, thumb_x0 + tw + 1, thumb_y0 + th + 1],
            outline=(180, 180, 180),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    logger.info(f"Contact sheet de celulas classificadas salva: {output_path} ({len(classifications)} celulas)")
    return output_path


def build_comparison_table(experiment: CellClassificationExperiment) -> List[dict]:
    """
    Tabela comparando, POR CÉLULA, o melhor template e score em cada
    representação (gray/binary/edges) — permite ver se as representações
    concordam ou divergem.
    """
    rows = []
    for c in experiment.classifications:
        row = {"cell_id": c.cell.cell_id, "is_first_cell": c.cell.is_first_cell}
        for rep in REPRESENTATIONS:
            ranked = c.scores_by_representation.get(rep, [])
            if ranked:
                row[f"{rep}_best"] = ranked[0].template_name
                row[f"{rep}_score"] = round(ranked[0].score, 4)
            else:
                row[f"{rep}_best"] = None
                row[f"{rep}_score"] = None
        row["agreement"] = c.agreement_across_representations
        row["accepted_primary"] = c.accepted
        rows.append(row)
    return rows


def print_comparison_table(rows: List[dict], only_accepted_or_agree: bool = True) -> None:
    """Log tabular legivel da comparacao entre representacoes."""
    logger.info("\n" + "=" * 110)
    logger.info("TABELA COMPARATIVA: gray vs binary vs edges (melhor template + score por celula)")
    logger.info("=" * 110)
    header = (f"{'cell_id':<16} | {'1st?':<5} | {'gray':<16} | {'binary':<16} | "
              f"{'edges':<16} | {'agree':<6} | accepted")
    logger.info(header)
    logger.info("-" * len(header))
    for row in rows:
        if only_accepted_or_agree and not (row["accepted_primary"] or row["agreement"]):
            continue
        gray_s = f"{row['gray_best']}({row['gray_score']:.2f})" if row["gray_score"] is not None else "-"
        bin_s = f"{row['binary_best']}({row['binary_score']:.2f})" if row["binary_score"] is not None else "-"
        edg_s = f"{row['edges_best']}({row['edges_score']:.2f})" if row["edges_score"] is not None else "-"
        logger.info(
            f"{row['cell_id']:<16} | {'yes' if row['is_first_cell'] else 'no':<5} | "
            f"{gray_s:<16} | {bin_s:<16} | {edg_s:<16} | "
            f"{'yes' if row['agreement'] else 'no':<6} | {row['accepted_primary']}"
        )


def save_cell_classification_json(experiment: CellClassificationExperiment, output_path: Path) -> Path:
    """Salva JSON completo com todas as classificacoes + fallback."""
    payload = {
        "total_cells": len(experiment.cells),
        "total_classifications": len(experiment.classifications),
        "accepted_count": sum(1 for c in experiment.classifications if c.accepted),
        "classifications": [c.to_json() for c in experiment.classifications],
        "fallback_matches": experiment.fallback_matches,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"JSON de classificacao por celula salvo: {output_path}")
    return output_path
