"""Classificação determinística do símbolo GD&T na primeira célula.

Esta etapa NÃO decide conformidade e NÃO usa LLM. Ela apenas:

1. recebe candidatos geométricos já detectados;
2. recorta a primeira célula de cada candidato;
3. normaliza o crop;
4. compara o crop contra templates organizados por classe;
5. devolve todos os scores, melhor classe, segunda melhor e margem.

Nenhum threshold de aceitação é aplicado aqui de propósito. Thresholds serão
calibrados depois usando a distribuição dos scores em casos rotulados.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import cv2
import fitz
import numpy as np

from src.gdt.detector import BBox, GdtFrameCandidate

REPRESENTATIONS = ("gray", "binary", "edges")
DEFAULT_TARGET_SIZE = 48
DEFAULT_MARGIN = 10


@dataclass(frozen=True)
class TemplateImage:
    class_name: str
    template_name: str
    path: str
    representations: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class TemplateScore:
    class_name: str
    template_name: str
    scores: Mapping[str, float]
    mean_score: float


@dataclass(frozen=True)
class CandidateSymbolScore:
    candidate_id: str
    class_scores: Mapping[str, float]
    template_scores: Sequence[TemplateScore]
    best_class: str | None
    best_score: float
    second_best_class: str | None
    second_best_score: float
    margin: float

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "class_scores": {
                key: round(float(value), 6)
                for key, value in sorted(self.class_scores.items())
            },
            "best_class": self.best_class,
            "best_score": round(float(self.best_score), 6),
            "second_best_class": self.second_best_class,
            "second_best_score": round(float(self.second_best_score), 6),
            "margin": round(float(self.margin), 6),
            "templates": [
                {
                    "class_name": item.class_name,
                    "template_name": item.template_name,
                    "scores": {
                        rep: round(float(score), 6)
                        for rep, score in item.scores.items()
                    },
                    "mean_score": round(float(item.mean_score), 6),
                }
                for item in self.template_scores
            ],
        }


def render_page_gray(pdf_bytes: bytes, page_index: int = 0, dpi: int = 300) -> Tuple[np.ndarray, float]:
    """Renderiza uma página PDF em grayscale e retorna (imagem, zoom)."""

    zoom = dpi / 72.0
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        page = doc[page_index]
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csGRAY, alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
        return arr.copy(), zoom
    finally:
        doc.close()


def crop_cell_interior(
    page_gray: np.ndarray,
    bbox: BBox,
    zoom: float,
    *,
    inset_fraction: float = 0.08,
    min_inset_pt: float = 0.6,
    max_inset_pt: float = 2.5,
) -> np.ndarray | None:
    """Recorta o interior da célula removendo a maior parte das bordas."""

    inset_x = min(max(bbox.width * inset_fraction, min_inset_pt), max_inset_pt)
    inset_y = min(max(bbox.height * inset_fraction, min_inset_pt), max_inset_pt)

    x0 = int(round((bbox.x0 + inset_x) * zoom))
    y0 = int(round((bbox.y0 + inset_y) * zoom))
    x1 = int(round((bbox.x1 - inset_x) * zoom))
    y1 = int(round((bbox.y1 - inset_y) * zoom))

    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(page_gray.shape[1], x1)
    y1 = min(page_gray.shape[0], y1)

    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    return page_gray[y0:y1, x0:x1].copy()


def normalize_gray(gray: np.ndarray) -> np.ndarray:
    """Normaliza contraste/polaridade preservando o desenho do símbolo."""

    if gray.size == 0:
        return gray
    result = gray.copy()
    if float(result.mean()) < 127.0:
        result = 255 - result

    low = float(np.percentile(result, 1))
    high = float(np.percentile(result, 99))
    if high - low > 1.0:
        result = np.clip((result.astype(np.float32) - low) / (high - low) * 255.0, 0, 255).astype(np.uint8)
    return result


def crop_foreground(gray: np.ndarray, *, threshold: int = 245, padding: int = 2) -> np.ndarray:
    """Remove whitespace externo do template/crop sem deformar o conteúdo."""

    if gray.size == 0:
        return gray
    mask = gray < threshold
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return gray

    x0 = max(0, int(xs.min()) - padding)
    x1 = min(gray.shape[1], int(xs.max()) + padding + 1)
    y0 = max(0, int(ys.min()) - padding)
    y1 = min(gray.shape[0], int(ys.max()) + padding + 1)
    return gray[y0:y1, x0:x1]


def canonicalize(
    gray: np.ndarray,
    *,
    target_size: int = DEFAULT_TARGET_SIZE,
    margin: int = DEFAULT_MARGIN,
) -> Tuple[np.ndarray, np.ndarray]:
    """Retorna (search_canvas, tight_canvas), preservando aspect ratio."""

    gray = crop_foreground(normalize_gray(gray))
    h, w = gray.shape[:2]
    if h == 0 or w == 0:
        return (
            np.full((target_size + 2 * margin, target_size + 2 * margin), 255, np.uint8),
            np.full((target_size, target_size), 255, np.uint8),
        )

    scale = target_size / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(gray, (new_w, new_h), interpolation=interpolation)

    tight = np.full((target_size, target_size), 255, np.uint8)
    tx = (target_size - new_w) // 2
    ty = (target_size - new_h) // 2
    tight[ty:ty + new_h, tx:tx + new_w] = resized

    search_size = target_size + 2 * margin
    search = np.full((search_size, search_size), 255, np.uint8)
    sx = margin + tx
    sy = margin + ty
    search[sy:sy + new_h, sx:sx + new_w] = resized
    return search, tight


def derive_representations(gray: np.ndarray) -> Dict[str, np.ndarray]:
    """Gera representações complementares para comparação."""

    gray = normalize_gray(gray)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(gray, 50, 150)
    return {"gray": gray, "binary": binary, "edges": edges}


def _prepare_forms(gray: np.ndarray, *, target_size: int, margin: int) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    search, tight = canonicalize(gray, target_size=target_size, margin=margin)
    return derive_representations(search), derive_representations(tight)


def _match_score(search: np.ndarray, template: np.ndarray) -> float:
    """Maior correlação normalizada entre template e pequena janela de busca."""

    if search.shape[0] < template.shape[0] or search.shape[1] < template.shape[1]:
        return -1.0
    result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    score = float(np.nanmax(result)) if result.size else -1.0
    if not np.isfinite(score):
        return -1.0
    return max(-1.0, min(1.0, score))


def load_template_catalog(
    root: str | Path,
    *,
    target_size: int = DEFAULT_TARGET_SIZE,
    margin: int = DEFAULT_MARGIN,
) -> List[TemplateImage]:
    """Carrega ``root/<classe>/*.(png|jpg|jpeg|webp)``."""

    root = Path(root)
    allowed = {".png", ".jpg", ".jpeg", ".webp"}
    templates: List[TemplateImage] = []

    if not root.exists():
        raise FileNotFoundError(f"Pasta de templates não encontrada: {root}")

    for class_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        class_name = class_dir.name.strip().lower()
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() not in allowed:
                continue
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            _, tight_reps = _prepare_forms(gray, target_size=target_size, margin=margin)
            templates.append(
                TemplateImage(
                    class_name=class_name,
                    template_name=path.stem,
                    path=str(path),
                    representations=tight_reps,
                )
            )

    if not templates:
        raise ValueError(f"Nenhum template de imagem encontrado em: {root}")
    return templates


def score_crop(
    crop: np.ndarray,
    templates: Sequence[TemplateImage],
    *,
    target_size: int = DEFAULT_TARGET_SIZE,
    margin: int = DEFAULT_MARGIN,
) -> Tuple[Dict[str, float], List[TemplateScore]]:
    """Compara um crop contra todos os templates e agrega melhor score por classe."""

    search_reps, _ = _prepare_forms(crop, target_size=target_size, margin=margin)
    template_scores: List[TemplateScore] = []
    class_scores: Dict[str, float] = {}

    for template in templates:
        rep_scores = {
            rep: _match_score(search_reps[rep], template.representations[rep])
            for rep in REPRESENTATIONS
        }
        mean_score = float(np.mean([rep_scores[rep] for rep in REPRESENTATIONS]))
        item = TemplateScore(
            class_name=template.class_name,
            template_name=template.template_name,
            scores=rep_scores,
            mean_score=mean_score,
        )
        template_scores.append(item)
        class_scores[template.class_name] = max(
            mean_score,
            class_scores.get(template.class_name, -1.0),
        )

    template_scores.sort(key=lambda item: item.mean_score, reverse=True)
    return class_scores, template_scores


def score_candidate_symbol(
    candidate: GdtFrameCandidate,
    page_gray: np.ndarray,
    zoom: float,
    templates: Sequence[TemplateImage],
    *,
    target_size: int = DEFAULT_TARGET_SIZE,
    margin: int = DEFAULT_MARGIN,
) -> Tuple[CandidateSymbolScore, np.ndarray | None]:
    """Pontua a primeira célula de um candidato sem aplicar threshold."""

    crop = crop_cell_interior(page_gray, candidate.symbol_bbox, zoom)
    if crop is None:
        empty = CandidateSymbolScore(
            candidate_id=candidate.candidate_id,
            class_scores={},
            template_scores=[],
            best_class=None,
            best_score=-1.0,
            second_best_class=None,
            second_best_score=-1.0,
            margin=0.0,
        )
        return empty, None

    class_scores, template_scores = score_crop(
        crop,
        templates,
        target_size=target_size,
        margin=margin,
    )
    ranked = sorted(class_scores.items(), key=lambda item: item[1], reverse=True)

    best_class, best_score = ranked[0] if ranked else (None, -1.0)
    second_class, second_score = ranked[1] if len(ranked) > 1 else (None, -1.0)
    margin_value = best_score - second_score if best_class is not None and second_class is not None else 0.0

    result = CandidateSymbolScore(
        candidate_id=candidate.candidate_id,
        class_scores=class_scores,
        template_scores=template_scores,
        best_class=best_class,
        best_score=float(best_score),
        second_best_class=second_class,
        second_best_score=float(second_score),
        margin=float(margin_value),
    )
    return result, crop


def score_candidates(
    candidates: Iterable[GdtFrameCandidate],
    page_gray: np.ndarray,
    zoom: float,
    templates: Sequence[TemplateImage],
    *,
    target_size: int = DEFAULT_TARGET_SIZE,
    margin: int = DEFAULT_MARGIN,
) -> List[Tuple[CandidateSymbolScore, np.ndarray | None]]:
    return [
        score_candidate_symbol(
            candidate,
            page_gray,
            zoom,
            templates,
            target_size=target_size,
            margin=margin,
        )
        for candidate in candidates
    ]
