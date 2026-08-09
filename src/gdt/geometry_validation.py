"""Validacao objetiva do recall geometrico dos quadros GD&T."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from src.gdt.detector import GdtFrameCandidate, GdtFrameDetector
from src.gdt.types import FrameMatch, GeometryMetrics, GroundTruthFrame


def _bbox_overlap_metrics(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    """Retorna (IoU, intersecao/smaller_box, razao_de_areas).

    ``overlap_smallest`` e util para ground truth desenhado manualmente: o ROI
    humano pode incluir alguns pixels de leader line ou deixar margem extra,
    sem que isso signifique que o detector errou a localizacao do frame.

    ``area_ratio`` e sempre >= 1 e impede que uma caixa minúscula inteiramente
    dentro de outra seja aceita apenas porque overlap_smallest=1.
    """

    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b

    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih

    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    if area_a <= 0 or area_b <= 0:
        return 0.0, 0.0, float("inf")

    union = area_a + area_b - inter
    iou = inter / union if union > 0 else 0.0
    overlap_smallest = inter / min(area_a, area_b) if inter > 0 else 0.0
    area_ratio = max(area_a, area_b) / min(area_a, area_b)
    return iou, overlap_smallest, area_ratio


def _bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Compatibilidade com testes/codigo anterior."""

    return _bbox_overlap_metrics(a, b)[0]


def load_ground_truth(path: str | Path) -> List[GroundTruthFrame]:
    """Carrega ground truth no formato versionado em ``validation/gdt/ground_truth``."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    frames = data.get("frames", [])
    result: List[GroundTruthFrame] = []
    for item in frames:
        bbox = item["bbox"]
        if len(bbox) != 4:
            raise ValueError(f"bbox invalido em {item.get('id')}: {bbox}")
        result.append(
            GroundTruthFrame(
                frame_id=str(item["id"]),
                page=int(item.get("page", 1)),
                characteristic=str(item["characteristic"]),
                bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
            )
        )
    return result


def match_ground_truth(
    ground_truth: Iterable[GroundTruthFrame],
    candidates: Sequence[GdtFrameCandidate],
    *,
    min_iou: float = 0.35,
    min_overlap_smallest: float = 0.50,
    max_area_ratio: float = 2.50,
) -> GeometryMetrics:
    """Faz matching 1:1 entre GT manual e candidatos geometricos.

    Um par e aceito quando:

    - IoU >= ``min_iou``; OU
    - pelo menos ``min_overlap_smallest`` da menor das duas caixas esta
      sobreposta E as areas nao diferem por mais que ``max_area_ratio``.

    O segundo criterio existe porque o ground truth e desenhado a mao sobre o
    PDF original e portanto nao deve exigir coincidencia pixel-perfect com as
    linhas vetoriais reconstruidas pelo detector. IoU continua registrado no
    relatorio para permitir auditoria.
    """

    gt_list = list(ground_truth)
    used_candidates: set[int] = set()
    matches: List[FrameMatch] = []
    tp = 0
    fn = 0

    for gt in gt_list:
        best_idx = None
        best_rank = (-1.0, -1.0)
        best_metrics = (0.0, 0.0, float("inf"))

        for idx, candidate in enumerate(candidates):
            if idx in used_candidates or candidate.page != gt.page:
                continue

            metrics = _bbox_overlap_metrics(gt.bbox, candidate.frame_bbox.to_list())
            iou, overlap_smallest, area_ratio = metrics
            rank = (iou, overlap_smallest)
            if rank > best_rank:
                best_rank = rank
                best_metrics = metrics
                best_idx = idx

        best_iou, best_overlap, best_area_ratio = best_metrics
        iou_match = best_idx is not None and best_iou >= min_iou
        overlap_match = (
            best_idx is not None
            and best_overlap >= min_overlap_smallest
            and best_area_ratio <= max_area_ratio
        )
        matched = iou_match or overlap_match

        if matched:
            used_candidates.add(best_idx)
            tp += 1
            candidate_id = candidates[best_idx].candidate_id
            match_reason = "iou" if iou_match else "manual_roi_overlap"
        else:
            fn += 1
            candidate_id = None
            match_reason = "below_threshold"

        matches.append(
            FrameMatch(
                ground_truth_id=gt.frame_id,
                candidate_id=candidate_id,
                iou=best_iou,
                overlap_smallest=best_overlap,
                area_ratio=best_area_ratio,
                matched=matched,
                match_reason=match_reason,
            )
        )

    fp = len(candidates) - len(used_candidates)
    return GeometryMetrics(
        true_positives=tp,
        false_negatives=fn,
        false_positives=fp,
        matches=matches,
    )


def detect_and_validate(
    pdf_path: str | Path,
    ground_truth_path: str | Path,
    *,
    page_index: int = 0,
    min_iou: float = 0.35,
    min_overlap_smallest: float = 0.50,
    max_area_ratio: float = 2.50,
    detector: GdtFrameDetector | None = None,
) -> Tuple[List[GdtFrameCandidate], GeometryMetrics]:
    """Executa o detector atual e calcula metricas contra um ground truth."""

    pdf_bytes = Path(pdf_path).read_bytes()
    detector = detector or GdtFrameDetector()
    candidates = detector.detect_frames(pdf_bytes, page_index=page_index)
    ground_truth = [gt for gt in load_ground_truth(ground_truth_path) if gt.page == page_index + 1]
    metrics = match_ground_truth(
        ground_truth,
        candidates,
        min_iou=min_iou,
        min_overlap_smallest=min_overlap_smallest,
        max_area_ratio=max_area_ratio,
    )
    return candidates, metrics
