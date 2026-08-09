"""Validacao objetiva do recall geometrico dos quadros GD&T."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from src.gdt.detector import GdtFrameCandidate, GdtFrameDetector
from src.gdt.types import FrameMatch, GeometryMetrics, GroundTruthFrame


def _bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b

    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


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
) -> GeometryMetrics:
    """Faz matching 1:1 entre GT e candidatos usando maior IoU disponivel.

    A meta desta etapa e recall. ``min_iou`` nao deve ser calibrado para
    esconder deteccoes ruins; ele apenas evita considerar qualquer
    sobreposicao minima como acerto.
    """

    gt_list = list(ground_truth)
    used_candidates: set[int] = set()
    matches: List[FrameMatch] = []
    tp = 0
    fn = 0

    for gt in gt_list:
        best_idx = None
        best_iou = 0.0
        for idx, candidate in enumerate(candidates):
            if idx in used_candidates or candidate.page != gt.page:
                continue
            score = _bbox_iou(gt.bbox, candidate.frame_bbox.to_list())
            if score > best_iou:
                best_iou = score
                best_idx = idx

        matched = best_idx is not None and best_iou >= min_iou
        if matched:
            used_candidates.add(best_idx)
            tp += 1
            candidate_id = candidates[best_idx].candidate_id
        else:
            fn += 1
            candidate_id = None

        matches.append(
            FrameMatch(
                ground_truth_id=gt.frame_id,
                candidate_id=candidate_id,
                iou=best_iou,
                matched=matched,
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
    detector: GdtFrameDetector | None = None,
) -> Tuple[List[GdtFrameCandidate], GeometryMetrics]:
    """Executa o detector atual e calcula metricas contra um ground truth."""

    pdf_bytes = Path(pdf_path).read_bytes()
    detector = detector or GdtFrameDetector()
    candidates = detector.detect_frames(pdf_bytes, page_index=page_index)
    ground_truth = [gt for gt in load_ground_truth(ground_truth_path) if gt.page == page_index + 1]
    metrics = match_ground_truth(ground_truth, candidates, min_iou=min_iou)
    return candidates, metrics
