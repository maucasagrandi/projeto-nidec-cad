"""Tipos compartilhados pelo pipeline GD&T deterministico."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class GroundTruthFrame:
    """Quadro GD&T anotado manualmente para validacao."""

    frame_id: str
    page: int
    characteristic: str
    bbox: Tuple[float, float, float, float]


@dataclass(frozen=True)
class FrameMatch:
    """Associacao entre um ground truth e um candidato geometrico."""

    ground_truth_id: str
    candidate_id: Optional[str]
    iou: float
    overlap_smallest: float
    area_ratio: float
    matched: bool
    match_reason: str = ""


@dataclass
class GeometryMetrics:
    """Metricas da etapa de deteccao geometrica."""

    true_positives: int = 0
    false_negatives: int = 0
    false_positives: int = 0
    matches: List[FrameMatch] = field(default_factory=list)

    @property
    def ground_truth_count(self) -> int:
        return self.true_positives + self.false_negatives

    @property
    def candidate_count(self) -> int:
        return self.true_positives + self.false_positives

    @property
    def recall(self) -> float:
        denom = self.ground_truth_count
        return self.true_positives / denom if denom else 0.0

    @property
    def precision(self) -> float:
        denom = self.candidate_count
        return self.true_positives / denom if denom else 0.0

    @property
    def f1(self) -> float:
        denom = self.precision + self.recall
        return 2 * self.precision * self.recall / denom if denom else 0.0

    def passes_recall_gate(self, minimum_recall: float = 0.95) -> bool:
        """Gate da Fase 1: nesta etapa recall tem prioridade sobre precisão."""

        return self.ground_truth_count > 0 and self.recall >= minimum_recall

    def to_dict(self, *, minimum_recall: float = 0.95) -> Dict[str, object]:
        return {
            "ground_truth_count": self.ground_truth_count,
            "candidate_count": self.candidate_count,
            "true_positives": self.true_positives,
            "false_negatives": self.false_negatives,
            "false_positives": self.false_positives,
            "recall": round(self.recall, 4),
            "precision": round(self.precision, 4),
            "f1": round(self.f1, 4),
            "recall_gate": {
                "minimum": minimum_recall,
                "passed": self.passes_recall_gate(minimum_recall),
            },
            "matches": [
                {
                    "ground_truth_id": match.ground_truth_id,
                    "candidate_id": match.candidate_id,
                    "iou": round(match.iou, 4),
                    "overlap_smallest": round(match.overlap_smallest, 4),
                    "area_ratio": round(match.area_ratio, 4),
                    "matched": match.matched,
                    "match_reason": match.match_reason,
                }
                for match in self.matches
            ],
        }
