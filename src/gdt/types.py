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
    matched: bool


@dataclass
class GeometryMetrics:
    """Metricas da etapa de deteccao geometrica."""

    true_positives: int = 0
    false_negatives: int = 0
    false_positives: int = 0
    matches: List[FrameMatch] = field(default_factory=list)

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom else 0.0

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom else 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "true_positives": self.true_positives,
            "false_negatives": self.false_negatives,
            "false_positives": self.false_positives,
            "recall": round(self.recall, 4),
            "precision": round(self.precision, 4),
            "matches": [
                {
                    "ground_truth_id": match.ground_truth_id,
                    "candidate_id": match.candidate_id,
                    "iou": round(match.iou, 4),
                    "matched": match.matched,
                }
                for match in self.matches
            ],
        }
