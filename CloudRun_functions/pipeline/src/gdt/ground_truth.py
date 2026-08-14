"""Construção e validação de ground truth GD&T.

Há dois tipos de anotação possíveis:

1. ``manual_annotation``: bbox anotado independentemente da saída do detector.
   Este é o único tipo adequado para medir recall/precision geométricos de
   forma oficial.
2. ``reviewed_candidate``: bbox copiado de um candidato após revisão humana.
   É útil para bootstrap e para treinar/validar etapas posteriores, mas NÃO é
   independente do detector e portanto não deve ser usado para declarar
   desempenho geométrico.

A distinção é persistida no JSON por ``independent_annotation``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


ALLOWED_BOOTSTRAP_CHARACTERISTICS = {"position", "profile"}


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _validate_characteristic(value: Any, *, prefix: str = "Classe") -> str:
    characteristic = str(value).strip().lower()
    if characteristic not in ALLOWED_BOOTSTRAP_CHARACTERISTICS:
        raise ValueError(
            f"{prefix} '{characteristic}' ainda não faz parte do bootstrap. "
            f"Permitidas: {sorted(ALLOWED_BOOTSTRAP_CHARACTERISTICS)}"
        )
    return characteristic


def build_ground_truth_payload(
    candidates_payload: Dict[str, Any],
    review_payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Cria ground truth versionado a partir de uma revisão humana.

    ``accepted_candidates`` é permitido apenas como bootstrap. Como o bbox é
    copiado da própria saída do detector, qualquer GT contendo esse tipo de
    entrada é marcado como ``independent_annotation=False``.

    Para benchmark geométrico, use somente ``manual_frames``.
    """

    candidates = {
        item["candidate_id"]: item
        for item in candidates_payload.get("candidates", [])
    }

    accepted = review_payload.get("accepted_candidates", [])
    manual_frames = review_payload.get("manual_frames", [])

    if not accepted and not manual_frames:
        raise ValueError("A revisão não contém nenhum quadro GD&T real.")

    frames: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    counter = 0

    for reviewed in accepted:
        candidate_id = str(reviewed["candidate_id"])
        characteristic = _validate_characteristic(reviewed["characteristic"])

        if candidate_id in seen_ids:
            raise ValueError(f"Candidato repetido na revisão: {candidate_id}")
        if candidate_id not in candidates:
            raise ValueError(f"Candidato não encontrado em candidates.json: {candidate_id}")

        candidate = candidates[candidate_id]
        counter += 1
        seen_ids.add(candidate_id)
        frames.append(
            {
                "id": f"GT-{counter:03d}",
                "page": int(candidate.get("page", 1)),
                "characteristic": characteristic,
                "bbox": [float(v) for v in candidate["frame_bbox"]],
                "source": "reviewed_candidate",
                "source_candidate_id": candidate_id,
            }
        )

    for manual in manual_frames:
        bbox = manual.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"manual_frame com bbox inválido: {bbox}")

        characteristic = _validate_characteristic(
            manual["characteristic"], prefix="Classe manual"
        )
        x0, y0, x1, y1 = [float(v) for v in bbox]
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"manual_frame com bbox degenerado: {bbox}")

        counter += 1
        frames.append(
            {
                "id": f"GT-{counter:03d}",
                "page": int(manual.get("page", 1)),
                "characteristic": characteristic,
                "bbox": [x0, y0, x1, y1],
                "source": "manual_annotation",
                "notes": str(manual.get("notes", "")),
            }
        )

    frames.sort(key=lambda item: (item["page"], item["bbox"][1], item["bbox"][0]))
    for index, frame in enumerate(frames, start=1):
        frame["id"] = f"GT-{index:03d}"

    independent = bool(frames) and all(
        frame.get("source") == "manual_annotation" for frame in frames
    )

    return {
        "schema_version": 2,
        "case_id": review_payload.get("case_id") or candidates_payload.get("case_id"),
        "pdf": candidates_payload.get("pdf"),
        "page": int(candidates_payload.get("page", 1)),
        "expected_frame_count": len(frames),
        "independent_annotation": independent,
        "benchmark_grade": independent,
        "frames": frames,
    }


def is_independent_ground_truth(payload: Dict[str, Any]) -> bool:
    """Retorna True somente se todos os bboxes foram anotados manualmente."""

    if payload.get("independent_annotation") is not None:
        return bool(payload["independent_annotation"])
    frames = payload.get("frames", [])
    return bool(frames) and all(
        frame.get("source") == "manual_annotation" for frame in frames
    )


def assert_independent_ground_truth(payload: Dict[str, Any]) -> None:
    """Impede uso acidental de GT circular em benchmark geométrico."""

    if not is_independent_ground_truth(payload):
        raise ValueError(
            "Ground truth não independente: contém bbox derivado de candidato. "
            "Use manual_frames para medir recall/precision geométricos ou execute "
            "explicitamente em modo exploratório."
        )


def build_ground_truth_file(
    candidates_path: str | Path,
    review_path: str | Path,
    output_path: str | Path,
) -> Dict[str, Any]:
    """Lê os dois JSONs, gera o ground truth e o persiste em disco."""

    payload = build_ground_truth_payload(
        _load_json(candidates_path),
        _load_json(review_path),
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload
