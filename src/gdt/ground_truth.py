"""Construção e validação do ground truth GD&T a partir de candidatos revisados.

O objetivo deste módulo é evitar digitação manual de coordenadas. A revisão
humana informa apenas quais candidatos são quadros GD&T reais e a classe de
cada um. O bbox congelado no ground truth é copiado do ``candidates.json``.

Se houver um quadro real que o detector não encontrou, ele deve entrar em
``manual_frames`` com bbox anotado separadamente. Isso evita esconder falsos
negativos durante a criação do ground truth.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


ALLOWED_BOOTSTRAP_CHARACTERISTICS = {"position", "profile"}


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_ground_truth_payload(
    candidates_payload: Dict[str, Any],
    review_payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Cria o payload versionado de ground truth a partir de uma revisão.

    Formato mínimo da revisão::

        {
          "accepted_candidates": [
            {"candidate_id": "GDT-CAND-P01-002", "characteristic": "position"}
          ],
          "manual_frames": []
        }

    ``manual_frames`` existe explicitamente para quadros verdadeiros que não
    aparecem na lista de candidatos; eles não podem ser omitidos do GT.
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
        characteristic = str(reviewed["characteristic"]).strip().lower()

        if candidate_id in seen_ids:
            raise ValueError(f"Candidato repetido na revisão: {candidate_id}")
        if candidate_id not in candidates:
            raise ValueError(f"Candidato não encontrado em candidates.json: {candidate_id}")
        if characteristic not in ALLOWED_BOOTSTRAP_CHARACTERISTICS:
            raise ValueError(
                f"Classe '{characteristic}' ainda não faz parte do bootstrap. "
                f"Permitidas: {sorted(ALLOWED_BOOTSTRAP_CHARACTERISTICS)}"
            )

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

        characteristic = str(manual["characteristic"]).strip().lower()
        if characteristic not in ALLOWED_BOOTSTRAP_CHARACTERISTICS:
            raise ValueError(
                f"Classe manual '{characteristic}' ainda não faz parte do bootstrap."
            )

        counter += 1
        frames.append(
            {
                "id": f"GT-{counter:03d}",
                "page": int(manual.get("page", 1)),
                "characteristic": characteristic,
                "bbox": [float(v) for v in bbox],
                "source": "manual_annotation",
                "notes": str(manual.get("notes", "")),
            }
        )

    frames.sort(key=lambda item: (item["page"], item["bbox"][1], item["bbox"][0]))
    for index, frame in enumerate(frames, start=1):
        frame["id"] = f"GT-{index:03d}"

    return {
        "schema_version": 1,
        "case_id": review_payload.get("case_id") or candidates_payload.get("case_id"),
        "pdf": candidates_payload.get("pdf"),
        "page": int(candidates_payload.get("page", 1)),
        "expected_frame_count": len(frames),
        "frames": frames,
    }


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
