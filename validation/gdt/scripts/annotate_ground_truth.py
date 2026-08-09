"""Anota ground truth geométrico diretamente sobre o CAD original.

O desenho mostrado NÃO contém candidatos nem overlays do detector. Assim os
bboxes do ground truth são independentes da saída que será avaliada.

Uso:
    python validation/gdt/scripts/annotate_ground_truth.py \
      --case validation/gdt/cases/case_41_rev8.json

Fluxo:
1. Uma janela abre com a página original.
2. Arraste um retângulo sobre cada quadro GD&T real.
3. ENTER/SPACE confirma a seleção; ESC cancela.
4. No terminal, informe a classe de cada ROI: p=position, r=profile, u=unknown.
5. O script salva o JSON independente + uma imagem de revisão.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import fitz
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


CLASS_KEYS = {
    "p": "position",
    "position": "position",
    "r": "profile",
    "profile": "profile",
    "u": "unknown",
    "unknown": "unknown",
}


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _render_page_rgb(pdf_bytes: bytes, page_index: int, dpi: int) -> tuple[np.ndarray, float]:
    zoom = dpi / 72.0
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        page = doc[page_index]
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csRGB, alpha=False)
        rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), zoom
    finally:
        doc.close()


def _ask_class(index: int) -> str:
    while True:
        raw = input(f"Classe ROI {index} [p=position, r=profile, u=unknown]: ").strip().lower()
        if raw in CLASS_KEYS:
            return CLASS_KEYS[raw]
        print("Valor inválido. Use p, r ou u.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--output")
    parser.add_argument("--preview")
    args = parser.parse_args()

    case_path = _project_path(args.case)
    config = json.loads(case_path.read_text(encoding="utf-8"))
    case_id = str(config["case_id"])
    pdf_path = _project_path(config["pdf"])
    page_index = int(config.get("page_index", 0))
    expected_count = config.get("expected", {}).get("frame_count")

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF não encontrado: {pdf_path}")

    page_bgr, zoom = _render_page_rgb(pdf_path.read_bytes(), page_index, args.dpi)

    window = "GD&T ground truth - selecione TODOS os quadros reais"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1600, 900)
    print("Selecione cada quadro GD&T real no CAD ORIGINAL.")
    print("ENTER/SPACE finaliza. ESC cancela a seleção atual.")
    rois = cv2.selectROIs(window, page_bgr, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    rois = [tuple(int(v) for v in roi) for roi in rois if roi[2] > 0 and roi[3] > 0]
    if not rois:
        raise SystemExit("Nenhum ROI selecionado; ground truth não foi alterado.")

    print(f"ROIs selecionados: {len(rois)}")
    if expected_count is not None and len(rois) != int(expected_count):
        print(f"AVISO: o caso espera {expected_count} quadros, mas foram selecionados {len(rois)}.")

    frames = []
    preview = page_bgr.copy()
    for idx, (x, y, w, h) in enumerate(rois, start=1):
        characteristic = _ask_class(idx)
        bbox_pdf = [x / zoom, y / zoom, (x + w) / zoom, (y + h) / zoom]
        frames.append(
            {
                "id": f"GT-{idx:03d}",
                "page": page_index + 1,
                "characteristic": characteristic,
                "bbox": [round(float(v), 3) for v in bbox_pdf],
                "source": "manual_annotation",
                "notes": "Annotated on original PDF page without detector overlay.",
            }
        )
        cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(
            preview,
            f"GT-{idx:03d} {characteristic}",
            (x, max(18, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    # Ordena para gerar IDs estáveis: de cima para baixo, esquerda para direita.
    frames.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    for idx, frame in enumerate(frames, start=1):
        frame["id"] = f"GT-{idx:03d}"

    payload = {
        "schema_version": 2,
        "case_id": case_id,
        "pdf": str(pdf_path),
        "page": page_index + 1,
        "expected_frame_count": len(frames),
        "independent_annotation": True,
        "benchmark_grade": True,
        "annotation": {
            "method": "manual_roi_on_original_pdf",
            "detector_overlay_visible": False,
            "dpi": args.dpi,
            "zoom": zoom,
        },
        "frames": frames,
    }

    output_path = _project_path(
        args.output or f"validation/gdt/ground_truth/{case_id}.json"
    )
    preview_path = _project_path(
        args.preview or f"validation/gdt/outputs/{case_id}/ground_truth_preview.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    cv2.imwrite(str(preview_path), preview)

    by_class: dict[str, int] = {}
    for frame in frames:
        by_class[frame["characteristic"]] = by_class.get(frame["characteristic"], 0) + 1

    print(f"ground_truth={output_path}")
    print(f"preview={preview_path}")
    print(f"frames={len(frames)}")
    print(f"classes={by_class}")
    print("independent_annotation=True")


if __name__ == "__main__":
    main()
