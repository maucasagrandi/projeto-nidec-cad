"""Sincroniza as referências da pasta ``cotas/`` para o catálogo da Fase 4.

As imagens de referência permanecem em ``cotas/`` como fonte humana/auditável.
O runtime usa somente cópias semânticas em ``assets/gdt/templates/<classe>/``.

Mapeamento inicial da Fase 4:
- straightness.png   -> straightness
- flatness.png       -> flatness
- Roundness.png      -> circularity (nome canônico)
- Cylindricity .png  -> cylindricity

Uso:
    python validation/gdt/scripts/sync_phase4_templates.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gdt.symbol_classifier import crop_foreground, normalize_gray

SOURCE_ROOT = PROJECT_ROOT / "cotas"
TEMPLATE_ROOT = PROJECT_ROOT / "assets" / "gdt" / "templates"

# Nome normalizado da referência -> classe canônica.
REFERENCE_SPECS = {
    "straightness.png": "straightness",
    "flatness.png": "flatness",
    "roundness.png": "circularity",
    "cylindricity.png": "cylindricity",
}


def _normalize_filename(name: str) -> str:
    # Tolera o espaço existente em ``Cylindricity .png`` sem perpetuá-lo.
    return name.strip().casefold().replace(" .", ".")


def _source_index() -> dict[str, Path]:
    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(f"Pasta de referências não encontrada: {SOURCE_ROOT}")

    index: dict[str, Path] = {}
    for path in SOURCE_ROOT.iterdir():
        if path.is_file():
            index[_normalize_filename(path.name)] = path
    return index


def main() -> None:
    index = _source_index()
    written: list[tuple[Path, Path, str]] = []
    missing: list[str] = []

    for expected_name, class_name in REFERENCE_SPECS.items():
        source = index.get(_normalize_filename(expected_name))
        if source is None:
            missing.append(expected_name)
            continue

        gray = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f"Não foi possível ler a imagem: {source}")

        # Remove somente whitespace externo/contraste. Não altera a geometria do símbolo.
        prepared = crop_foreground(normalize_gray(gray), padding=3)
        destination_dir = TEMPLATE_ROOT / class_name
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / f"{class_name}_01.png"

        if not cv2.imwrite(str(destination), prepared):
            raise OSError(f"Falha ao gravar template: {destination}")
        written.append((source, destination, class_name))

    print("phase=4_template_sync")
    for source, destination, class_name in written:
        print(
            f"registered class={class_name} "
            f"source={source.relative_to(PROJECT_ROOT)} "
            f"target={destination.relative_to(PROJECT_ROOT)}"
        )

    if missing:
        print("missing=" + ",".join(missing))
        raise SystemExit(2)

    print(f"registered_count={len(written)}")
    print("canonical_classes=straightness,flatness,circularity,cylindricity")


if __name__ == "__main__":
    main()
