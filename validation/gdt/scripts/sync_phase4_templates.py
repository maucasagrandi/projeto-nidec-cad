"""Sincroniza referências versionadas de ``cotas/`` para o catálogo visual.

O mapeamento é data-driven e fica em:
    validation/gdt/reference_catalog.json

Assim novas classes podem ser adicionadas ao manifesto sem alterar este script.

Uso:
    python validation/gdt/scripts/sync_phase4_templates.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gdt.symbol_classifier import crop_foreground, normalize_gray

DEFAULT_MANIFEST = PROJECT_ROOT / "validation" / "gdt" / "reference_catalog.json"


def _normalize_filename(name: str) -> str:
    # Tolera diferenças de caixa e espaços antes da extensão.
    return name.strip().casefold().replace(" .", ".")


def _load_manifest(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Manifesto não encontrado: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("entries"), list):
        raise ValueError("reference_catalog.json deve conter uma lista 'entries'.")
    return payload


def _source_index(source_root: Path) -> dict[str, Path]:
    if not source_root.exists():
        raise FileNotFoundError(f"Pasta de referências não encontrada: {source_root}")

    index: dict[str, Path] = {}
    for path in source_root.iterdir():
        if path.is_file():
            index[_normalize_filename(path.name)] = path
    return index


def main() -> None:
    manifest = _load_manifest(DEFAULT_MANIFEST)
    source_root = PROJECT_ROOT / str(manifest.get("source_root", "cotas"))
    template_root = PROJECT_ROOT / str(manifest.get("template_root", "assets/gdt/templates"))
    index = _source_index(source_root)

    written: list[tuple[Path, Path, str]] = []
    missing: list[str] = []
    active_classes: set[str] = set()

    for entry in manifest["entries"]:
        if str(entry.get("status", "active")).lower() != "active":
            continue

        source_name = str(entry["source"])
        class_name = str(entry["class_name"]).strip().lower()
        target_name = str(entry.get("target_name") or f"{class_name}_01.png")

        # Primeiro tenta o caminho exato; depois tolera caixa/espaço de filename.
        exact_source = source_root / source_name
        source = exact_source if exact_source.exists() else index.get(_normalize_filename(source_name))
        if source is None:
            missing.append(source_name)
            continue

        gray = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f"Não foi possível ler a imagem: {source}")

        # Somente normalização de contraste + remoção de whitespace externo.
        # A geometria interna do símbolo não é redesenhada nem sintetizada.
        prepared = crop_foreground(normalize_gray(gray), padding=3)
        destination_dir = template_root / class_name
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / target_name

        if not cv2.imwrite(str(destination), prepared):
            raise OSError(f"Falha ao gravar template: {destination}")

        active_classes.add(class_name)
        written.append((source, destination, class_name))

    print("phase=4_template_sync")
    print(f"manifest={DEFAULT_MANIFEST.relative_to(PROJECT_ROOT)}")
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
    print("canonical_classes=" + ",".join(sorted(active_classes)))


if __name__ == "__main__":
    main()
