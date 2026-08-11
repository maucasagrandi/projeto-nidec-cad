"""Sincroniza templates com borda de ``cotas/`` para o catálogo de detecção.

O mapeamento é data-driven e fica em:
    validation/gdt/reference_catalog.json

Nesta versão (schema v3), os templates incluem a borda do feature control frame
e são copiados diretamente, sem preprocessing de crop/normalização. Eles serão
usados pelo detector via template matching na página renderizada.

O script verifica cobertura: toda imagem de referência em ``cotas/`` deve estar
registrada no manifesto. Isso evita adicionar uma cota e ela ficar fora do
detector silenciosamente.

Uso:
    python validation/gdt/scripts/sync_phase4_templates.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_MANIFEST = PROJECT_ROOT / "validation" / "gdt" / "reference_catalog.json"
ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def _normalize_filename(name: str) -> str:
    """Tolera diferenças de caixa e espaços antes da extensão."""
    return name.strip().casefold().replace(" .", ".")


def _load_manifest(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Manifesto não encontrado: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("entries"), list):
        raise ValueError("reference_catalog.json deve conter uma lista 'entries'.")
    return payload


def _source_index(source_root: Path) -> dict[str, Path]:
    """Index all image files in source_root by normalized filename."""
    if not source_root.exists():
        raise FileNotFoundError(f"Pasta de referências não encontrada: {source_root}")

    index: dict[str, Path] = {}
    for path in source_root.iterdir():
        if not path.is_file() or path.suffix.lower() not in ALLOWED_SUFFIXES:
            continue
        normalized = _normalize_filename(path.name)
        if normalized in index:
            raise ValueError(
                "Dois arquivos de referência colidem após normalização: "
                f"{index[normalized].name} / {path.name}"
            )
        index[normalized] = path
    return index


def main() -> None:
    manifest = _load_manifest(DEFAULT_MANIFEST)
    source_root = PROJECT_ROOT / str(manifest.get("source_root", "cotas"))
    template_root = PROJECT_ROOT / str(manifest.get("template_root", "assets/gdt/templates"))
    index = _source_index(source_root)

    entries = [
        entry
        for entry in manifest["entries"]
        if str(entry.get("status", "active")).lower() == "active"
    ]

    registered_sources = {
        _normalize_filename(str(entry.get("source", "")))
        for entry in entries
        if str(entry.get("source", "")).strip()
    }
    unregistered = sorted(
        path.name
        for normalized, path in index.items()
        if normalized not in registered_sources
    )

    written: list[tuple[Path, Path, str]] = []
    missing: list[str] = []
    active_classes: set[str] = set()
    target_paths: set[Path] = set()

    for entry in entries:
        source_name = str(entry["source"])
        class_name = str(entry["class_name"]).strip().lower()
        target_name = str(entry.get("target_name") or f"{class_name}_01.png")

        if not class_name:
            raise ValueError(f"class_name vazio para source={source_name!r}")

        # Try exact path first; then tolerate case/space normalization.
        exact_source = source_root / source_name
        source = exact_source if exact_source.exists() else index.get(_normalize_filename(source_name))
        if source is None:
            missing.append(source_name)
            continue

        # Verify image is readable
        gray = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f"Não foi possível ler a imagem: {source}")

        # Copy directly — bordered templates are used as-is for detection
        destination_dir = template_root / class_name
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / target_name

        if destination in target_paths:
            raise ValueError(f"target_name duplicado no manifesto: {destination}")
        target_paths.add(destination)

        shutil.copy2(str(source), str(destination))

        active_classes.add(class_name)
        written.append((source, destination, class_name))

    print("phase=4_template_sync")
    print(f"manifest={DEFAULT_MANIFEST.relative_to(PROJECT_ROOT)}")
    print(f"reference_count={len(index)}")
    print(f"manifest_active_entries={len(entries)}")
    for source, destination, class_name in written:
        print(
            f"registered class={class_name} "
            f"source={source.relative_to(PROJECT_ROOT)} "
            f"target={destination.relative_to(PROJECT_ROOT)}"
        )

    if missing:
        print("missing=" + ",".join(missing))
    if unregistered:
        print("unregistered=" + ",".join(unregistered))
    if missing or unregistered:
        raise SystemExit(2)

    print(f"registered_count={len(written)}")
    print("canonical_classes=" + ",".join(sorted(active_classes)))
    print("reference_coverage=PASS")


if __name__ == "__main__":
    main()
