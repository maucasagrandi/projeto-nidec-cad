"""Cadastra imagens locais como templates GD&T com nomes semânticos.

Exemplo (PowerShell, uma linha):
    python validation/gdt/scripts/register_templates.py --position C:\\temp\\position.png --profile C:\\temp\\profile.png --negative C:\\temp\\circle.png

O script COPIA as imagens; os arquivos de origem não são alterados.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = PROJECT_ROOT / "assets" / "gdt" / "templates"
ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def _copy_many(paths: list[str], class_name: str, root: Path) -> list[Path]:
    destination = root / class_name
    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for index, raw_path in enumerate(paths, start=1):
        source = Path(raw_path).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"Template não encontrado: {source}")
        if source.suffix.lower() not in ALLOWED_SUFFIXES:
            raise ValueError(f"Formato não suportado: {source.suffix} ({source})")

        prefix = "negative" if class_name == "negative_controls" else class_name
        target = destination / f"{prefix}_{index:02d}{source.suffix.lower()}"
        shutil.copy2(source, target)
        written.append(target)

    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--position", action="append", default=[], help="Imagem de símbolo Position; pode repetir.")
    parser.add_argument("--profile", action="append", default=[], help="Imagem de símbolo Profile; pode repetir.")
    parser.add_argument("--negative", action="append", default=[], help="Controle negativo; pode repetir.")
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    args = parser.parse_args()

    if not (args.position or args.profile or args.negative):
        raise SystemExit("Informe ao menos um --position, --profile ou --negative.")

    root = Path(args.root)
    if not root.is_absolute():
        root = PROJECT_ROOT / root

    written: list[Path] = []
    written += _copy_many(args.position, "position", root)
    written += _copy_many(args.profile, "profile", root)
    written += _copy_many(args.negative, "negative_controls", root)

    print(f"template_root={root}")
    for path in written:
        print(f"registered={path.relative_to(PROJECT_ROOT)}")
    print(f"count={len(written)}")


if __name__ == "__main__":
    main()
