"""Cadastra imagens locais como templates GD&T com nomes semânticos.

O script COPIA as imagens; os arquivos de origem não são alterados.

Exemplo (PowerShell, uma linha):
    python validation/gdt/scripts/register_templates.py --position C:\\temp\\position.png --profile C:\\temp\\profile.png --straightness C:\\temp\\straightness.png --flatness C:\\temp\\flatness.png

``--roundness`` é aceito como alias de entrada, mas a classe canônica usada no
catálogo é ``circularity`` para manter a nomenclatura determinística da norma.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = PROJECT_ROOT / "assets" / "gdt" / "templates"
ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}

# argumento CLI -> classe canônica no catálogo
ARG_TO_CLASS = {
    "position": "position",
    "profile": "profile",
    "straightness": "straightness",
    "flatness": "flatness",
    "circularity": "circularity",
    "roundness": "circularity",  # alias comum / nome usado na referência recebida
    "cylindricity": "cylindricity",
    "negative": "negative_controls",
}


def _copy_many(paths: list[str], class_name: str, root: Path) -> list[Path]:
    destination = root / class_name
    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # Continua a numeração se já existirem templates da mesma classe.
    existing = [
        path
        for path in destination.iterdir()
        if path.is_file() and path.suffix.lower() in ALLOWED_SUFFIXES
    ]
    start_index = len(existing) + 1

    for index, raw_path in enumerate(paths, start=start_index):
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
    parser.add_argument("--position", action="append", default=[], help="Símbolo Position; pode repetir.")
    parser.add_argument("--profile", action="append", default=[], help="Símbolo Profile; pode repetir.")
    parser.add_argument("--straightness", action="append", default=[], help="Símbolo Straightness; pode repetir.")
    parser.add_argument("--flatness", action="append", default=[], help="Símbolo Flatness; pode repetir.")
    parser.add_argument("--circularity", action="append", default=[], help="Símbolo Circularity; pode repetir.")
    parser.add_argument("--roundness", action="append", default=[], help="Alias de Circularity; pode repetir.")
    parser.add_argument("--cylindricity", action="append", default=[], help="Símbolo Cylindricity; pode repetir.")
    parser.add_argument(
        "--negative",
        action="append",
        default=[],
        help="Controle negativo legado. Não use círculo depois que Circularity estiver ativa.",
    )
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    args = parser.parse_args()

    values_by_class: dict[str, list[str]] = {}
    for arg_name, class_name in ARG_TO_CLASS.items():
        values = list(getattr(args, arg_name, []) or [])
        if values:
            values_by_class.setdefault(class_name, []).extend(values)

    if not values_by_class:
        raise SystemExit(
            "Informe ao menos um template (--position, --profile, --straightness, "
            "--flatness, --circularity/--roundness, --cylindricity ou --negative)."
        )

    root = Path(args.root)
    if not root.is_absolute():
        root = PROJECT_ROOT / root

    written: list[Path] = []
    for class_name, paths in values_by_class.items():
        written += _copy_many(paths, class_name, root)

    print(f"template_root={root}")
    for path in written:
        print(f"registered={path.relative_to(PROJECT_ROOT)}")
    print(f"count={len(written)}")


if __name__ == "__main__":
    main()
