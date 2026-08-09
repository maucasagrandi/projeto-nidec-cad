"""Gera ground truth versionado a partir de candidates.json + revisão humana.

Exemplo:
    python validation/gdt/scripts/build_ground_truth.py \
        --candidates validation/gdt/outputs/case_41_rev8/candidates.json \
        --review validation/gdt/review/case_41_rev8.review.json \
        --output validation/gdt/ground_truth/case_41_rev8.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gdt.ground_truth import build_ground_truth_file


def _project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    candidates = _project_path(args.candidates)
    review = _project_path(args.review)
    output = _project_path(args.output)

    payload = build_ground_truth_file(candidates, review, output)

    print(f"case_id={payload.get('case_id')}")
    print(f"frames={payload['expected_frame_count']}")
    print(f"output={output}")


if __name__ == "__main__":
    main()
