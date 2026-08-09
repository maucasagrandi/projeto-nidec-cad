"""Gera ground truth versionado a partir de candidates.json + revisão humana.

Exemplo:
    python validation/gdt/scripts/build_ground_truth.py \
        --candidates validation/gdt/outputs/case_41_rev8/candidates.json \
        --review validation/gdt/review/case_41_rev8.review.json \
        --output validation/gdt/ground_truth/case_41_rev8.json
"""

from __future__ import annotations

import argparse

from src.gdt.ground_truth import build_ground_truth_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = build_ground_truth_file(args.candidates, args.review, args.output)

    print(f"case_id={payload.get('case_id')}")
    print(f"frames={payload['expected_frame_count']}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
