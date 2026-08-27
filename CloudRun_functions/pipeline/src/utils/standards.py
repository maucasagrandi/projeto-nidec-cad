"""Deterministic helpers for normalizing standards extracted by the LLM."""

from __future__ import annotations

import re
from typing import Any


def standard_key(value: Any) -> str:
    """Return a stable key for matching standard codes across data sources."""
    return " ".join(str(value or "").upper().split())


def is_generic_standard(value: Any) -> bool:
    """Return whether a value is a generic ISO mention rather than a code."""
    normalized = re.sub(r"[^A-Z0-9]+", " ", standard_key(value)).strip()
    return bool(re.fullmatch(r"(?:SEE )?(?:GENERAL )?ISO(?: STANDARDS?)?", normalized))


def filter_standard_entries(
    standards: list[Any] | None,
    evidence: list[Any] | None = None,
) -> tuple[list[str], list[str]]:
    """Remove generic standards while preserving standard/evidence alignment."""
    evidence_values = list(evidence or [])
    kept_standards: list[str] = []
    kept_evidence: list[str] = []
    seen: set[str] = set()

    for index, raw_standard in enumerate(standards or []):
        standard = str(raw_standard).strip()
        key = standard_key(standard)
        if not standard or not key or key in seen or is_generic_standard(standard):
            continue
        seen.add(key)
        kept_standards.append(standard)
        kept_evidence.append(
            str(evidence_values[index]).strip() if index < len(evidence_values) else ""
        )

    return kept_standards, kept_evidence
