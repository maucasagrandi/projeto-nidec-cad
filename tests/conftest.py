"""Pytest bootstrap for repository-local imports.

Ensures tests can import modules such as ``src.gdt`` when pytest is invoked
from the repository root on Windows/Linux without requiring a manual
PYTHONPATH environment variable.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
