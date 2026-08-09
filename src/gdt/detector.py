"""API estável para detecção geométrica de quadros GD&T.

Este módulo é a porta de entrada nova do pipeline GD&T. Por enquanto ele
reexporta a implementação já validada que ainda vive em ``src.utils``.

A vantagem é desacoplar o restante do projeto do caminho legado. Quando o
detector for migrado/refatorado, os consumidores continuam importando daqui.
"""

from src.utils.gdt_detector import (
    BBox,
    GdtCell,
    GdtFrameCandidate,
    GdtFrameDetector,
    detect_gdt_frames,
)

__all__ = [
    "BBox",
    "GdtCell",
    "GdtFrameCandidate",
    "GdtFrameDetector",
    "detect_gdt_frames",
]
