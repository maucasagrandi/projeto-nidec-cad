"""GD&T detection via direct template matching on rendered pages.

Scans a rendered PDF page for GD&T feature control frame symbols using bordered
templates at multiple scales and rotations. Uses thread-pool parallelism for
the heavy matchTemplate operations.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import fitz
import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TemplateDef:
    class_name: str
    template_name: str
    path: str
    image: np.ndarray


@dataclass(frozen=True)
class Detection:
    class_name: str
    template_name: str
    score: float
    x: float
    y: float
    width: float
    height: float
    scale: float
    rotation: int
    pixel_bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)

    def to_dict(self) -> dict:
        return {
            "class_name": self.class_name,
            "template_name": self.template_name,
            "score": round(float(self.score), 4),
            "bbox_pt": [round(self.x, 2), round(self.y, 2),
                        round(self.x + self.width, 2), round(self.y + self.height, 2)],
            "width_pt": round(self.width, 2),
            "height_pt": round(self.height, 2),
            "scale": round(self.scale, 3),
            "rotation": self.rotation,
            "pixel_bbox": list(self.pixel_bbox),
        }


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_DPI = 150
DEFAULT_SCALES = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0, 1.1)
DEFAULT_ROTATIONS = (0, 90, -90)
DEFAULT_SCORE_THRESHOLD = 0.74
DEFAULT_NMS_IOU_THRESHOLD = 0.30
ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


# ---------------------------------------------------------------------------
# Template loading
# ---------------------------------------------------------------------------


def load_templates(root: str | Path) -> List[TemplateDef]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Template directory not found: {root}")
    templates: List[TemplateDef] = []
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        class_name = class_dir.name.strip().lower()
        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() not in ALLOWED_SUFFIXES:
                continue
            gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            templates.append(TemplateDef(
                class_name=class_name, template_name=img_path.stem,
                path=str(img_path), image=gray,
            ))
    return templates


# ---------------------------------------------------------------------------
# Page rendering
# ---------------------------------------------------------------------------


def render_page_gray(
    pdf_bytes: bytes, page_index: int = 0, dpi: int = DEFAULT_DPI,
) -> Tuple[np.ndarray, float]:
    zoom = dpi / 72.0
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        page = doc[page_index]
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom),
                              colorspace=fitz.csGRAY, alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
        return arr.copy(), zoom
    finally:
        doc.close()


# ---------------------------------------------------------------------------
# Core matching (parallelized)
# ---------------------------------------------------------------------------


def _rotate_image(image: np.ndarray, angle: int) -> np.ndarray:
    if angle == 0:
        return image
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if angle == -90:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
    cos_a, sin_a = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    matrix[0, 2] += (new_w - w) / 2.0
    matrix[1, 2] += (new_h - h) / 2.0
    return cv2.warpAffine(image, matrix, (new_w, new_h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def _scale_image(image: np.ndarray, scale: float) -> np.ndarray:
    if abs(scale - 1.0) < 1e-6:
        return image
    h, w = image.shape[:2]
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(image, (new_w, new_h), interpolation=interp)


def _match_one_variant(
    page_gray: np.ndarray,
    template_def: TemplateDef,
    scale: float,
    rotation: int,
    threshold: float,
    zoom: float,
) -> List[Detection]:
    """Match one template at one scale+rotation. Called in parallel."""
    rotated = _rotate_image(template_def.image, rotation)
    scaled = _scale_image(rotated, scale)
    th, tw = scaled.shape[:2]
    ph, pw = page_gray.shape[:2]

    if th < 5 or tw < 5 or th > ph or tw > pw:
        return []

    result = cv2.matchTemplate(page_gray, scaled, cv2.TM_CCOEFF_NORMED)
    mask = result >= threshold
    if not mask.any():
        return []

    locs = np.where(mask)
    scores = result[locs]
    detections: List[Detection] = []
    for y_pos, x_pos, score in zip(locs[0], locs[1], scores):
        px, py = int(x_pos), int(y_pos)
        detections.append(Detection(
            class_name=template_def.class_name,
            template_name=template_def.template_name,
            score=float(score),
            x=px / zoom, y=py / zoom,
            width=tw / zoom, height=th / zoom,
            scale=scale, rotation=rotation,
            pixel_bbox=(px, py, tw, th),
        ))
    return detections


def _nms(detections: List[Detection], iou_threshold: float) -> List[Detection]:
    if not detections:
        return []
    sorted_dets = sorted(detections, key=lambda d: d.score, reverse=True)
    kept: List[Detection] = []
    for det in sorted_dets:
        if not any(_iou(det.pixel_bbox, k.pixel_bbox) > iou_threshold for k in kept):
            kept.append(det)
    return kept


def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax0 + aw, bx0 + bw), min(ay0 + ah, by0 + bh)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _filter_blank(detections: List[Detection], page_gray: np.ndarray) -> List[Detection]:
    ph, pw = page_gray.shape[:2]
    kept = []
    for det in detections:
        px, py, pw_b, ph_b = det.pixel_bbox
        x0, y0 = max(0, px), max(0, py)
        x1, y1 = min(pw, px + pw_b), min(ph, py + ph_b)
        if x1 <= x0 or y1 <= y0:
            continue
        region = page_gray[y0:y1, x0:x1]
        _, bw = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        if float(bw.sum()) / (255.0 * bw.size) >= 0.02:
            kept.append(det)
    return kept


# ---------------------------------------------------------------------------
# Main detector class
# ---------------------------------------------------------------------------


class GdtTemplateDetector:
    """Multi-scale, multi-rotation template detector with thread parallelism."""

    def __init__(
        self,
        template_root: str | Path = "assets/gdt/templates",
        *,
        dpi: int = DEFAULT_DPI,
        scales: Sequence[float] = DEFAULT_SCALES,
        rotations: Sequence[int] = DEFAULT_ROTATIONS,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        nms_iou_threshold: float = DEFAULT_NMS_IOU_THRESHOLD,
        max_workers: int = 8,
    ):
        self.template_root = Path(template_root)
        self.dpi = dpi
        self.scales = list(scales)
        self.rotations = list(rotations)
        self.score_threshold = score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_workers = max_workers
        self.templates: List[TemplateDef] = []
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.templates = load_templates(self.template_root)
            self._loaded = True

    def detect(self, pdf_bytes: bytes, *, page_index: int = 0) -> List[Detection]:
        self._ensure_loaded()
        page_gray, zoom = render_page_gray(pdf_bytes, page_index=page_index, dpi=self.dpi)
        return self.detect_on_image(page_gray, zoom=zoom)

    def detect_on_image(self, page_gray: np.ndarray, *, zoom: float = 1.0) -> List[Detection]:
        self._ensure_loaded()
        all_dets: List[Detection] = []

        # Build work items: (template, scale, rotation)
        work = [
            (tpl, scale, rot)
            for tpl in self.templates
            for rot in self.rotations
            for scale in self.scales
        ]

        # Parallel execution
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(
                    _match_one_variant,
                    page_gray, tpl, scale, rot,
                    self.score_threshold, zoom,
                ): (tpl, scale, rot)
                for tpl, scale, rot in work
            }
            for future in as_completed(futures):
                dets = future.result()
                all_dets.extend(dets)

        # Global NMS + blank filter
        result = _nms(all_dets, self.nms_iou_threshold)
        result = _filter_blank(result, page_gray)
        return sorted(result, key=lambda d: d.score, reverse=True)


# ---------------------------------------------------------------------------
# Visualization helper
# ---------------------------------------------------------------------------


def render_detection_overlay(
    pdf_bytes: bytes, detections: List[Detection], *,
    page_index: int = 0, dpi: int = DEFAULT_DPI,
) -> np.ndarray:
    page_gray, zoom = render_page_gray(pdf_bytes, page_index=page_index, dpi=dpi)
    page_bgr = cv2.cvtColor(page_gray, cv2.COLOR_GRAY2BGR)
    class_names = sorted(set(d.class_name for d in detections))
    colors = _gen_colors(len(class_names))
    color_map = dict(zip(class_names, colors))
    for det in detections:
        color = color_map.get(det.class_name, (0, 255, 0))
        px, py, pw, ph = det.pixel_bbox
        cv2.rectangle(page_bgr, (px, py), (px + pw, py + ph), color, 2)
        cv2.putText(page_bgr, f"{det.class_name} {det.score:.2f}",
                    (px, max(py - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return page_bgr


def _gen_colors(n: int) -> List[Tuple[int, int, int]]:
    if n == 0:
        return []
    colors = []
    for i in range(n):
        hue = int(180 * i / n)
        hsv = np.array([[[hue, 200, 220]]], dtype=np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        colors.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    return colors


__all__ = [
    "Detection", "GdtTemplateDetector", "TemplateDef",
    "load_templates", "render_detection_overlay", "render_page_gray",
]
