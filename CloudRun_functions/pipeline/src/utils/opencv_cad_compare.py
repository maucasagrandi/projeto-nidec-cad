"""OpenCV-based CAD drawing comparison pipeline.

This module replaces the LLM-based comparison with a purely computer-vision approach:

1. PDF → high-res image (PyMuPDF at 300 DPI)
2. Title block detection (structural line/corner detection in lower-right region)
3. Homography estimation from title block to align drawing 2 → drawing 1
4. Pixel-level difference detection with morphological cleanup
5. Red semi-transparent bounding boxes over changed regions

The title block (lower-right table with consistent structure across revisions) serves
as the geometric anchor. Its shape doesn't change between revisions, making it ideal
for estimating the transformation needed to perfectly overlay two drawings.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Optional

import cv2
import fitz  # PyMuPDF
import numpy as np
from PIL import Image


# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class CompareConfig:
    """Configuration for the CAD comparison pipeline."""

    # PDF rasterization
    dpi: int = 300

    # Title block detection
    title_block_search_fraction: float = 0.45
    """Fraction of image (from right and bottom) to search for the title block."""

    hough_threshold: int = 100
    """Minimum votes for Hough line detection."""

    min_line_length_fraction: float = 0.03
    """Min line length as fraction of image diagonal."""

    max_line_gap: int = 10
    """Max gap in pixels for Hough line segments."""

    # Alignment
    homography_method: int = cv2.RANSAC
    ransac_threshold: float = 5.0

    # Difference detection
    diff_threshold: int = 40
    """Pixel intensity threshold for detecting changes (0-255).
    Higher = ignores subtle rendering differences (dash-dot offsets, arrow tips).
    40 is tuned to catch text/geometry changes while ignoring line rendering noise."""

    morph_kernel_size: int = 9
    """Kernel size for morphological operations."""

    morph_close_iterations: int = 3
    morph_dilate_iterations: int = 2

    min_contour_area: int = 200
    """Minimum contour area (in pixels) to consider as a real difference."""

    min_divergence_pct: float = 8.0
    """Minimum divergence percentage within a bbox to keep the detection.
    Filters out boxes where only a tiny fraction of pixels actually differ
    (e.g., arrow tips, slight line offsets). Range: 0-100."""

    # Visualization
    highlight_color: tuple[int, int, int] = (0, 0, 255)
    """BGR color for the highlight boxes (default: red)."""

    highlight_alpha: float = 0.35
    """Transparency for the highlight overlay."""

    box_padding: int = 8
    """Padding around each detected difference bounding box."""

    merge_distance: int = 50
    """Maximum gap (in pixels) between two bounding boxes to merge them into one.
    Nearby boxes often correspond to the same modification (e.g., a note with
    multiple text lines). Set to 0 to disable merging."""

    # --- Alignment refinement (opt4) ---
    ecc_refine: bool = False
    """Refine the title-block homography with an ECC (Enhanced Correlation
    Coefficient) pass over the whole page, correcting a residual *global*
    translation/affine shift. It is validated and only kept if it measurably
    reduces the diff (see ecc_min_improvement); otherwise the original alignment
    is preserved, so enabling it is always safe.

    Disabled by default: on the representative CAD set the residual is per-glyph
    rendering noise (different PDF anti-aliasing), not a global transform, so ECC
    converges to near-identity and is rejected while roughly doubling runtime.
    Enable it for inputs known to have a genuine global misregistration."""

    ecc_max_iterations: int = 50
    """Maximum iterations for the ECC solver."""

    ecc_epsilon: float = 1e-4
    """Convergence threshold for the ECC solver."""

    ecc_downscale_max_dim: int = 1200
    """ECC is estimated on a downscaled copy (longest side capped here) for speed
    and robustness, then the translation/affine is rescaled to full resolution."""

    ecc_min_improvement: float = 0.03
    """Minimum relative reduction in changed-pixel count required to accept the
    ECC-refined alignment over the original. Guards against mis-convergence."""

    # --- Frame-border handling ---
    ignore_frame_border: bool = True
    """Suppress difference pixels on the drawing's outer frame/border. The frame
    is a continuous rectangle that, when slightly misregistered, produces one
    page-spanning connected contour. It is almost never a real change."""

    frame_border_fraction: float = 0.02
    """Thickness of the ignored border band, as a fraction of the smaller page
    dimension. Only the diff mask is suppressed here; the imagery is untouched."""

    # --- Box-size cap + split (opt1) ---
    max_box_area_fraction: float = 0.35
    """A single accepted box may not exceed this fraction of the page area. Boxes
    larger than this are re-detected internally with stricter settings (split),
    not silently discarded, so a real change hidden inside is preserved."""

    max_box_dim_fraction: float = 0.85
    """A single accepted box may not span more than this fraction of the page
    width or height. Complements max_box_area_fraction for long, thin boxes."""

    split_diff_threshold_boost: int = 25
    """When splitting an oversized box, raise diff_threshold by this amount to
    isolate the strongest (most likely real) changes within the region."""

    split_morph_kernel_size: int = 3
    """Smaller morphological kernel used when splitting an oversized box, so the
    internal re-detection does not re-fuse everything into one blob again."""

    # --- Bounded merge (opt3) ---
    max_merge_area_fraction: float = 0.35
    """Union-find merging may not grow a merged box beyond this fraction of the
    page area. Prevents a chain of nearby boxes from collapsing into a giant one.
    Set to 1.0 to disable the bound (legacy behavior)."""


# ==============================================================================
# Result dataclass
# ==============================================================================

@dataclass
class CompareResult:
    """Result of comparing two CAD drawing pages."""

    # Aligned images (after homography)
    image1: np.ndarray
    """Original drawing (reference), as BGR numpy array."""

    image2_aligned: np.ndarray
    """Revised drawing after alignment to match image1."""

    # Difference visualization
    diff_highlighted: np.ndarray
    """Image2 with red-transparent boxes over detected differences."""

    # Detected difference regions
    diff_bboxes: list[tuple[int, int, int, int]]
    """List of (x, y, w, h) bounding boxes of detected differences."""

    diff_divergences: list[float] = field(default_factory=list)
    """Divergence percentage (0-100) for each bbox — ratio of changed pixels within the box."""

    diff_subboxes: list[list[tuple[int, int, int, int]]] = field(default_factory=list)
    """For each merged group box in diff_bboxes, the list of individual member
    boxes that were merged into it. A single-member group has a one-element list.
    Used to render sub-differences and per-sub-difference descriptions."""

    # Alignment metadata
    homography_matrix: Optional[np.ndarray] = None
    """3x3 homography matrix used for alignment (None if alignment skipped)."""

    title_block_bbox1: Optional[tuple[int, int, int, int]] = None
    """Title block bounding box in image1 (x, y, w, h)."""

    title_block_bbox2: Optional[tuple[int, int, int, int]] = None
    """Title block bounding box in image2 (x, y, w, h)."""

    alignment_score: float = 0.0
    """Quality score of the alignment (0-1, based on inlier ratio)."""

    @property
    def num_differences(self) -> int:
        return len(self.diff_bboxes)

    def diff_highlighted_pil(self) -> Image.Image:
        """Return the highlighted diff image as PIL Image."""
        return Image.fromarray(cv2.cvtColor(self.diff_highlighted, cv2.COLOR_BGR2RGB))

    def diff_highlighted_base64(self) -> str:
        """Return the highlighted diff image as base64 PNG."""
        pil_img = self.diff_highlighted_pil()
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")


# ==============================================================================
# Step 1: PDF to Image
# ==============================================================================

def pdf_page_to_cv2(pdf_bytes: bytes, page_index: int = 0, dpi: int = 300) -> np.ndarray:
    """Convert a single PDF page to a high-resolution BGR numpy array.

    Args:
        pdf_bytes: Raw PDF file bytes.
        page_index: Which page to extract (0-indexed).
        dpi: Resolution for rasterization.

    Returns:
        BGR numpy array of the rendered page.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        if page_index >= len(doc):
            raise ValueError(f"Page {page_index} not found (PDF has {len(doc)} pages)")
        page = doc[page_index]
        matrix = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=matrix)
        # Convert to numpy array (RGB)
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        # Handle alpha channel if present
        if pix.n == 4:
            img_array = img_array[:, :, :3]
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        return img_bgr
    finally:
        doc.close()


def pdf_all_pages_to_cv2(pdf_bytes: bytes, dpi: int = 300) -> list[np.ndarray]:
    """Convert all pages of a PDF to BGR numpy arrays."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []
    try:
        for i in range(len(doc)):
            page = doc[i]
            matrix = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=matrix)
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            if pix.n == 4:
                img_array = img_array[:, :, :3]
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            images.append(img_bgr)
        return images
    finally:
        doc.close()


# ==============================================================================
# Step 2: Title Block Detection
# ==============================================================================

def _detect_title_block_region(
    img: np.ndarray,
    config: CompareConfig,
) -> Optional[tuple[int, int, int, int]]:
    """Detect the title block bounding box in a CAD drawing image.

    The title block is located in the lower-right corner and consists of a
    rectangular grid of horizontal and vertical lines forming a table structure.

    Strategy:
    1. Crop to the lower-right region (where title blocks are found)
    2. Detect strong horizontal and vertical lines via morphological line extraction
    3. Find the dense cluster of intersecting lines that forms the title block
    4. Return the bounding box of that cluster

    Args:
        img: BGR image of the full CAD drawing.
        config: Pipeline configuration.

    Returns:
        (x, y, w, h) bounding box in full-image coordinates, or None if not found.
    """
    h, w = img.shape[:2]
    frac = config.title_block_search_fraction

    # Crop to lower-right region
    x_start = int(w * (1 - frac))
    y_start = int(h * (1 - frac))
    crop = img[y_start:, x_start:]
    crop_h, crop_w = crop.shape[:2]

    # Convert to grayscale and threshold (binary inverse for dark lines on white)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Detect horizontal lines using a wide horizontal kernel
    horiz_kernel_len = max(40, crop_w // 8)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_kernel_len, 1))
    horiz_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_kernel, iterations=1)

    # Detect vertical lines using a tall vertical kernel
    vert_kernel_len = max(40, crop_h // 8)
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_kernel_len))
    vert_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vert_kernel, iterations=1)

    # Combine horizontal + vertical lines
    grid_mask = cv2.bitwise_or(horiz_lines, vert_lines)

    # Dilate slightly to connect nearby line segments
    connect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    grid_mask = cv2.dilate(grid_mask, connect_kernel, iterations=2)

    # Find contours — the title block should be the largest connected region
    contours, _ = cv2.findContours(grid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Find the largest contour by area — this should be the title block
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)

    # Sanity check: title block should be at least 5% of the crop area
    min_area = crop_h * crop_w * 0.05
    if area < min_area:
        return None

    x, y, bw, bh = cv2.boundingRect(largest)

    # Convert back to full-image coordinates
    full_x = x_start + x
    full_y = y_start + y

    return (full_x, full_y, bw, bh)


def _extract_title_block_features(
    img: np.ndarray,
    bbox: tuple[int, int, int, int],
    margin: int = 20,
) -> tuple[list[cv2.KeyPoint], np.ndarray]:
    """Extract ORB features from the title block region for matching.

    Args:
        img: Full BGR image.
        bbox: (x, y, w, h) bounding box of the title block.
        margin: Extra pixels around the bbox to include for context.

    Returns:
        (keypoints, descriptors) from the title block region.
        Keypoint coordinates are in full-image space.
    """
    h, w = img.shape[:2]
    x, y, bw, bh = bbox

    # Add margin
    x0 = max(0, x - margin)
    y0 = max(0, y - margin)
    x1 = min(w, x + bw + margin)
    y1 = min(h, y + bh + margin)

    crop = img[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Use ORB for robust feature detection
    orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8)
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    if keypoints is None or descriptors is None:
        return [], np.array([])

    # Shift keypoints back to full-image coordinates
    shifted_kps = []
    for kp in keypoints:
        new_kp = cv2.KeyPoint(
            x=kp.pt[0] + x0,
            y=kp.pt[1] + y0,
            size=kp.size,
            angle=kp.angle,
            response=kp.response,
            octave=kp.octave,
            class_id=kp.class_id,
        )
        shifted_kps.append(new_kp)

    return shifted_kps, descriptors


# ==============================================================================
# Step 3: Homography Estimation and Alignment
# ==============================================================================

def _estimate_homography(
    kp1: list[cv2.KeyPoint],
    desc1: np.ndarray,
    kp2: list[cv2.KeyPoint],
    desc2: np.ndarray,
    config: CompareConfig,
) -> tuple[Optional[np.ndarray], float]:
    """Estimate homography from image2 to image1 using feature matching.

    Uses BFMatcher with Hamming distance (for ORB binary descriptors) and
    Lowe's ratio test to filter good matches, then RANSAC for robust
    homography estimation.

    Args:
        kp1: Keypoints from image1 title block.
        desc1: Descriptors from image1 title block.
        kp2: Keypoints from image2 title block.
        desc2: Descriptors from image2 title block.
        config: Pipeline configuration.

    Returns:
        (homography_matrix, inlier_ratio) or (None, 0.0) if estimation fails.
    """
    if desc1 is None or desc2 is None or len(desc1) < 4 or len(desc2) < 4:
        return None, 0.0

    # Match descriptors using BFMatcher with Hamming distance (binary descriptors)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(desc2, desc1, k=2)

    # Lowe's ratio test
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    if len(good_matches) < 10:
        return None, 0.0

    # Extract matched point coordinates
    pts2 = np.float32([kp2[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts1 = np.float32([kp1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Estimate homography with RANSAC
    H, mask = cv2.findHomography(pts2, pts1, config.homography_method, config.ransac_threshold)

    if H is None:
        return None, 0.0

    # Compute inlier ratio as alignment quality metric
    inlier_ratio = float(mask.sum()) / len(mask) if mask is not None else 0.0

    return H, inlier_ratio


def _align_image(
    img2: np.ndarray,
    H: np.ndarray,
    target_shape: tuple[int, int],
) -> np.ndarray:
    """Warp image2 using homography to align with image1.

    Args:
        img2: The second (revised) drawing image (BGR).
        H: 3x3 homography matrix that maps img2 → img1 coordinate space.
        target_shape: (height, width) of the output aligned image.

    Returns:
        Warped image2 aligned to image1's geometry.
    """
    h, w = target_shape
    aligned = cv2.warpPerspective(
        img2, H, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),  # white border for CAD drawings
    )
    return aligned


def _changed_pixel_count(
    img1: np.ndarray, img2_aligned: np.ndarray, threshold: int
) -> int:
    """Count thresholded difference pixels between two BGR images.

    Used as a scalar quality metric to decide whether an alignment refinement
    actually improved registration.
    """
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2_aligned, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    _, th = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    return int(np.count_nonzero(th))


def _refine_alignment_ecc(
    img1: np.ndarray,
    img2_aligned: np.ndarray,
    config: CompareConfig,
) -> tuple[np.ndarray, bool]:
    """Refine registration of an already-aligned revised image using ECC.

    Title-block homography leaves a small (often sub-pixel) global residual that
    lights up every line/character edge across the page. ECC estimates a
    correcting affine warp on a downscaled grayscale copy, which is then applied
    at full resolution. The refinement is validated: it is only kept if it
    reduces the thresholded difference by at least ecc_min_improvement (relative).

    Args:
        img1: Reference image (BGR).
        img2_aligned: Revised image already coarsely aligned to img1 (BGR).
        config: Pipeline configuration.

    Returns:
        (image, refined) where `image` is the refined (or original) aligned image
        and `refined` indicates whether the ECC result was accepted.
    """
    if not config.ecc_refine:
        return img2_aligned, False

    h, w = img1.shape[:2]

    # Downscale for a robust, fast ECC estimate.
    longest = max(h, w)
    scale = min(1.0, config.ecc_downscale_max_dim / longest)
    small_size = (max(1, round(w * scale)), max(1, round(h * scale)))

    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2_aligned, cv2.COLOR_BGR2GRAY)
    if scale < 1.0:
        g1s = cv2.resize(g1, small_size, interpolation=cv2.INTER_AREA)
        g2s = cv2.resize(g2, small_size, interpolation=cv2.INTER_AREA)
    else:
        g1s, g2s = g1, g2

    # Normalize to float32 for ECC.
    g1f = g1s.astype(np.float32) / 255.0
    g2f = g2s.astype(np.float32) / 255.0

    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        config.ecc_max_iterations,
        config.ecc_epsilon,
    )

    try:
        _cc, warp_matrix = cv2.findTransformECC(
            g1f, g2f, warp_matrix, cv2.MOTION_AFFINE, criteria, None, 5
        )
    except cv2.error:
        # ECC failed to converge — keep the original alignment.
        return img2_aligned, False

    # Rescale the translation components from the downscaled estimate to full res.
    if scale < 1.0:
        warp_matrix[0, 2] /= scale
        warp_matrix[1, 2] /= scale

    refined = cv2.warpAffine(
        img2_aligned,
        warp_matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )

    # Validate: only accept if it reduces the difference meaningfully.
    before = _changed_pixel_count(img1, img2_aligned, config.diff_threshold)
    after = _changed_pixel_count(img1, refined, config.diff_threshold)
    if before <= 0:
        return img2_aligned, False
    improvement = (before - after) / before
    if improvement >= config.ecc_min_improvement:
        return refined, True
    return img2_aligned, False


def _suppress_frame_border(mask: np.ndarray, config: CompareConfig) -> np.ndarray:
    """Zero out difference pixels on the outer frame/border band of the page.

    The drawing frame is a continuous rectangle; a slight registration residual
    turns it into a single page-spanning connected component. Suppressing a thin
    border band removes that artifact without touching interior content.

    Args:
        mask: Binary difference mask (uint8, 0/255).
        config: Pipeline configuration.

    Returns:
        The mask with the border band cleared (a copy).
    """
    if not config.ignore_frame_border:
        return mask

    h, w = mask.shape[:2]
    band = max(1, round(min(h, w) * config.frame_border_fraction))
    if band * 2 >= min(h, w):
        return mask

    out = mask.copy()
    out[:band, :] = 0
    out[h - band:, :] = 0
    out[:, :band] = 0
    out[:, w - band:] = 0
    return out


# ==============================================================================
# Step 4: Difference Detection
# ==============================================================================

def _boxes_are_close(box1: tuple, box2: tuple, max_gap: int) -> bool:
    """Check if two bounding boxes are within max_gap pixels of each other."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    # Compute gap between the two boxes (negative = overlapping)
    gap_x = max(0, max(x1, x2) - min(x1 + w1, x2 + w2))
    gap_y = max(0, max(y1, y2) - min(y1 + h1, y2 + h2))
    return gap_x <= max_gap and gap_y <= max_gap


def _merge_nearby_boxes(
    bboxes: list[tuple[int, int, int, int]],
    divergences: list[float],
    thresh_img: np.ndarray,
    max_gap: int,
    max_area_fraction: float = 1.0,
) -> tuple[
    list[tuple[int, int, int, int]],
    list[float],
    list[list[tuple[int, int, int, int]]],
]:
    """Merge bounding boxes that are within max_gap pixels of each other.

    Greedily merges overlapping/close boxes, but refuses any merge that would
    grow the resulting box beyond max_area_fraction of the page. Recomputes
    divergence for merged boxes and preserves the individual member boxes that
    formed each merged group.

    Args:
        bboxes: List of (x, y, w, h) bounding boxes.
        divergences: List of divergence percentages per bbox.
        thresh_img: Binary threshold image for recomputing divergence.
        max_gap: Maximum pixel gap to consider boxes as part of the same group.
        max_area_fraction: A merge is rejected if the resulting union box would
            exceed this fraction of the page area. This prevents a chain of
            nearby boxes from collapsing into a single page-spanning box. Use
            1.0 to disable the bound (legacy behavior).

    Returns:
        (merged_bboxes, merged_divergences, merged_subboxes) where merged_subboxes[i]
        is the list of original member boxes that were merged into merged_bboxes[i].
        A group with a single member has a one-element sub-box list.
    """
    h_img, w_img = thresh_img.shape[:2]
    page_area = h_img * w_img
    area_cap = max_area_fraction * page_area

    def _union_box(a: tuple, b: tuple) -> tuple[int, int, int, int]:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x_min = min(ax, bx)
        y_min = min(ay, by)
        x_max = max(ax + aw, bx + bw)
        y_max = max(ay + ah, by + bh)
        return (x_min, y_min, x_max - x_min, y_max - y_min)

    # Greedy iterative merging: repeatedly merge a close pair, but only if the
    # resulting box stays within the area cap. Because a rejected pair is left
    # untouched, an oversized cluster naturally splits into several bounded boxes.
    # `members[i]` tracks the original boxes that were merged into `boxes[i]`.
    boxes = list(bboxes)
    members: list[list[tuple[int, int, int, int]]] = [[b] for b in bboxes]
    changed = True
    while changed and len(boxes) > 1:
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if not _boxes_are_close(boxes[i], boxes[j], max_gap):
                    continue
                candidate = _union_box(boxes[i], boxes[j])
                if candidate[2] * candidate[3] > area_cap:
                    # Merging these would create an oversized box — keep separate.
                    continue
                boxes[i] = candidate
                members[i] = members[i] + members[j]
                del boxes[j]
                del members[j]
                changed = True
                break
            if changed:
                break

    # Recompute divergence for each final box from the threshold mask.
    merged_bboxes = []
    merged_divergences = []
    merged_subboxes = []
    for (x, y, w, h), group_members in zip(boxes, members):
        x0, y0 = max(0, x), max(0, y)
        region = thresh_img[y0:y0 + h, x0:x0 + w]
        total_pixels = region.size
        if total_pixels > 0:
            diff_pixels = int(np.count_nonzero(region))
            div_pct = (diff_pixels / total_pixels) * 100.0
        else:
            div_pct = 0.0
        merged_bboxes.append((x, y, w, h))
        merged_divergences.append(div_pct)
        merged_subboxes.append(group_members)

    return merged_bboxes, merged_divergences, merged_subboxes


def _boxes_from_mask(
    cleaned_mask: np.ndarray,
    thresh_mask: np.ndarray,
    image_shape: tuple[int, int],
    config: CompareConfig,
) -> tuple[
    list[tuple[int, int, int, int]],
    list[float],
    list[tuple[int, int, int, int]],
    list[float],
]:
    """Turn a cleaned binary mask into padded, area-filtered bounding boxes.

    Boxes are split into accepted (divergence >= min_divergence_pct) and
    excluded (below threshold) using the raw thresh_mask to compute divergence.

    Args:
        cleaned_mask: Morphologically cleaned binary mask used for contours.
        thresh_mask: Raw binary threshold mask used for divergence computation.
        image_shape: (height, width) of the source image.
        config: Pipeline configuration.

    Returns:
        (bboxes, divergences, excluded_bboxes, excluded_divergences)
    """
    contours, _ = cv2.findContours(
        cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    bboxes: list[tuple[int, int, int, int]] = []
    divergences: list[float] = []
    excluded_bboxes: list[tuple[int, int, int, int]] = []
    excluded_divergences: list[float] = []
    h, w = image_shape
    padding = config.box_padding

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < config.min_contour_area:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        x = max(0, x - padding)
        y = max(0, y - padding)
        bw = min(w - x, bw + 2 * padding)
        bh = min(h - y, bh + 2 * padding)

        bbox_region = thresh_mask[y:y + bh, x:x + bw]
        total_pixels = bbox_region.size
        if total_pixels == 0:
            continue
        diff_pixels = int(np.count_nonzero(bbox_region))
        divergence_pct = (diff_pixels / total_pixels) * 100.0

        if divergence_pct < config.min_divergence_pct:
            excluded_bboxes.append((x, y, bw, bh))
            excluded_divergences.append(divergence_pct)
        else:
            bboxes.append((x, y, bw, bh))
            divergences.append(divergence_pct)

    return bboxes, divergences, excluded_bboxes, excluded_divergences


def _split_oversized_boxes(
    bboxes: list[tuple[int, int, int, int]],
    divergences: list[float],
    gray1: np.ndarray,
    gray2: np.ndarray,
    config: CompareConfig,
) -> tuple[list[tuple[int, int, int, int]], list[float]]:
    """Break boxes that cover too much of the page into localized sub-boxes.

    A box larger than max_box_area_fraction of the page (or spanning more than
    max_box_dim_fraction of a dimension) is re-analyzed *inside its own region*
    with a higher difference threshold and a smaller morphological kernel. This
    isolates the strongest changes instead of discarding the region — so a real
    change hidden inside an over-merged blob is preserved.

    If the internal re-detection yields nothing (e.g. the whole box was
    registration noise), the box is dropped.

    Args:
        bboxes: Accepted boxes from the first pass.
        divergences: Divergence per accepted box.
        gray1: Grayscale reference image.
        gray2: Grayscale aligned revised image.
        config: Pipeline configuration.

    Returns:
        (bboxes, divergences) with oversized boxes replaced by their sub-boxes.
    """
    h, w = gray1.shape[:2]
    page_area = h * w
    area_cap = config.max_box_area_fraction * page_area
    dim_cap_w = config.max_box_dim_fraction * w
    dim_cap_h = config.max_box_dim_fraction * h

    out_boxes: list[tuple[int, int, int, int]] = []
    out_divs: list[float] = []

    for (x, y, bw, bh), div in zip(bboxes, divergences):
        oversized = (
            bw * bh > area_cap or bw > dim_cap_w or bh > dim_cap_h
        )
        if not oversized:
            out_boxes.append((x, y, bw, bh))
            out_divs.append(div)
            continue

        # Re-detect within the region using stricter settings.
        sub1 = gray1[y:y + bh, x:x + bw]
        sub2 = gray2[y:y + bh, x:x + bw]
        sub_diff = cv2.absdiff(sub1, sub2)
        sub_diff = cv2.GaussianBlur(sub_diff, (5, 5), 0)
        strict_thresh = min(255, config.diff_threshold + config.split_diff_threshold_boost)
        _, sub_th = cv2.threshold(sub_diff, strict_thresh, 255, cv2.THRESH_BINARY)

        ksize = max(1, config.split_morph_kernel_size)
        sub_kernel = np.ones((ksize, ksize), np.uint8)
        sub_clean = cv2.morphologyEx(sub_th, cv2.MORPH_CLOSE, sub_kernel, iterations=1)
        sub_clean = cv2.dilate(sub_clean, sub_kernel, iterations=1)

        sub_boxes, sub_divs, _, _ = _boxes_from_mask(
            sub_clean, sub_th, (bh, bw), config
        )

        if not sub_boxes:
            # Nothing strong enough inside — this was registration noise. Drop it.
            continue

        # Merge the sub-boxes (bounded) and translate back to full-image coords.
        # The final merge in _detect_differences will re-group these, so here we
        # only need the flat post-split boxes (sub-box grouping happens later).
        if len(sub_boxes) > 1:
            sub_boxes, sub_divs, _ = _merge_nearby_boxes(
                sub_boxes, sub_divs, sub_th, config.merge_distance,
                config.max_merge_area_fraction,
            )
        for (sx, sy, sw, sh), sdiv in zip(sub_boxes, sub_divs):
            out_boxes.append((x + sx, y + sy, sw, sh))
            out_divs.append(sdiv)

    return out_boxes, out_divs


def _detect_differences(
    img1: np.ndarray,
    img2_aligned: np.ndarray,
    config: CompareConfig,
) -> tuple[
    list[tuple[int, int, int, int]],
    list[float],
    list[tuple[int, int, int, int]],
    list[float],
    np.ndarray,
    list[list[tuple[int, int, int, int]]],
]:
    """Detect regions of difference between aligned images.

    Pipeline:
    1. Absolute difference in grayscale
    2. Gaussian blur to reduce noise
    3. Binary threshold
    4. Morphological close to fill small gaps
    5. Morphological dilate to merge nearby regions
    6. Find contours and filter by area
    7. Compute divergence percentage per bounding box
    8. Separate into accepted (above threshold) and excluded (below threshold)

    Args:
        img1: Reference image (BGR).
        img2_aligned: Aligned revised image (BGR).
        config: Pipeline configuration.

    Returns:
        (bboxes, divergences, excluded_bboxes, excluded_divergences, binary_mask, subboxes):
            - bboxes: List of (x, y, w, h) for accepted difference regions (merged groups)
            - divergences: List of divergence percentages (0-100) per accepted bbox
            - excluded_bboxes: List of (x, y, w, h) for below-threshold regions
            - excluded_divergences: List of divergence percentages for excluded boxes
            - binary_mask: The cleaned binary mask of differences
            - subboxes: For each accepted bbox, the list of individual member boxes
              that were merged into it (one-element list when unmerged)
    """
    # Convert first and subtract in one channel. Subtracting BGR first creates a
    # full-size 3-channel temporary array, which can exceed available RAM for
    # 300-DPI engineering drawings on Windows.
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2_aligned, cv2.COLOR_BGR2GRAY)
    gray_diff = cv2.absdiff(gray1, gray2)

    # Gaussian blur to reduce rendering noise
    gray_diff = cv2.GaussianBlur(gray_diff, (5, 5), 0)

    # Binary threshold
    _, thresh = cv2.threshold(
        gray_diff, config.diff_threshold, 255, cv2.THRESH_BINARY
    )

    # Suppress the outer frame/border band. A continuous frame rectangle with a
    # small registration residual otherwise becomes one page-spanning contour.
    thresh = _suppress_frame_border(thresh, config)

    # Morphological operations to clean up
    kernel = np.ones(
        (config.morph_kernel_size, config.morph_kernel_size), np.uint8
    )
    cleaned = cv2.morphologyEx(
        thresh, cv2.MORPH_CLOSE, kernel, iterations=config.morph_close_iterations
    )
    cleaned = cv2.dilate(
        cleaned, kernel, iterations=config.morph_dilate_iterations
    )

    # Find contours -> accepted / excluded boxes
    bboxes, divergences, excluded_bboxes, excluded_divergences = _boxes_from_mask(
        cleaned, thresh, img1.shape[:2], config
    )

    # Split any oversized accepted box into localized changes rather than
    # shipping a single page-covering "difference".
    bboxes, divergences = _split_oversized_boxes(
        bboxes, divergences, gray1, gray2, config
    )

    # Merge nearby accepted boxes that likely belong to the same modification,
    # but never grow a merged box past the area bound. Preserve the individual
    # member boxes of each group for sub-difference rendering/description.
    if config.merge_distance > 0 and len(bboxes) > 1:
        bboxes, divergences, subboxes = _merge_nearby_boxes(
            bboxes, divergences, thresh, config.merge_distance,
            config.max_merge_area_fraction,
        )
    else:
        # No merging: each accepted box is its own single-member group.
        subboxes = [[b] for b in bboxes]

    return bboxes, divergences, excluded_bboxes, excluded_divergences, cleaned, subboxes


# ==============================================================================
# Step 5: Visualization
# ==============================================================================

def _draw_diff_highlights(
    img: np.ndarray,
    bboxes: list[tuple[int, int, int, int]],
    divergences: list[float],
    excluded_bboxes: list[tuple[int, int, int, int]],
    excluded_divergences: list[float],
    config: CompareConfig,
) -> np.ndarray:
    """Draw red semi-transparent boxes (accepted) and green boxes (excluded) with divergence labels.

    Args:
        img: Image to annotate (BGR). Will be copied, not modified in-place.
        bboxes: List of (x, y, w, h) bounding boxes for accepted differences.
        divergences: List of divergence percentages per accepted bbox.
        excluded_bboxes: List of (x, y, w, h) for below-threshold detections.
        excluded_divergences: List of divergence percentages for excluded boxes.
        config: Pipeline configuration.

    Returns:
        Annotated image with red (accepted) and green (excluded) highlight boxes and percentage labels.
    """
    output = img.copy()
    overlay = img.copy()

    # Draw accepted boxes (red fill)
    for (x, y, w, h) in bboxes:
        cv2.rectangle(overlay, (x, y), (x + w, y + h), config.highlight_color, -1)

    # Draw excluded boxes (green fill)
    excluded_color = (0, 180, 0)  # green in BGR
    for (x, y, w, h) in excluded_bboxes:
        cv2.rectangle(overlay, (x, y), (x + w, y + h), excluded_color, -1)

    # Blend the overlay with the original
    cv2.addWeighted(overlay, config.highlight_alpha, output, 1 - config.highlight_alpha, 0, output)

    # Scale font relative to image size
    img_h, img_w = img.shape[:2]
    font_scale = max(0.4, min(img_h, img_w) / 3000.0)
    font_thickness = max(1, int(font_scale * 2))

    # Draw borders and labels for accepted boxes (red)
    for (x, y, w, h), div_pct in zip(bboxes, divergences):
        cv2.rectangle(output, (x, y), (x + w, y + h), config.highlight_color, 2)
        _draw_divergence_label(output, x, y, div_pct, config.highlight_color, font_scale, font_thickness)

    # Draw borders and labels for excluded boxes (green)
    for (x, y, w, h), div_pct in zip(excluded_bboxes, excluded_divergences):
        cv2.rectangle(output, (x, y), (x + w, y + h), excluded_color, 2)
        _draw_divergence_label(output, x, y, div_pct, excluded_color, font_scale, font_thickness)

    return output


def _draw_divergence_label(
    img: np.ndarray,
    x: int,
    y: int,
    divergence_pct: float,
    color: tuple[int, int, int],
    font_scale: float,
    font_thickness: int,
) -> None:
    """Draw a divergence percentage label near the top-left of a bbox."""
    label = f"{divergence_pct:.0f}%"
    label_size, baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
    )
    lw, lh = label_size

    # Position: just above the bbox, or inside if too close to top edge
    label_x = x + 3
    label_y = y - 5 if y - 5 > lh else y + lh + 5

    # Background rectangle for readability
    cv2.rectangle(
        img,
        (label_x - 1, label_y - lh - 2),
        (label_x + lw + 2, label_y + baseline + 2),
        (255, 255, 255),
        -1,
    )
    cv2.putText(
        img, label, (label_x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
        color, font_thickness, cv2.LINE_AA,
    )


# ==============================================================================
# Main Pipeline
# ==============================================================================

def compare_cad_pages_opencv(
    pdf1_bytes: bytes,
    pdf2_bytes: bytes,
    page_index: int = 0,
    config: Optional[CompareConfig] = None,
    *,
    include_visualization: bool = True,
) -> CompareResult:
    """Compare two CAD drawing pages using OpenCV-based alignment and diff detection.

    Full pipeline:
    1. Rasterize both PDFs at high DPI
    2. Detect title block in both images
    3. Extract features from title blocks
    4. Estimate homography and align drawing 2 to drawing 1
    5. Detect pixel-level differences
    6. Highlight differences with red-transparent boxes

    Args:
        pdf1_bytes: Raw bytes of the original (reference) PDF.
        pdf2_bytes: Raw bytes of the revised PDF.
        page_index: Which page to compare (0-indexed).
        config: Pipeline configuration. Uses defaults if None.
        include_visualization: Build the unverified OpenCV overlay. The
            integrated LLM flow disables it because only verified highlights
            are customer-facing.

    Returns:
        CompareResult with aligned images, highlighted differences, and metadata.
    """
    if config is None:
        config = CompareConfig()

    # --- Step 1: Rasterize ---
    img1 = pdf_page_to_cv2(pdf1_bytes, page_index, config.dpi)
    img2 = pdf_page_to_cv2(pdf2_bytes, page_index, config.dpi)

    # --- Step 2: Detect title blocks ---
    tb_bbox1 = _detect_title_block_region(img1, config)
    tb_bbox2 = _detect_title_block_region(img2, config)

    homography = None
    alignment_score = 0.0
    img2_aligned = img2

    if tb_bbox1 is not None and tb_bbox2 is not None:
        # --- Step 3: Extract features from title blocks ---
        kp1, desc1 = _extract_title_block_features(img1, tb_bbox1)
        kp2, desc2 = _extract_title_block_features(img2, tb_bbox2)

        # --- Step 4: Estimate homography and align ---
        if len(kp1) >= 10 and len(kp2) >= 10:
            H, score = _estimate_homography(kp1, desc1, kp2, desc2, config)
            if H is not None and score > 0.3:
                homography = H
                alignment_score = score
                target_shape = (img1.shape[0], img1.shape[1])
                img2_aligned = _align_image(img2, H, target_shape)

    # Fallback: if no homography found, just resize img2 to match img1
    if homography is None:
        if img1.shape[:2] != img2.shape[:2]:
            img2_aligned = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        else:
            img2_aligned = img2.copy()

    # --- Step 4b: Sub-pixel alignment refinement (ECC) ---
    # Corrects the small residual left by the title-block homography that
    # otherwise lights up every edge across the page. Kept only if validated.
    img2_aligned, _ecc_used = _refine_alignment_ecc(img1, img2_aligned, config)

    # --- Step 5: Detect differences ---
    bboxes, divergences, excluded_bboxes, excluded_divergences, _, subboxes = _detect_differences(img1, img2_aligned, config)

    # --- Step 6: Visualize ---
    diff_highlighted = (
        _draw_diff_highlights(
            img2_aligned,
            bboxes,
            divergences,
            excluded_bboxes,
            excluded_divergences,
            config,
        )
        if include_visualization
        else np.empty((0, 0, 3), dtype=np.uint8)
    )

    return CompareResult(
        image1=img1,
        image2_aligned=img2_aligned,
        diff_highlighted=diff_highlighted,
        diff_bboxes=bboxes,
        diff_divergences=divergences,
        diff_subboxes=subboxes,
        homography_matrix=homography,
        title_block_bbox1=tb_bbox1,
        title_block_bbox2=tb_bbox2,
        alignment_score=alignment_score,
    )


def compare_cad_all_pages_opencv(
    pdf1_bytes: bytes,
    pdf2_bytes: bytes,
    config: Optional[CompareConfig] = None,
) -> list[CompareResult]:
    """Compare all common pages between two CAD PDFs.

    Args:
        pdf1_bytes: Raw bytes of the original PDF.
        pdf2_bytes: Raw bytes of the revised PDF.
        config: Pipeline configuration. Uses defaults if None.

    Returns:
        List of CompareResult, one per page pair.
    """
    doc1 = fitz.open(stream=pdf1_bytes, filetype="pdf")
    doc2 = fitz.open(stream=pdf2_bytes, filetype="pdf")
    n_pages = min(len(doc1), len(doc2))
    doc1.close()
    doc2.close()

    results = []
    for i in range(n_pages):
        result = compare_cad_pages_opencv(pdf1_bytes, pdf2_bytes, i, config)
        results.append(result)

    return results


# ==============================================================================
# Utility: Save result as image or PDF
# ==============================================================================

def save_result_image(result: CompareResult, output_path: str | Path) -> None:
    """Save the highlighted diff image to a file (PNG or PDF).

    Args:
        result: CompareResult from the comparison pipeline.
        output_path: Destination file path (.png or .pdf).
    """
    output_path = Path(output_path)

    if output_path.suffix.lower() == ".pdf":
        # Save as single-page PDF
        pil_img = result.diff_highlighted_pil()
        pil_img.save(str(output_path), "PDF", resolution=300)
    else:
        # Save as image (PNG, JPG, etc.)
        cv2.imwrite(str(output_path), result.diff_highlighted)


def save_side_by_side(
    result: CompareResult,
    output_path: str | Path,
) -> None:
    """Save a side-by-side comparison image.

    Shows: [Original] | [Revised (aligned) with differences highlighted]

    The highlights (red/green boxes with divergence labels) are drawn directly
    on the revised image, so only two panels are needed for visual inspection.

    Args:
        result: CompareResult from the comparison pipeline.
        output_path: Destination file path.
    """
    panels = [result.image1, result.diff_highlighted]

    # Ensure all panels have the same height
    max_h = max(p.shape[0] for p in panels)
    resized = []
    for p in panels:
        if p.shape[0] != max_h:
            scale = max_h / p.shape[0]
            new_w = int(p.shape[1] * scale)
            p = cv2.resize(p, (new_w, max_h))
        resized.append(p)

    combined = np.hstack(resized)
    output_path = Path(output_path)

    if output_path.suffix.lower() == ".pdf":
        pil_img = Image.fromarray(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
        pil_img.save(str(output_path), "PDF", resolution=150)
    else:
        cv2.imwrite(str(output_path), combined)


# ==============================================================================
# Production Export
# ==============================================================================

@dataclass
class DiffRegion:
    """A single detected difference region for the production JSON manifest."""

    id: str
    """Unique identifier for this region (e.g., 'diff_001')."""

    x: int
    """X coordinate of the bounding box (top-left) in image1 coordinates."""

    y: int
    """Y coordinate of the bounding box (top-left) in image1 coordinates."""

    width: int
    """Width of the bounding box in pixels."""

    height: int
    """Height of the bounding box in pixels."""

    divergence_pct: float
    """Percentage of pixels within the bbox that actually differ (0-100)."""


def export_comparison(
    result: CompareResult,
    output_dir: str | Path,
    page_index: int = 0,
    page_prefix: str = "",
) -> Path:
    """Export production artifacts from a single-page comparison result.

    Produces:
    1. `{prefix}original.png` — hi-res image of drawing 1 (the reference)
    2. `crops/{prefix}<id>.png` — cropped sections from the aligned image 2 (revision),
       one per detected difference region
    3. Entries for the manifest (or standalone `manifest.json` when called directly)

    The bounding boxes in the manifest correspond to coordinates on the original image.
    Each crop is the same region extracted from the aligned revision image.

    Args:
        result: CompareResult from compare_cad_pages_opencv().
        output_dir: Directory to write outputs into (created if needed).
        page_index: Page number for the manifest metadata.
        page_prefix: Filename prefix for multi-page exports (e.g., 'page_01_').

    Returns:
        Path to the manifest.json file.
    """
    import json

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save hi-res original image
    original_path = output_dir / f"{page_prefix}original.png"
    cv2.imwrite(str(original_path), result.image1)

    # 2. Crop regions from aligned image2 and build manifest entries
    regions: list[dict] = []

    for idx, ((x, y, w, h), div_pct) in enumerate(
        zip(result.diff_bboxes, result.diff_divergences), start=1
    ):
        region_id = f"{page_prefix}diff_{idx:03d}"

        # Crop from the aligned revision image
        crop = result.image2_aligned[y : y + h, x : x + w]
        crop_path = crops_dir / f"{region_id}.png"
        cv2.imwrite(str(crop_path), crop)

        regions.append({
            "id": region_id,
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "divergence_pct": round(div_pct, 2),
        })

    # 3. Write manifest JSON
    manifest = {
        "page_index": page_index,
        "image_width": result.image1.shape[1],
        "image_height": result.image1.shape[0],
        "original_file": f"{page_prefix}original.png",
        "alignment": {
            "method": "homography" if result.homography_matrix is not None else "resize",
            "score": round(result.alignment_score, 4),
        },
        "num_regions": len(regions),
        "regions": regions,
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return manifest_path


def export_comparison_all_pages(
    pdf1_bytes: bytes,
    pdf2_bytes: bytes,
    output_dir: str | Path,
    config: Optional[CompareConfig] = None,
) -> Path:
    """Export production artifacts for ALL pages of a multi-page CAD PDF comparison.

    Processes every common page between the two PDFs and produces a unified
    output directory:

    ```
    output_dir/
    ├── manifest.json              # Combined manifest for all pages
    ├── page_01_original.png       # Hi-res image 1, page 1
    ├── page_02_original.png       # Hi-res image 1, page 2
    ├── ...
    └── crops/
        ├── page_01_diff_001.png   # Crop from page 1, region 1
        ├── page_01_diff_002.png   # Crop from page 1, region 2
        ├── page_02_diff_001.png   # Crop from page 2, region 1
        └── ...
    ```

    The manifest contains per-page entries with bounding boxes in each page's
    coordinate system. Region IDs are globally unique across pages.

    Args:
        pdf1_bytes: Raw bytes of the original (reference) PDF.
        pdf2_bytes: Raw bytes of the revised PDF.
        output_dir: Directory to write all outputs into (created if needed).
        config: Pipeline configuration. Uses defaults if None.

    Returns:
        Path to the combined manifest.json file.
    """
    import json

    if config is None:
        config = CompareConfig()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    # Determine page count
    doc1 = fitz.open(stream=pdf1_bytes, filetype="pdf")
    doc2 = fitz.open(stream=pdf2_bytes, filetype="pdf")
    n_pages = min(len(doc1), len(doc2))
    doc1.close()
    doc2.close()

    pages_data: list[dict] = []

    for page_idx in range(n_pages):
        page_prefix = f"page_{page_idx + 1:02d}_"

        # Run comparison for this page
        result = compare_cad_pages_opencv(pdf1_bytes, pdf2_bytes, page_idx, config)

        # Save hi-res original
        original_filename = f"{page_prefix}original.png"
        cv2.imwrite(str(output_dir / original_filename), result.image1)

        # Crop regions and collect metadata
        regions: list[dict] = []
        for idx, ((x, y, w, h), div_pct) in enumerate(
            zip(result.diff_bboxes, result.diff_divergences), start=1
        ):
            region_id = f"{page_prefix}diff_{idx:03d}"

            crop = result.image2_aligned[y : y + h, x : x + w]
            cv2.imwrite(str(crops_dir / f"{region_id}.png"), crop)

            regions.append({
                "id": region_id,
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "divergence_pct": round(div_pct, 2),
            })

        pages_data.append({
            "page_index": page_idx,
            "image_width": result.image1.shape[1],
            "image_height": result.image1.shape[0],
            "original_file": original_filename,
            "alignment": {
                "method": "homography" if result.homography_matrix is not None else "resize",
                "score": round(result.alignment_score, 4),
            },
            "num_regions": len(regions),
            "regions": regions,
        })

    # Write combined manifest
    manifest = {
        "total_pages": n_pages,
        "total_regions": sum(p["num_regions"] for p in pages_data),
        "pages": pages_data,
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return manifest_path
