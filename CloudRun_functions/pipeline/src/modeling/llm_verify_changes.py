"""LLM-based verification of OpenCV-detected differences.

After the OpenCV pipeline detects candidate change regions, this module sends
the original image (for context) plus each cropped region to Gemini to:

1. Classify each region as a true change or false positive
2. Describe what actually changed in each true-positive region

The final output is a structured report with contiguously-indexed true changes
and a side-by-side image highlighting only the verified differences.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


@dataclass
class AnalysisMetadata:
    """LLM usage metadata kept local to avoid eager Gemini client creation."""

    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    model: str
    timestamp: str


# ==============================================================================
# Spatial clustering (dependency-free, numpy only)
# ==============================================================================

# Default COMPLETE-linkage distance threshold in NORMALIZED page units (0..1 on
# each axis, Euclidean). Complete linkage bounds each cluster's DIAMETER by the
# threshold, so a cluster only forms when ALL its members are mutually within
# the threshold — this keeps groups spatially compact (a "region" of the sheet)
# and prevents scattered changes from chaining into one page-spanning blob (as
# single linkage would). ~0.38 gives a handful of compact, balanced regions on a
# dense drawing (no single page-spanning blob). Lower -> more, tighter clusters;
# higher -> fewer, broader clusters.
CLUSTER_DISTANCE_THRESHOLD = 0.38


def _agglomerative_cluster(
    centroids: np.ndarray,
    threshold: float,
) -> list[int]:
    """Complete-linkage agglomerative clustering with a distance threshold.

    Repeatedly merges the two clusters whose COMPLETE-linkage distance (the
    maximum pairwise distance between their members) is smallest, stopping once
    that distance would exceed ``threshold``. Complete linkage bounds each
    cluster's diameter, keeping clusters spatially compact and avoiding the
    chaining that single linkage produces. No fixed number of clusters — the
    count emerges from the layout; isolated points remain their own cluster.

    Args:
        centroids: (N, 2) array of normalized centroids (each coord in 0..1).
        threshold: Max allowed cluster diameter in normalized units.

    Returns:
        List of N integer cluster labels (0-based, assigned in first-seen order).
    """
    n = len(centroids)
    if n == 0:
        return []
    if n == 1:
        return [0]

    # Full pairwise distance matrix (small N; O(N^2) is fine).
    diff = centroids[:, None, :] - centroids[None, :, :]
    dist = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))

    # Active clusters, each a list of member point indices.
    clusters: list[list[int]] = [[i] for i in range(n)]

    def complete_linkage(a: list[int], b: list[int]) -> float:
        # Max pairwise distance between members of a and b.
        return float(dist[np.ix_(a, b)].max())

    while len(clusters) > 1:
        best = None
        best_pair = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                d = complete_linkage(clusters[i], clusters[j])
                if best is None or d < best:
                    best = d
                    best_pair = (i, j)
        if best is None or best > threshold:
            break  # closest merge would exceed the diameter bound — stop
        i, j = best_pair
        clusters[i].extend(clusters[j])
        del clusters[j]

    # Assign labels in order of each cluster's smallest member index (stable,
    # roughly first-appearance order).
    clusters.sort(key=min)
    labels = [0] * n
    for label, members in enumerate(clusters):
        for idx in members:
            labels[idx] = label
    return labels


# ==============================================================================
# Pydantic Models — LLM Response Schema
# ==============================================================================

class RegionVerdict(BaseModel):
    """LLM verdict for a single candidate sub-box (one localized difference)."""

    id: str = Field(description="The sub-box ID (e.g., 'page_01_diff_003')")
    is_true_change: bool = Field(
        description="True if this sub-box contains a real, meaningful difference"
    )
    description: str = Field(
        description=(
            "If is_true_change=true: concise technical description of what changed "
            "in THIS sub-box, comparing the original crop against the revised crop "
            "(e.g., 'Perpendicularity tolerance changed from 0.1 to 0.05'). "
            "If is_true_change=false: brief reason why it's a false positive "
            "(e.g., 'Rendering artifact on dash-dot line')."
        )
    )


class VerificationOutput(BaseModel):
    """Structured output from the LLM verification call."""

    verdicts: list[RegionVerdict] = Field(
        description="One verdict per candidate sub-box, in the same order as provided"
    )


# ==============================================================================
# Result dataclass
# ==============================================================================

@dataclass
class VerifiedChange:
    """A single verified true change (a grouped region), with contiguous indexing."""

    index: int
    """1-based contiguous index for display."""

    original_id: str
    """Original region ID from the OpenCV manifest."""

    x: int
    y: int
    width: int
    height: int
    divergence_pct: float
    description: str
    """Overall summary of the grouped change."""

    sub_boxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    """All detected member boxes (x, y, w, h) inside this group, in full-image
    coordinates. A single-member group has one."""

    sub_differences: list[dict] = field(default_factory=list)
    """One entry per individual difference the LLM described, each a dict:
        {
            "sub_id": "1.2",                 # group_index.sub_ordinal
            "description": "...",            # bullet topic text
            "boxes": [(x, y, w, h), ...],    # the sub-boxes this difference covers
        }
    The sub_id is drawn on each covered box and used as the bullet prefix."""


@dataclass
class VerificationResult:
    """Full result of the verification pipeline for one page."""

    page_index: int
    true_changes: list[VerifiedChange]
    false_positive_ids: list[str]
    metadata: AnalysisMetadata | None = None

    # Images
    image_original: np.ndarray | None = None
    image_revised_aligned: np.ndarray | None = None
    image_highlighted: np.ndarray | None = None

    @property
    def num_true_changes(self) -> int:
        return len(self.true_changes)

    @property
    def num_false_positives(self) -> int:
        return len(self.false_positive_ids)

    def report_text(self) -> str:
        """Generate a plain-text summary report of verified changes."""
        lines = []
        lines.append(f"CAD Comparison Report — Page {self.page_index + 1}")
        lines.append("=" * 50)
        lines.append(f"Total candidates analyzed: {self.num_true_changes + self.num_false_positives}")
        lines.append(f"True changes: {self.num_true_changes}")
        lines.append(f"False positives filtered: {self.num_false_positives}")
        lines.append("")

        if self.true_changes:
            lines.append("Changes Identified:")
            lines.append("-" * 50)
            for change in self.true_changes:
                lines.append(f"  [{change.index}] {change.description}")
                for sub in (change.sub_differences or []):
                    lines.append(f"        [{sub['sub_id']}] {sub['description']}")
                lines.append(f"       Location: ({change.x}, {change.y}) "
                             f"Size: {change.width}x{change.height} "
                             f"Divergence: {change.divergence_pct:.1f}%")
                lines.append("")
        else:
            lines.append("No meaningful changes detected.")

        return "\n".join(lines)

    def report_json(self) -> dict:
        """Generate a JSON-serializable report."""
        return {
            "page_index": self.page_index,
            "total_candidates": self.num_true_changes + self.num_false_positives,
            "num_true_changes": self.num_true_changes,
            "num_false_positives": self.num_false_positives,
            "changes": [
                {
                    "index": c.index,
                    "original_id": c.original_id,
                    "description": c.description,
                    "x": c.x,
                    "y": c.y,
                    "width": c.width,
                    "height": c.height,
                    "divergence_pct": c.divergence_pct,
                    "sub_boxes": [list(b) for b in c.sub_boxes],
                    "sub_differences": [
                        {
                            "sub_id": s["sub_id"],
                            "description": s["description"],
                            "boxes": [list(b) for b in s.get("boxes", [])],
                        }
                        for s in c.sub_differences
                    ],
                }
                for c in self.true_changes
            ],
            "false_positive_ids": self.false_positive_ids,
        }


# ==============================================================================
# Prompt
# ==============================================================================

VERIFICATION_PROMPT = """\
You are an expert CAD drawing analyst. You are given:

1. A full-page image of the ORIGINAL CAD drawing (for overall context).
2. For each candidate: a crop from the ORIGINAL and a crop from the REVISED
   drawing covering the SAME area. An ORANGE rectangle marks the TARGET region.
   The crop includes a margin around the target so you can see surrounding
   context (the dimension line, tolerance frame, note, table cell, etc.), but
   the change you must judge and describe is the one INSIDE the orange rectangle.
   Any difference visible only OUTSIDE the orange rectangle belongs to a
   neighboring candidate — treat it as context and ignore it.

Each candidate is identified by an "id" and its bounding box location on the full
page. Analyze each candidate INDEPENDENTLY: compare the ORIGINAL crop against the
REVISED crop, focusing on the orange-marked region, and decide:
- TRUE CHANGE: the content INSIDE the orange rectangle actually differs. This
  includes BOTH annotation changes AND geometry/shape changes:
    * Annotations: text, dimensions, tolerances, GD&T symbols, notes, table
      content, revision block, datum labels.
    * GEOMETRY / SHAPE: the drawn part itself changing — an altered profile or
      contour, a moved/resized/added/removed feature (hole, boss, fillet,
      chamfer, rib, slot, cut), a changed edge or outline, hatching/section
      changes, a line that appears, disappears, or shifts.
    * Any content present in one crop but absent in the other.
- FALSE POSITIVE: inside the orange rectangle the crops are effectively identical
  or differ only by rendering artifacts (anti-aliasing, slight line-weight
  variation, sub-pixel shifts). If the ONLY visible difference is outside the
  orange rectangle, this candidate is a FALSE POSITIVE (that change belongs to a
  neighboring candidate).

For each candidate provide:
  - "id": the candidate id, exactly as given.
  - "is_true_change": true or false.
  - "description": if true, a concise technical description of what changed
    INSIDE the orange rectangle (e.g., 'Dimension changed from 26 to 22',
    'Perpendicularity tolerance tightened from 0.1 to 0.05', 'Fillet added at the
    lower-left corner of the flange', 'Boss profile widened / contour reshaped').
    Use the surrounding context to identify WHAT the changed element is, but the
    change itself must be inside the marked region. Describe ONLY what you can
    visually confirm — do NOT invent changes. If false, a brief reason for the
    false positive.

Accuracy is critical: the description of each candidate MUST correspond to the
change inside that candidate's orange rectangle. Do not describe changes that are
only visible outside it (those belong to other candidates).

The candidates to analyze are:
{regions_json}

Respond with exactly one verdict per candidate, in the same order as provided.
"""


# ==============================================================================
# LLM Call
# ==============================================================================

def _encode_image_to_base64(img: np.ndarray) -> bytes:
    """Encode a BGR numpy array to PNG bytes."""
    _, buffer = cv2.imencode(".png", img)
    return buffer.tobytes()


def _draw_roi_marker(
    crop: np.ndarray,
    target_local: tuple[int, int, int, int],
    *,
    color: tuple[int, int, int] = (0, 165, 255),  # orange (BGR)
) -> np.ndarray:
    """Outline the target sub-box inside a padded crop (region-of-interest).

    The crop includes padding so the model has local context, but that padding
    may also contain neighboring changes. Drawing a marker around the exact
    target region tells the model which element to describe, so a neighbor's
    change seen in the margin is treated as context, not as this box's change.

    Args:
        crop: The padded crop (BGR).
        target_local: (x, y, w, h) of the target sub-box in CROP-LOCAL coords.
        color: BGR outline color.

    Returns:
        A copy of the crop with the target region outlined.
    """
    img = crop.copy()
    ch, cw = img.shape[:2]
    tx, ty, tw, th = target_local
    x0 = max(0, tx)
    y0 = max(0, ty)
    x1 = min(cw, tx + tw)
    y1 = min(ch, ty + th)
    if x1 <= x0 or y1 <= y0:
        return img
    thickness = max(2, cw // 300)
    cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness)
    return img


# Maximum sub-box candidates per LLM call. A page above this splits into a few
# calls (each still carrying the full-page context) to bound the image count.
MAX_CANDIDATES_PER_CALL = 40


def _verify_batch(
    client,
    types,
    original_image: np.ndarray,
    crops_original: list[tuple[str, np.ndarray]],
    crops_revised: list[tuple[str, np.ndarray]],
    regions_metadata: list[dict],
    model: str,
) -> tuple[VerificationOutput, object]:
    """Run a single LLM verification call over a batch of sub-box candidates."""
    # Per-candidate manifest (id + location/size on the full page).
    regions_desc = [
        {
            "id": meta["id"],
            "location": f"({meta['x']}, {meta['y']})",
            "size": f"{meta['width']}x{meta['height']}",
            "divergence_pct": meta.get("divergence_pct", 0.0),
        }
        for meta in regions_metadata
    ]
    regions_json = json.dumps(regions_desc, indent=2)

    content_parts = []

    # 1. Full original image for overall context (once per call).
    content_parts.append(
        types.Part.from_bytes(
            data=_encode_image_to_base64(original_image), mime_type="image/png"
        )
    )
    content_parts.append(
        types.Part.from_text(
            text=(
                "Above: the full ORIGINAL drawing (for overall context).\n\n"
                "Below: for each candidate, a padded ORIGINAL crop and REVISED crop "
                "of the SAME area. An ORANGE rectangle marks the TARGET region for "
                "that candidate. Describe ONLY the change inside the orange rectangle; "
                "anything outside it is context from neighboring candidates and must "
                "be ignored. Compare each candidate's pair independently.\n"
            )
        )
    )

    # 2. One original+revised crop pair per candidate, each with the target
    #    region-of-interest outlined so the model ignores neighboring changes
    #    that fall inside the padding margin.
    meta_by_id = {m["id"]: m for m in regions_metadata}
    for (cand_id, crop_orig), (_, crop_rev) in zip(crops_original, crops_revised):
        target_local = tuple(
            meta_by_id.get(cand_id, {}).get(
                "target_local", (0, 0, crop_rev.shape[1], crop_rev.shape[0])
            )
        )
        marked_orig = _draw_roi_marker(crop_orig, target_local)
        marked_rev = _draw_roi_marker(crop_rev, target_local)
        content_parts.append(
            types.Part.from_text(text=f"\n--- Candidate {cand_id} — ORIGINAL crop (target in orange): ---")
        )
        content_parts.append(
            types.Part.from_bytes(data=_encode_image_to_base64(marked_orig), mime_type="image/png")
        )
        content_parts.append(
            types.Part.from_text(text=f"--- Candidate {cand_id} — REVISED crop (target in orange): ---")
        )
        content_parts.append(
            types.Part.from_bytes(data=_encode_image_to_base64(marked_rev), mime_type="image/png")
        )

    # 3. The prompt.
    content_parts.append(
        types.Part.from_text(text=VERIFICATION_PROMPT.format(regions_json=regions_json))
    )

    response = client.models.generate_content(
        model=model,
        contents=content_parts,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=VerificationOutput,
        ),
    )
    parsed = VerificationOutput.model_validate_json(response.text)
    return parsed, response.usage_metadata


def verify_changes_with_llm(
    original_image: np.ndarray,
    crops_original: list[tuple[str, np.ndarray]],
    crops_revised: list[tuple[str, np.ndarray]],
    regions_metadata: list[dict],
    model: str = "gemini-2.5-flash",
) -> tuple[VerificationOutput, AnalysisMetadata]:
    """Verify each sub-box candidate: one padded crop pair per candidate.

    All candidates are sent in a single batched call per page (full-page image +
    all crop pairs). If a page has more than MAX_CANDIDATES_PER_CALL candidates,
    it is split into a few calls (still far fewer than one-per-change), and the
    verdicts + token usage are merged.

    Args:
        original_image: Full-page BGR image of the original drawing (context).
        crops_original: List of (candidate_id, padded_original_crop) pairs.
        crops_revised: List of (candidate_id, padded_revised_crop) pairs.
        regions_metadata: Per-candidate dicts (id, x, y, width, height, ...).
        model: Gemini model to use.

    Returns:
        (VerificationOutput, AnalysisMetadata)
    """
    import os

    from dotenv import load_dotenv
    from google import genai
    from google.genai import types

    load_dotenv()

    project = os.getenv("GCP_PROJECT_ID", "acim-global-data-lake-sandbox")
    location = os.getenv("GCP_REGION", "global")
    client = genai.Client(vertexai=True, project=project, location=location)

    start_time = time.time()

    n = len(crops_revised)
    # Split into batches only when needed.
    batch_size = MAX_CANDIDATES_PER_CALL
    num_batches = max(1, (n + batch_size - 1) // batch_size)

    all_verdicts = []
    total_tokens = prompt_tokens = completion_tokens = 0

    for b in range(num_batches):
        lo = b * batch_size
        hi = min(n, lo + batch_size)
        logger.info(
            "Verifying candidates %d-%d of %d (batch %d/%d) with %s...",
            lo + 1, hi, n, b + 1, num_batches, model,
        )
        parsed, usage = _verify_batch(
            client, types, original_image,
            crops_original[lo:hi], crops_revised[lo:hi], regions_metadata[lo:hi],
            model,
        )
        all_verdicts.extend(parsed.verdicts)
        if usage is not None:
            total_tokens += usage.total_token_count or 0
            prompt_tokens += usage.prompt_token_count or 0
            completion_tokens += usage.candidates_token_count or 0

    combined = VerificationOutput(verdicts=all_verdicts)

    latency_ms = (time.time() - start_time) * 1000
    metadata = AnalysisMetadata(
        total_tokens=total_tokens,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=latency_ms,
        model=model,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )

    logger.info(
        "Verification complete: %d true changes, %d false positives",
        sum(1 for v in combined.verdicts if v.is_true_change),
        sum(1 for v in combined.verdicts if not v.is_true_change),
    )

    return combined, metadata


# ==============================================================================
# Post-processing: re-index and build result
# ==============================================================================

def _build_verified_result(
    page_index: int,
    regions_metadata: list[dict],
    llm_output: VerificationOutput,
    metadata: AnalysisMetadata,
    page_width: int = 0,
    page_height: int = 0,
    cluster_threshold: float = CLUSTER_DISTANCE_THRESHOLD,
) -> VerificationResult:
    """Process LLM verdicts into a VerificationResult.

    Grouping (which changes share a report page) is decided by AGGLOMERATIVE
    CLUSTERING of the surviving true-change boxes' centroids in normalized page
    coordinates — NOT by the OpenCV proximity-merge. This keeps spatially close
    changes on one page while avoiding a lonely tiny box occupying a whole page
    only because the OpenCV merge happened to isolate it.

    Args:
        page_index: Page number.
        regions_metadata: Per-sub-box dicts from the manifest.
        llm_output: Parsed LLM structured output (one verdict per sub-box).
        metadata: LLM call metadata.
        page_width: Full-page width in px (for centroid normalization).
        page_height: Full-page height in px (for centroid normalization).
        cluster_threshold: Single-linkage distance threshold in normalized units.

    Returns:
        VerificationResult with clustered, contiguously-indexed true changes.
    """
    meta_by_id = {r["id"]: r for r in regions_metadata}

    def _union(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
        xs0 = min(b[0] for b in boxes)
        ys0 = min(b[1] for b in boxes)
        xs1 = max(b[0] + b[2] for b in boxes)
        ys1 = max(b[1] + b[3] for b in boxes)
        return (xs0, ys0, xs1 - xs0, ys1 - ys0)

    # 1. Collect surviving true-change boxes (each with box, description, div).
    #    Global dedup by (rounded centroid + normalized description) removes
    #    duplicate reports caused by padding capturing a neighbor's change.
    false_positive_ids: list[str] = []
    kept: list[dict] = []  # each: {box, desc, div}
    seen_keys: set[tuple] = set()

    for verdict in llm_output.verdicts:
        if not verdict.is_true_change:
            false_positive_ids.append(verdict.id)
            continue
        desc = (verdict.description or "").strip()
        if not desc:
            continue
        meta = meta_by_id.get(verdict.id, {})
        box = (int(meta.get("x", 0)), int(meta.get("y", 0)),
               int(meta.get("width", 0)), int(meta.get("height", 0)))
        cx = box[0] + box[2] / 2.0
        cy = box[1] + box[3] / 2.0
        desc_key = " ".join(desc.lower().split())
        # Round centroid to ~10px grid so near-identical duplicates collapse.
        dedup_key = (round(cx / 10.0), round(cy / 10.0), desc_key)
        if dedup_key in seen_keys:
            continue
        seen_keys.add(dedup_key)
        kept.append({
            "box": box,
            "desc": desc,
            "div": float(meta.get("divergence_pct", 0.0)),
            "centroid": (cx, cy),
        })

    if not kept:
        return VerificationResult(
            page_index=page_index,
            true_changes=[],
            false_positive_ids=false_positive_ids,
            metadata=metadata,
        )

    # 2. Cluster surviving boxes by normalized centroid.
    pw = float(page_width) if page_width else 1.0
    ph = float(page_height) if page_height else 1.0
    centroids = np.array(
        [[k["centroid"][0] / pw, k["centroid"][1] / ph] for k in kept],
        dtype=np.float64,
    )
    labels = _agglomerative_cluster(centroids, cluster_threshold)

    # 3. Assemble one VerifiedChange per cluster, ordered by first appearance.
    clusters: dict[int, list[dict]] = {}
    cluster_order: list[int] = []
    for item, label in zip(kept, labels):
        if label not in clusters:
            clusters[label] = []
            cluster_order.append(label)
        clusters[label].append(item)

    true_changes: list[VerifiedChange] = []
    contiguous_idx = 1
    for label in cluster_order:
        members = clusters[label]
        member_boxes = [m["box"] for m in members]
        group_box = _union(member_boxes)

        sub_differences = [
            {
                "sub_id": f"{contiguous_idx}.{ordinal}",
                "description": m["desc"],
                "boxes": [m["box"]],
            }
            for ordinal, m in enumerate(members, start=1)
        ]

        if len(sub_differences) == 1:
            summary = sub_differences[0]["description"]
        else:
            summary = f"{len(sub_differences)} differences in this region"

        divs = [m["div"] for m in members]
        avg_div = sum(divs) / len(divs) if divs else 0.0

        true_changes.append(VerifiedChange(
            index=contiguous_idx,
            original_id=f"page_{page_index + 1:02d}_cluster_{label:03d}",
            x=group_box[0],
            y=group_box[1],
            width=group_box[2],
            height=group_box[3],
            divergence_pct=round(avg_div, 2),
            description=summary,
            sub_boxes=member_boxes,
            sub_differences=sub_differences,
        ))
        contiguous_idx += 1

    return VerificationResult(
        page_index=page_index,
        true_changes=true_changes,
        false_positive_ids=false_positive_ids,
        metadata=metadata,
    )


# ==============================================================================
# Final image rendering
# ==============================================================================

def render_verified_highlights(
    image_original: np.ndarray,
    image_revised_aligned: np.ndarray,
    true_changes: list[VerifiedChange],
    highlight_color: tuple[int, int, int] = (0, 0, 255),
    highlight_alpha: float = 0.35,
    max_output_height: int = 1800,
) -> np.ndarray:
    """Render the side-by-side image with only true-change highlights and contiguous ID labels.

    Layout: [Original] | [Revised with red boxes + ID labels]

    Args:
        image_original: Full BGR image of the original drawing.
        image_revised_aligned: Aligned BGR image of the revised drawing.
        true_changes: List of verified true changes with contiguous indices.
        highlight_color: BGR color for boxes (default red).
        highlight_alpha: Transparency for the overlay.
        max_output_height: Maximum height of the side-by-side output. The
            source images remain at detection resolution; only the report
            visualization is reduced.

    Returns:
        Combined side-by-side BGR image.
    """
    if max_output_height < 1:
        raise ValueError("max_output_height must be at least 1")

    source_height = image_revised_aligned.shape[0]
    output_scale = min(1.0, max_output_height / source_height)

    def resize_panel(image: np.ndarray) -> np.ndarray:
        if output_scale >= 1.0:
            return image.copy()
        return cv2.resize(
            image,
            (
                max(1, round(image.shape[1] * output_scale)),
                max(1, round(image.shape[0] * output_scale)),
            ),
            interpolation=cv2.INTER_AREA,
        )

    # Lilac (BGR) for the individual sub-difference boxes drawn inside a group.
    subbox_color = (219, 112, 147)  # medium purple / lilac in BGR
    subbox_alpha = 0.30

    # Reduce the panels before creating overlays or concatenating. A pair of
    # 300-DPI A3 pages can otherwise require a contiguous allocation >200 MB.
    original = resize_panel(image_original)
    revised = resize_panel(image_revised_aligned)
    overlay = revised.copy()

    def _scale_box(bx, by, bw, bh):
        return (
            round(bx * output_scale),
            round(by * output_scale),
            max(1, round(bw * output_scale)),
            max(1, round(bh * output_scale)),
        )

    def _has_multiple_subs(change) -> bool:
        # Only draw lilac sub-boxes when the group holds more than one distinct
        # difference (a single-difference group's box == the red group box).
        subs = getattr(change, "sub_differences", None) or []
        return len(subs) > 1

    # Fill: lilac sub-difference boxes first, then the red group box over them.
    for change in true_changes:
        if not _has_multiple_subs(change):
            continue
        for sub in change.sub_differences:
            for (sbx, sby, sbw, sbh) in sub.get("boxes", []):
                sx, sy, sw, sh = _scale_box(sbx, sby, sbw, sbh)
                cv2.rectangle(overlay, (sx, sy), (sx + sw, sy + sh), subbox_color, -1)

    for change in true_changes:
        x, y, w, h = _scale_box(change.x, change.y, change.width, change.height)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), highlight_color, -1)

    cv2.addWeighted(overlay, highlight_alpha, revised, 1 - highlight_alpha, 0, revised)

    # Draw borders and ID labels
    img_h, img_w = revised.shape[:2]
    font_scale = max(0.5, min(img_h, img_w) / 2500.0)
    font_thickness = max(1, int(font_scale * 2.5))
    sub_font_scale = max(0.4, font_scale * 0.8)
    sub_font_thickness = max(1, int(sub_font_scale * 2.0))

    def _draw_label(text: str, bx: int, by: int, color: tuple[int, int, int],
                    fscale: float, fthick: float) -> None:
        (lw, lh), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, fscale, fthick
        )
        label_x = bx + 3
        label_y = by - 6 if by - 6 > lh else by + lh + 6
        cv2.rectangle(
            revised,
            (label_x - 2, label_y - lh - 3),
            (label_x + lw + 3, label_y + baseline + 3),
            (255, 255, 255),
            -1,
        )
        cv2.putText(
            revised, text, (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, fscale, color, fthick, cv2.LINE_AA,
        )

    # Lilac borders + sub_id labels for each sub-difference (multi-diff groups).
    for change in true_changes:
        if not _has_multiple_subs(change):
            continue
        for sub in change.sub_differences:
            sub_id = sub.get("sub_id", "")
            for (sbx, sby, sbw, sbh) in sub.get("boxes", []):
                sx, sy, sw, sh = _scale_box(sbx, sby, sbw, sbh)
                cv2.rectangle(revised, (sx, sy), (sx + sw, sy + sh), subbox_color, 2)
                if sub_id:
                    _draw_label(sub_id, sx, sy, subbox_color, sub_font_scale, sub_font_thickness)

    # Red group box + numeric group label.
    for change in true_changes:
        x, y, w, h = _scale_box(change.x, change.y, change.width, change.height)
        cv2.rectangle(revised, (x, y), (x + w, y + h), highlight_color, 2)
        _draw_label(str(change.index), x, y, highlight_color, font_scale, font_thickness)

    # Combine side by side: [Original] | [Revised highlighted]
    panels = [original, revised]
    max_h = max(p.shape[0] for p in panels)
    resized = []
    for p in panels:
        if p.shape[0] != max_h:
            scale = max_h / p.shape[0]
            new_w = int(p.shape[1] * scale)
            p = cv2.resize(p, (new_w, max_h))
        resized.append(p)

    return np.hstack(resized)


# ==============================================================================
# Top-level pipeline
# ==============================================================================

def run_verification_pipeline(
    pdf1_bytes: bytes,
    pdf2_bytes: bytes,
    page_index: int = 0,
    model: str = "gemini-2.5-flash",
    opencv_config=None,
) -> VerificationResult:
    """Full pipeline: OpenCV detection → LLM verification → report + image.

    1. Run OpenCV comparison to detect candidate regions
    2. Send original + crops to LLM for verification
    3. Filter to true changes with contiguous indexing
    4. Render final highlighted image

    Args:
        pdf1_bytes: Raw bytes of the original PDF.
        pdf2_bytes: Raw bytes of the revised PDF.
        page_index: Which page to compare.
        model: Gemini model ID.
        opencv_config: Optional CompareConfig for the OpenCV step.

    Returns:
        VerificationResult with report, images, and metadata.
    """
    from src.utils.opencv_cad_compare import CompareConfig, compare_cad_pages_opencv

    if opencv_config is None:
        opencv_config = CompareConfig()

    # Step 1: OpenCV comparison
    logger.info(f"Running OpenCV comparison on page {page_index}...")
    cv_result = compare_cad_pages_opencv(
        pdf1_bytes,
        pdf2_bytes,
        page_index,
        opencv_config,
        include_visualization=False,
    )

    if cv_result.num_differences == 0:
        logger.info("No candidate regions detected by OpenCV.")
        return VerificationResult(
            page_index=page_index,
            true_changes=[],
            false_positive_ids=[],
            image_original=None,
            image_highlighted=render_verified_highlights(
                cv_result.image1, cv_result.image2_aligned, []
            ),
        )

    # Build ONE padded crop pair per SUB-BOX (not per group). Each sub-box is an
    # independent candidate so the LLM's description maps 1:1 to a box. Padding
    # restores local context (dimension line, tolerance frame, note) and is
    # clipped within the parent group's bounds to avoid bleeding into neighbors.
    img_h, img_w = cv_result.image1.shape[:2]

    # diff_subboxes is parallel to diff_bboxes; fall back to a single-member
    # group when it is absent (e.g. merging disabled).
    subboxes_by_group = cv_result.diff_subboxes or [
        [b] for b in cv_result.diff_bboxes
    ]

    crops_original = []
    crops_revised = []
    regions_metadata = []

    for gi, ((gx, gy, gw, gh), div_pct, sub_boxes) in enumerate(
        zip(cv_result.diff_bboxes, cv_result.diff_divergences, subboxes_by_group),
        start=1,
    ):
        # Group bounds (clip window for padding). Fall back to the group box
        # itself if no sub-boxes were recorded.
        members = sub_boxes or [(gx, gy, gw, gh)]
        gx0, gy0, gx1, gy1 = gx, gy, gx + gw, gy + gh

        for si, (sbx, sby, sbw, sbh) in enumerate(members, start=1):
            # Pad the sub-box by a fraction of its size (min 12px), clipped to
            # the parent group window so context stays local to the group. The
            # ROI marker (drawn later) disambiguates the target from neighbors
            # captured in the margin, so modest padding is enough.
            pad = max(12, int(0.3 * max(sbw, sbh)))
            px0 = max(gx0, sbx - pad)
            py0 = max(gy0, sby - pad)
            px1 = min(gx1, sbx + sbw + pad)
            py1 = min(gy1, sby + sbh + pad)
            # Guard against degenerate windows.
            if px1 <= px0 or py1 <= py0:
                px0, py0, px1, py1 = sbx, sby, sbx + sbw, sby + sbh
            px0 = max(0, px0); py0 = max(0, py0)
            px1 = min(img_w, px1); py1 = min(img_h, py1)

            sub_id = f"page_{page_index + 1:02d}_grp_{gi:03d}_sub_{si:03d}"
            crop_orig = cv_result.image1[py0:py1, px0:px1]
            crop_rev = cv_result.image2_aligned[py0:py1, px0:px1]
            crops_original.append((sub_id, crop_orig))
            crops_revised.append((sub_id, crop_rev))
            regions_metadata.append({
                "id": sub_id,
                "group_index": gi,
                # The sub-box itself (used for drawing lilac boxes in the report).
                "x": int(sbx),
                "y": int(sby),
                "width": int(sbw),
                "height": int(sbh),
                # The padded crop window (context shown to the LLM).
                "crop_x": int(px0),
                "crop_y": int(py0),
                "crop_width": int(px1 - px0),
                "crop_height": int(py1 - py0),
                # Target sub-box in CROP-LOCAL coords, for the ROI marker.
                "target_local": [int(sbx - px0), int(sby - py0), int(sbw), int(sbh)],
                "divergence_pct": round(div_pct, 2),
                # Parent group box, carried so we can draw the red group box.
                "group_box": [int(gx), int(gy), int(gw), int(gh)],
            })

    # Step 2: LLM verification (single batched call per page, size-capped).
    logger.info(f"Sending {len(crops_revised)} sub-box candidates to LLM for verification...")
    llm_output, metadata = verify_changes_with_llm(
        cv_result.image1, crops_original, crops_revised, regions_metadata, model=model
    )

    # Step 3: Post-process — cluster surviving changes into report groups.
    page_h, page_w = cv_result.image1.shape[:2]
    result = _build_verified_result(
        page_index, regions_metadata, llm_output, metadata,
        page_width=page_w, page_height=page_h,
    )

    # Step 4: Render final image
    result.image_original = cv_result.image1
    result.image_revised_aligned = cv_result.image2_aligned
    result.image_highlighted = render_verified_highlights(
        cv_result.image1, cv_result.image2_aligned, result.true_changes
    )

    return result


def run_verification_pipeline_all_pages(
    pdf1_bytes: bytes,
    pdf2_bytes: bytes,
    model: str = "gemini-2.5-flash",
    opencv_config=None,
) -> list[VerificationResult]:
    """Run the full verification pipeline on all common pages.

    Args:
        pdf1_bytes: Raw bytes of the original PDF.
        pdf2_bytes: Raw bytes of the revised PDF.
        model: Gemini model ID.
        opencv_config: Optional CompareConfig for the OpenCV step.

    Returns:
        List of VerificationResult, one per page.
    """
    import fitz

    doc1 = fitz.open(stream=pdf1_bytes, filetype="pdf")
    doc2 = fitz.open(stream=pdf2_bytes, filetype="pdf")
    n_pages = min(len(doc1), len(doc2))
    doc1.close()
    doc2.close()

    results = []
    for i in range(n_pages):
        result = run_verification_pipeline(pdf1_bytes, pdf2_bytes, i, model, opencv_config)
        results.append(result)

    return results


# ==============================================================================
# Save outputs
# ==============================================================================

def save_verification_result(
    result: VerificationResult,
    output_dir: str | Path,
) -> Path:
    """Save the verification result to disk.

    Produces:
        output_dir/
        ├── report.json        # Structured report
        ├── report.txt         # Human-readable report
        ├── comparison.png     # Side-by-side image with ID-labeled highlights
        ├── original_full.png  # Full-page original image at detection resolution
        └── revised_full.png   # Full-page revised aligned image at detection resolution

    Args:
        result: VerificationResult from the pipeline.
        output_dir: Directory to write into.

    Returns:
        Path to the output directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Report JSON
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(result.report_json(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Report text
    txt_path = output_dir / "report.txt"
    txt_path.write_text(result.report_text(), encoding="utf-8")

    # Comparison image
    if result.image_highlighted is not None:
        img_path = output_dir / "comparison.png"
        cv2.imwrite(str(img_path), result.image_highlighted)

    # Save full-page images for crop extraction in reports
    if result.image_original is not None:
        orig_path = output_dir / "original_full.png"
        cv2.imwrite(str(orig_path), result.image_original)
    
    # Note: image_original is currently always None after render_verified_highlights.
    # We need to preserve it in run_verification_pipeline.

    return output_dir
