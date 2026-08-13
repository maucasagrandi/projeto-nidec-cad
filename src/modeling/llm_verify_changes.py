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
from dataclasses import dataclass
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
# Pydantic Models — LLM Response Schema
# ==============================================================================

class RegionVerdict(BaseModel):
    """LLM verdict for a single candidate region."""

    id: str = Field(description="The region ID (e.g., 'page_01_diff_003')")
    is_true_change: bool = Field(description="True if the region contains a real, meaningful difference")
    description: str = Field(
        description=(
            "If is_true_change=true: concise description of what changed "
            "(e.g., 'Dimension tolerance changed from ±0.1 to ±0.2'). "
            "If is_true_change=false: brief reason why it's a false positive "
            "(e.g., 'Rendering artifact on dash-dot line')"
        )
    )


class VerificationOutput(BaseModel):
    """Structured output from the LLM verification call."""

    verdicts: list[RegionVerdict] = Field(
        description="One verdict per candidate region, in the same order as provided"
    )


# ==============================================================================
# Result dataclass
# ==============================================================================

@dataclass
class VerifiedChange:
    """A single verified true change, with contiguous indexing."""

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


@dataclass
class VerificationResult:
    """Full result of the verification pipeline for one page."""

    page_index: int
    true_changes: list[VerifiedChange]
    false_positive_ids: list[str]
    metadata: AnalysisMetadata | None = None

    # Images
    image_original: np.ndarray | None = None
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

1. A full-page image of the ORIGINAL CAD drawing (for overall context)
2. For each detected region: a crop from the ORIGINAL and a crop from the REVISED drawing at the exact same location

For each region pair, compare the ORIGINAL crop directly against the REVISED crop and determine:
- Is this a TRUE CHANGE (text, dimensions, geometry, symbols, notes, table content actually differ between the two crops)?
- Or is it a FALSE POSITIVE (the crops are visually identical or differ only by rendering artifacts like anti-aliasing, slight line weight variation, or sub-pixel shifts)?

TRUE CHANGES include: modified dimensions/tolerances, added/removed text or notes, changed GD&T symbols, altered geometry, new or deleted features, table content changes, revision block updates, any content that is present in one crop but absent in the other.

FALSE POSITIVES include: crops that look identical between original and revised, slight rendering differences in line thickness, minor positional shifts of identical content, anti-aliasing artifacts.

IMPORTANT: If a crop from the revised drawing shows content that is NOT present in the corresponding original crop (or vice versa), that IS a true change — content was added or removed.

For each true change, provide a concise technical description of what actually changed.

The regions to analyze are:
{regions_json}

Respond with a verdict for each region in the same order.
"""


# ==============================================================================
# LLM Call
# ==============================================================================

def _encode_image_to_base64(img: np.ndarray) -> bytes:
    """Encode a BGR numpy array to PNG bytes."""
    _, buffer = cv2.imencode(".png", img)
    return buffer.tobytes()


def verify_changes_with_llm(
    original_image: np.ndarray,
    crops_original: list[tuple[str, np.ndarray]],
    crops_revised: list[tuple[str, np.ndarray]],
    regions_metadata: list[dict],
    model: str = "gemini-2.5-flash",
) -> tuple[VerificationOutput, AnalysisMetadata]:
    """Send original and revised crops to Gemini for verification.

    Args:
        original_image: Full-page BGR image of the original drawing (for overall context).
        crops_original: List of (region_id, crop_from_original) pairs.
        crops_revised: List of (region_id, crop_from_revised) pairs.
        regions_metadata: List of region dicts from the manifest (id, x, y, width, height, divergence_pct).
        model: Gemini model to use.

    Returns:
        (VerificationOutput, AnalysisMetadata)
    """
    import os

    from dotenv import load_dotenv
    from google import genai
    from google.genai import types

    load_dotenv()

    # Initialize client (same pattern as llm_models.py)
    project = os.getenv("GCP_PROJECT_ID", "acim-global-data-lake-sandbox")
    location = os.getenv("GCP_REGION", "us-east5")

    client = genai.Client(
        vertexai=True,
        project=project,
        location=location,
    )

    start_time = time.time()

    # Build the regions description for the prompt
    regions_desc = []
    for meta in regions_metadata:
        regions_desc.append({
            "id": meta["id"],
            "location": f"({meta['x']}, {meta['y']})",
            "size": f"{meta['width']}x{meta['height']}",
            "divergence_pct": meta["divergence_pct"],
        })
    regions_json = json.dumps(regions_desc, indent=2)

    # Build content parts
    content_parts = []

    # 1. The full original image for overall context
    original_bytes = _encode_image_to_base64(original_image)
    content_parts.append(
        types.Part.from_bytes(data=original_bytes, mime_type="image/png")
    )
    content_parts.append(
        types.Part.from_text(text="Above: the full ORIGINAL drawing (for overall context).\n\nBelow: pairs of cropped regions showing the SAME AREA from both ORIGINAL and REVISED drawings. Compare each pair to determine if there is a real difference:\n")
    )

    # 2. Each region: original crop + revised crop side by side
    for (region_id, crop_orig), (_, crop_rev) in zip(crops_original, crops_revised):
        content_parts.append(
            types.Part.from_text(text=f"\n--- Region {region_id} — ORIGINAL crop: ---")
        )
        orig_bytes = _encode_image_to_base64(crop_orig)
        content_parts.append(
            types.Part.from_bytes(data=orig_bytes, mime_type="image/png")
        )
        content_parts.append(
            types.Part.from_text(text=f"--- Region {region_id} — REVISED crop: ---")
        )
        rev_bytes = _encode_image_to_base64(crop_rev)
        content_parts.append(
            types.Part.from_bytes(data=rev_bytes, mime_type="image/png")
        )

    # 3. The prompt
    prompt_text = VERIFICATION_PROMPT.format(regions_json=regions_json)
    content_parts.append(types.Part.from_text(text=prompt_text))

    # Make the call with structured output
    logger.info(f"Sending {len(crops_revised)} regions to {model} for verification...")

    response = client.models.generate_content(
        model=model,
        contents=content_parts,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=VerificationOutput,
        ),
    )

    parsed = VerificationOutput.model_validate_json(response.text)

    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000

    usage = response.usage_metadata
    metadata = AnalysisMetadata(
        total_tokens=usage.total_token_count,
        prompt_tokens=usage.prompt_token_count,
        completion_tokens=usage.candidates_token_count,
        latency_ms=latency_ms,
        model=model,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )

    logger.info(
        f"Verification complete: "
        f"{sum(1 for v in parsed.verdicts if v.is_true_change)} true changes, "
        f"{sum(1 for v in parsed.verdicts if not v.is_true_change)} false positives"
    )

    return parsed, metadata


# ==============================================================================
# Post-processing: re-index and build result
# ==============================================================================

def _build_verified_result(
    page_index: int,
    regions_metadata: list[dict],
    llm_output: VerificationOutput,
    metadata: AnalysisMetadata,
) -> VerificationResult:
    """Process LLM verdicts into a VerificationResult with contiguous indexing.

    Args:
        page_index: Page number.
        regions_metadata: Region dicts from the OpenCV manifest.
        llm_output: Parsed LLM structured output.
        metadata: LLM call metadata.

    Returns:
        VerificationResult with contiguous 1..N indexing on true changes.
    """
    # Build a lookup from ID -> metadata
    meta_by_id = {r["id"]: r for r in regions_metadata}

    true_changes = []
    false_positive_ids = []
    contiguous_idx = 1

    for verdict in llm_output.verdicts:
        if verdict.is_true_change:
            meta = meta_by_id.get(verdict.id, {})
            true_changes.append(VerifiedChange(
                index=contiguous_idx,
                original_id=verdict.id,
                x=meta.get("x", 0),
                y=meta.get("y", 0),
                width=meta.get("width", 0),
                height=meta.get("height", 0),
                divergence_pct=meta.get("divergence_pct", 0.0),
                description=verdict.description,
            ))
            contiguous_idx += 1
        else:
            false_positive_ids.append(verdict.id)

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

    # Reduce the panels before creating overlays or concatenating. A pair of
    # 300-DPI A3 pages can otherwise require a contiguous allocation >200 MB.
    original = resize_panel(image_original)
    revised = resize_panel(image_revised_aligned)
    overlay = revised.copy()

    for change in true_changes:
        x, y, w, h = (
            round(change.x * output_scale),
            round(change.y * output_scale),
            max(1, round(change.width * output_scale)),
            max(1, round(change.height * output_scale)),
        )
        cv2.rectangle(overlay, (x, y), (x + w, y + h), highlight_color, -1)

    cv2.addWeighted(overlay, highlight_alpha, revised, 1 - highlight_alpha, 0, revised)

    # Draw borders and ID labels
    img_h, img_w = revised.shape[:2]
    font_scale = max(0.5, min(img_h, img_w) / 2500.0)
    font_thickness = max(1, int(font_scale * 2.5))

    for change in true_changes:
        x, y, w, h = (
            round(change.x * output_scale),
            round(change.y * output_scale),
            max(1, round(change.width * output_scale)),
            max(1, round(change.height * output_scale)),
        )
        cv2.rectangle(revised, (x, y), (x + w, y + h), highlight_color, 2)

        # ID label
        label = str(change.index)
        label_size, baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )
        lw, lh = label_size

        # Position: top-left, slightly above the box
        label_x = x + 3
        label_y = y - 6 if y - 6 > lh else y + lh + 6

        # White background for readability
        cv2.rectangle(
            revised,
            (label_x - 2, label_y - lh - 3),
            (label_x + lw + 3, label_y + baseline + 3),
            (255, 255, 255),
            -1,
        )
        cv2.putText(
            revised, label, (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
            highlight_color, font_thickness, cv2.LINE_AA,
        )

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

    # Build crops from BOTH images and metadata for LLM
    crops_original = []
    crops_revised = []
    regions_metadata = []
    for idx, ((x, y, w, h), div_pct) in enumerate(
        zip(cv_result.diff_bboxes, cv_result.diff_divergences), start=1
    ):
        region_id = f"page_{page_index + 1:02d}_diff_{idx:03d}"
        crop_rev = cv_result.image2_aligned[y:y+h, x:x+w]
        crop_orig = cv_result.image1[y:y+h, x:x+w]
        crops_original.append((region_id, crop_orig))
        crops_revised.append((region_id, crop_rev))
        regions_metadata.append({
            "id": region_id,
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "divergence_pct": round(div_pct, 2),
        })

    # Step 2: LLM verification
    logger.info(f"Sending {len(crops_revised)} candidates to LLM for verification...")
    llm_output, metadata = verify_changes_with_llm(
        cv_result.image1, crops_original, crops_revised, regions_metadata, model=model
    )

    # Step 3: Post-process into contiguously-indexed result
    result = _build_verified_result(page_index, regions_metadata, llm_output, metadata)

    # Step 4: Render final image
    result.image_original = None
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
        └── comparison.png     # Side-by-side image with ID-labeled highlights

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

    return output_dir
