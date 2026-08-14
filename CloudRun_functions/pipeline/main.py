"""Cloud Run Function 2: Pipeline.

Executes the CAD review pipeline (run_integrated_review + save_integrated_review)
using PDF files from GCS and writing outputs to a GCS output path.

Expected JSON payload:
{
    "process_id": "abc-123",
    "base_gcs_path": "gs://bucket/process/abc-123",
    "original_pdf_gcs_path": "gs://bucket/process/abc-123/original.pdf",
    "revised_pdf_gcs_path": "gs://bucket/process/abc-123/revised.pdf"
}

Environment variables:
    GCP_PROJECT_ID  - Google Cloud project ID
    GCP_REGION      - Vertex AI region (e.g. us-east5)
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

import functions_framework
from flask import Request, jsonify
from google.cloud import storage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GCS output subdirectory for pipeline results
PROCESSING_OUTPUTS_DIR = "PROCESSING_OUTPUTS"


def _parse_gcs_path(gcs_path: str) -> tuple[str, str]:
    """Parse a gs://bucket/path string into (bucket_name, blob_path)."""
    if not gcs_path.startswith("gs://"):
        raise ValueError(f"Invalid GCS path (must start with gs://): {gcs_path}")
    parts = gcs_path[5:].split("/", 1)
    if len(parts) < 2:
        raise ValueError(f"Invalid GCS path (missing object path): {gcs_path}")
    return parts[0], parts[1]


def _download_blob(client: storage.Client, gcs_path: str) -> bytes:
    """Download a GCS object and return its bytes."""
    bucket_name, blob_path = _parse_gcs_path(gcs_path)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    return blob.download_as_bytes()


def _upload_directory(client: storage.Client, local_dir: Path, gcs_base_path: str) -> list[str]:
    """Upload all files in a local directory tree to GCS, preserving structure.

    Returns list of uploaded GCS paths.
    """
    bucket_name, base_blob = _parse_gcs_path(gcs_base_path)
    bucket = client.bucket(bucket_name)
    uploaded = []

    for local_file in sorted(local_dir.rglob("*")):
        if not local_file.is_file():
            continue
        relative = local_file.relative_to(local_dir)
        blob_path = f"{base_blob}/{relative}"
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(str(local_file))
        uploaded.append(f"gs://{bucket_name}/{blob_path}")
        logger.info("Uploaded %s", f"gs://{bucket_name}/{blob_path}")

    return uploaded


def run_pipeline(
    original_pdf: bytes,
    revised_pdf: bytes,
    original_name: str,
    revised_name: str,
    output_dir: Path,
) -> dict[str, Path]:
    """Execute the integrated CAD review pipeline and save results locally.

    Returns the paths dict from save_integrated_review.
    """
    from src.cad_review.integrated_review import run_integrated_review, save_integrated_review
    from src.utils.opencv_cad_compare import CompareConfig

    result = run_integrated_review(
        original_pdf,
        revised_pdf,
        original_name=original_name,
        revised_name=revised_name,
        comparison_model="gemini-2.5-flash",
        gdt_dpi=150,
        gdt_threshold=0.74,
        gdt_workers=1,
        opencv_config=CompareConfig(dpi=150, diff_threshold=40, merge_distance=50),
    )

    paths = save_integrated_review(result, output_dir)
    return paths


@functions_framework.http
def pipeline(request: Request):
    """HTTP Cloud Run function entry point for the pipeline."""
    # Parse request
    try:
        payload = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON payload"}), 400

    required_fields = ["process_id", "base_gcs_path", "original_pdf_gcs_path", "revised_pdf_gcs_path"]
    missing = [f for f in required_fields if f not in payload]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    process_id = payload["process_id"]
    base_gcs_path = payload["base_gcs_path"].rstrip("/")
    original_gcs = payload["original_pdf_gcs_path"]
    revised_gcs = payload["revised_pdf_gcs_path"]

    logger.info("Pipeline started | process_id=%s", process_id)

    try:
        # Initialize GCS client
        gcs_client = storage.Client()

        # Download PDFs from GCS
        logger.info("Downloading original PDF: %s", original_gcs)
        original_pdf = _download_blob(gcs_client, original_gcs)

        logger.info("Downloading revised PDF: %s", revised_gcs)
        revised_pdf = _download_blob(gcs_client, revised_gcs)

        # Extract filenames from GCS paths
        original_name = original_gcs.rsplit("/", 1)[-1]
        revised_name = revised_gcs.rsplit("/", 1)[-1]

        # Run pipeline in a temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()

            paths = run_pipeline(
                original_pdf,
                revised_pdf,
                original_name=original_name,
                revised_name=revised_name,
                output_dir=output_dir,
            )

            # Upload results to GCS under base_gcs_path/PROCESSING_OUTPUTS
            output_gcs_path = f"{base_gcs_path}/{PROCESSING_OUTPUTS_DIR}"
            uploaded_files = _upload_directory(gcs_client, output_dir, output_gcs_path)

            # The report PDF path in GCS
            report_gcs_path = f"{output_gcs_path}/integrated_review_report.pdf"

        logger.info("Pipeline completed | process_id=%s | files_uploaded=%d", process_id, len(uploaded_files))

        return jsonify({
            "status": "success",
            "process_id": process_id,
            "report_gcs_path": report_gcs_path,
            "output_gcs_path": output_gcs_path,
            "files_uploaded": len(uploaded_files),
        }), 200

    except Exception as e:
        logger.exception("Pipeline failed | process_id=%s", process_id)
        return jsonify({
            "status": "error",
            "process_id": process_id,
            "error": str(e),
        }), 500
