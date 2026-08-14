"""Cloud Run Function 1: Orchestrator.

Coordinates the CAD review workflow:
1. Check if manifest.txt exists at the base GCS path (idempotency guard).
2. If manifest exists → stop (already processed).
3. If not → create manifest.txt with start timestamp.
4. Call Function 2 (pipeline) to execute the CAD review.
5. Call Function 3 (mailer) to send the report via email.
6. Write execution summary to manifest.txt.

Expected JSON payload:
{
    "process_id": "abc-123",
    "base_gcs_path": "gs://bucket/process/abc-123",
    "original_pdf_gcs_path": "gs://bucket/process/abc-123/original.pdf",
    "revised_pdf_gcs_path": "gs://bucket/process/abc-123/revised.pdf"
}

Environment variables:
    PIPELINE_FUNCTION_URL   - URL of the pipeline Cloud Run function
    MAILER_FUNCTION_URL     - URL of the mailer Cloud Run function
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

import functions_framework
import google.auth.transport.requests
import google.oauth2.id_token
import requests
from flask import Request, jsonify
from google.cloud import storage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "manifest.txt"


def _parse_gcs_path(gcs_path: str) -> tuple[str, str]:
    """Parse a gs://bucket/path string into (bucket_name, blob_path)."""
    if not gcs_path.startswith("gs://"):
        raise ValueError(f"Invalid GCS path (must start with gs://): {gcs_path}")
    parts = gcs_path[5:].split("/", 1)
    if len(parts) < 2:
        raise ValueError(f"Invalid GCS path (missing object path): {gcs_path}")
    return parts[0], parts[1]


def _manifest_exists(client: storage.Client, base_gcs_path: str) -> bool:
    """Check if manifest.txt already exists at the base GCS path."""
    bucket_name, base_blob = _parse_gcs_path(base_gcs_path)
    manifest_blob_path = f"{base_blob}/{MANIFEST_FILENAME}"
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(manifest_blob_path)
    return blob.exists()


def _write_manifest(client: storage.Client, base_gcs_path: str, content: str) -> None:
    """Write (or overwrite) manifest.txt at the base GCS path."""
    bucket_name, base_blob = _parse_gcs_path(base_gcs_path)
    manifest_blob_path = f"{base_blob}/{MANIFEST_FILENAME}"
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(manifest_blob_path)
    blob.upload_from_string(content, content_type="text/plain")
    logger.info("Manifest written: gs://%s/%s", bucket_name, manifest_blob_path)


def _get_id_token(audience: str) -> str:
    """Get an ID token for authenticating to another Cloud Run service."""
    auth_req = google.auth.transport.requests.Request()
    token = google.oauth2.id_token.fetch_id_token(auth_req, audience)
    return token


def _call_function(url: str, payload: dict, timeout: int = 600) -> dict:
    """Call a Cloud Run function with an authenticated request.

    Uses the service's URL as the audience for the ID token.
    """
    token = _get_id_token(url)
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    response = requests.post(url, json=payload, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.json()


@functions_framework.http
def orchestrator(request: Request):
    """HTTP Cloud Run function entry point for the orchestrator."""
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

    logger.info("Orchestrator started | process_id=%s", process_id)

    # Read function URLs from env
    pipeline_url = os.environ.get("PIPELINE_FUNCTION_URL", "")
    mailer_url = os.environ.get("MAILER_FUNCTION_URL", "")

    if not pipeline_url or not mailer_url:
        return jsonify({
            "error": "Missing environment variables: PIPELINE_FUNCTION_URL and/or MAILER_FUNCTION_URL",
        }), 500

    # Initialize GCS client
    gcs_client = storage.Client()

    # Step 1: Check manifest (idempotency)
    if _manifest_exists(gcs_client, base_gcs_path):
        logger.info("Manifest already exists, stopping | process_id=%s", process_id)
        return jsonify({
            "status": "skipped",
            "process_id": process_id,
            "reason": "manifest.txt already exists — process was already executed",
        }), 200

    # Step 2: Create manifest with start timestamp
    start_time = datetime.now(timezone.utc)
    initial_manifest = (
        f"process_id: {process_id}\n"
        f"status: in_progress\n"
        f"started_at: {start_time.isoformat()}\n"
    )
    _write_manifest(gcs_client, base_gcs_path, initial_manifest)

    # Step 3: Call pipeline function
    pipeline_payload = {
        "process_id": process_id,
        "base_gcs_path": base_gcs_path,
        "original_pdf_gcs_path": original_gcs,
        "revised_pdf_gcs_path": revised_gcs,
    }

    try:
        logger.info("Calling pipeline function | process_id=%s", process_id)
        pipeline_response = _call_function(pipeline_url, pipeline_payload, timeout=900)
    except Exception as e:
        logger.exception("Pipeline call failed | process_id=%s", process_id)
        # Update manifest with failure
        end_time = datetime.now(timezone.utc)
        fail_manifest = (
            f"process_id: {process_id}\n"
            f"status: failed\n"
            f"started_at: {start_time.isoformat()}\n"
            f"ended_at: {end_time.isoformat()}\n"
            f"duration_seconds: {(end_time - start_time).total_seconds():.1f}\n"
            f"stage: pipeline\n"
            f"error: {str(e)}\n"
        )
        _write_manifest(gcs_client, base_gcs_path, fail_manifest)
        return jsonify({
            "status": "error",
            "process_id": process_id,
            "stage": "pipeline",
            "error": str(e),
        }), 500

    report_gcs_path = pipeline_response.get("report_gcs_path", "")

    # Step 4: Call mailer function
    mailer_payload = {
        "process_id": process_id,
        "report_gcs_path": report_gcs_path,
    }

    try:
        logger.info("Calling mailer function | process_id=%s", process_id)
        mailer_response = _call_function(mailer_url, mailer_payload, timeout=120)
    except Exception as e:
        logger.exception("Mailer call failed | process_id=%s", process_id)
        # Update manifest — pipeline succeeded but mailer failed
        end_time = datetime.now(timezone.utc)
        fail_manifest = (
            f"process_id: {process_id}\n"
            f"status: partial_failure\n"
            f"started_at: {start_time.isoformat()}\n"
            f"ended_at: {end_time.isoformat()}\n"
            f"duration_seconds: {(end_time - start_time).total_seconds():.1f}\n"
            f"pipeline: success\n"
            f"mailer: failed\n"
            f"report_gcs_path: {report_gcs_path}\n"
            f"error: {str(e)}\n"
        )
        _write_manifest(gcs_client, base_gcs_path, fail_manifest)
        return jsonify({
            "status": "error",
            "process_id": process_id,
            "stage": "mailer",
            "pipeline_status": "success",
            "report_gcs_path": report_gcs_path,
            "error": str(e),
        }), 500

    # Step 5: Write final execution summary to manifest
    end_time = datetime.now(timezone.utc)
    duration = (end_time - start_time).total_seconds()
    final_manifest = (
        f"process_id: {process_id}\n"
        f"status: completed\n"
        f"started_at: {start_time.isoformat()}\n"
        f"ended_at: {end_time.isoformat()}\n"
        f"duration_seconds: {duration:.1f}\n"
        f"pipeline: success\n"
        f"mailer: success\n"
        f"report_gcs_path: {report_gcs_path}\n"
        f"files_uploaded: {pipeline_response.get('files_uploaded', 'N/A')}\n"
        f"recipients: {mailer_response.get('recipients', [])}\n"
    )
    _write_manifest(gcs_client, base_gcs_path, final_manifest)

    logger.info(
        "Orchestrator completed | process_id=%s | duration=%.1fs",
        process_id,
        duration,
    )

    return jsonify({
        "status": "success",
        "process_id": process_id,
        "duration_seconds": round(duration, 1),
        "report_gcs_path": report_gcs_path,
    }), 200
