"""Cloud Run Service: Orchestrator.

Coordinates the CAD review workflow:
1. Receives only the base GCS path to a CT folder (e.g. gs://bucket/.../CT_xxxxx).
2. Resolves the process_id (CT folder name) and the two PDF file paths
   from the ResultingObjects/ subfolder.
3. Checks manifest.txt for previous execution state:
   - No manifest       → run full pipeline + mailer
   - manifest=failed   → retry full pipeline + mailer
   - manifest=partial_failure (pipeline ok, mailer failed) → retry mailer only
   - manifest=completed → skip (already done)
4. Writes manifest.txt with status after each step.

Invocation modes:

1. Eventarc GCS trigger (production): a CloudEvent of type
   ``google.cloud.storage.object.v1.finalized`` is delivered when a PDF is
   uploaded. The orchestrator reacts only to PDFs landing under
   ``ResultingObjects/`` inside the ``TRIGGER_PREFIX`` scope. Because a CT folder
   may receive one or two PDFs, two uploads produce two events. To handle this:
     - A short settle delay (``SETTLE_DELAY_SECONDS``) is applied before
       resolving PDFs, so a run does not start in single-PDF mode while a second
       PDF is still in flight.
     - The manifest is claimed atomically (create-if-absent). Only the first
       event to win the claim proceeds; the other stands down. This is the real
       dedupe — the settle delay only improves single-vs-comparison accuracy.

2. Direct JSON call (manual / testing):
   {
       "base_gcs_path": "gs://bucket/.../CT_13358002",
       "recipients": ["a@x.com", "b@y.com"]   # optional, overrides MAIL_RECIPIENTS
   }

Environment variables:
    PIPELINE_FUNCTION_URL      - URL of the pipeline Cloud Run service
    MAILER_FUNCTION_URL        - URL of the mailer Cloud Run service
    TRIGGER_PREFIX             - only objects whose name starts with this prefix
                                 trigger processing (default: temp/Windchill/cadreview)
    SETTLE_DELAY_SECONDS       - seconds to wait after a PDF event before
                                 resolving PDFs, to let a second PDF settle (default: 5)
    STALE_IN_PROGRESS_MINUTES  - age after which an in_progress manifest is
                                 considered stale/crashed and retried (default: 30)
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone

import google.auth.transport.requests
import google.oauth2.id_token
import requests
from cloudevents.http import from_http
from flask import Flask, Request, jsonify, request as flask_request
from google.api_core.exceptions import PreconditionFailed
from google.cloud import storage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "manifest.txt"

# Only objects under this prefix trigger processing. The exporter is not under
# our control, so we scope strictly to the expected upload area.
TRIGGER_PREFIX = os.environ.get("TRIGGER_PREFIX", "temp/Windchill/cadreview")

# Subfolder (relative to a CT folder) where the PDFs land.
RESULTING_OBJECTS_MARKER = "/ResultingObjects/"

# Wait this long after a PDF event before resolving the PDF set, so a second PDF
# arriving moments later is included (comparison vs single-PDF accuracy).
SETTLE_DELAY_SECONDS = int(os.environ.get("SETTLE_DELAY_SECONDS", "5"))

# An in_progress manifest older than this is treated as a crashed run and retried.
STALE_IN_PROGRESS_MINUTES = int(os.environ.get("STALE_IN_PROGRESS_MINUTES", "30"))


def _parse_gcs_path(gcs_path: str) -> tuple[str, str]:
    """Parse a gs://bucket/path string into (bucket_name, blob_path)."""
    if not gcs_path.startswith("gs://"):
        raise ValueError(f"Invalid GCS path (must start with gs://): {gcs_path}")
    parts = gcs_path[5:].split("/", 1)
    if len(parts) < 2:
        raise ValueError(f"Invalid GCS path (missing object path): {gcs_path}")
    return parts[0], parts[1]


def _read_manifest(client: storage.Client, base_gcs_path: str) -> dict | None:
    """Read and parse manifest.txt. Returns dict of key-value pairs or None."""
    bucket_name, base_blob = _parse_gcs_path(base_gcs_path)
    manifest_blob_path = f"{base_blob}/{MANIFEST_FILENAME}"
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(manifest_blob_path)

    if not blob.exists():
        return None

    content = blob.download_as_text()
    manifest = {}
    for line in content.strip().split("\n"):
        if ": " in line:
            key, value = line.split(": ", 1)
            manifest[key.strip()] = value.strip()
    return manifest


def _write_manifest(client: storage.Client, base_gcs_path: str, content: str) -> None:
    """Write (or overwrite) manifest.txt at the base GCS path."""
    bucket_name, base_blob = _parse_gcs_path(base_gcs_path)
    manifest_blob_path = f"{base_blob}/{MANIFEST_FILENAME}"
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(manifest_blob_path)
    blob.upload_from_string(content, content_type="text/plain")
    logger.info("Manifest written: gs://%s/%s", bucket_name, manifest_blob_path)


def _claim_manifest(client: storage.Client, base_gcs_path: str, content: str) -> bool:
    """Atomically create manifest.txt only if it does not already exist.

    Uses the GCS precondition ``if_generation_match=0`` (create-if-absent).
    Returns True if this invocation won the claim, False if another invocation
    already created the manifest (i.e. we lost the race and should stand down).
    """
    bucket_name, base_blob = _parse_gcs_path(base_gcs_path)
    manifest_blob_path = f"{base_blob}/{MANIFEST_FILENAME}"
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(manifest_blob_path)
    try:
        blob.upload_from_string(
            content, content_type="text/plain", if_generation_match=0
        )
        logger.info("Manifest claimed: gs://%s/%s", bucket_name, manifest_blob_path)
        return True
    except PreconditionFailed:
        logger.info(
            "Manifest already exists, claim lost: gs://%s/%s",
            bucket_name,
            manifest_blob_path,
        )
        return False


def _resolve_pdf_paths(client: storage.Client, base_gcs_path: str) -> tuple[str | None, str]:
    """List PDFs in ResultingObjects/ and return (original, revised) paths.

    If 2 PDFs: lower revision is original, higher is revised (comparison mode).
    If 1 PDF: it is the revised (single analysis mode), original is None.
    If 0 PDFs: raises ValueError.
    """
    bucket_name, base_blob = _parse_gcs_path(base_gcs_path)
    prefix = f"{base_blob}/ResultingObjects/"
    bucket = client.bucket(bucket_name)

    blobs = list(bucket.list_blobs(prefix=prefix))
    pdf_blobs = [b for b in blobs if b.name.lower().endswith(".pdf")]

    if len(pdf_blobs) == 0:
        raise ValueError(
            f"No PDFs found in {base_gcs_path}/ResultingObjects/"
        )

    # Sort by name — lower revision first (original), higher second (revised)
    pdf_blobs.sort(key=lambda b: b.name)

    if len(pdf_blobs) == 1:
        revised_path = f"gs://{bucket_name}/{pdf_blobs[0].name}"
        logger.info("Single PDF mode — revised PDF: %s", revised_path)
        return None, revised_path

    # 2+ PDFs: first is original, last is revised
    original_path = f"gs://{bucket_name}/{pdf_blobs[0].name}"
    revised_path = f"gs://{bucket_name}/{pdf_blobs[-1].name}"

    logger.info("Comparison mode — original PDF: %s", original_path)
    logger.info("Comparison mode — revised PDF: %s", revised_path)

    return original_path, revised_path


def _get_id_token(audience: str) -> str:
    """Get an ID token for authenticating to another Cloud Run service."""
    auth_req = google.auth.transport.requests.Request()
    token = google.oauth2.id_token.fetch_id_token(auth_req, audience)
    return token


def _call_function(url: str, payload: dict, timeout: int = 600) -> dict:
    """Call a Cloud Run service with an authenticated request."""
    token = _get_id_token(url)
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    response = requests.post(url, json=payload, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _is_stale_in_progress(manifest: dict) -> bool:
    """Return True if an in_progress manifest is old enough to be considered crashed."""
    started_raw = manifest.get("started_at", "")
    if not started_raw:
        # No timestamp — cannot judge age; treat as stale so we don't deadlock.
        return True
    try:
        started = datetime.fromisoformat(started_raw)
    except ValueError:
        return True
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    age = datetime.now(timezone.utc) - started
    return age >= timedelta(minutes=STALE_IN_PROGRESS_MINUTES)


def _run(base_gcs_path: str, recipients: list[str] | str | None = None):
    """Execute the CAD review workflow for a single CT folder.

    If ``recipients`` is provided, it overrides the mailer's default recipient
    list (MAIL_RECIPIENTS env var) for this run only.

    Returns a (flask_response, status_code) tuple.
    """
    base_gcs_path = base_gcs_path.rstrip("/")

    # Derive process_id from the CT folder name
    process_id = base_gcs_path.rsplit("/", 1)[-1]

    logger.info("Orchestrator started | process_id=%s | path=%s", process_id, base_gcs_path)

    # Read function URLs from env
    pipeline_url = os.environ.get("PIPELINE_FUNCTION_URL", "")
    mailer_url = os.environ.get("MAILER_FUNCTION_URL", "")

    if not pipeline_url or not mailer_url:
        return jsonify({
            "error": "Missing environment variables: PIPELINE_FUNCTION_URL and/or MAILER_FUNCTION_URL",
        }), 500

    # Initialize GCS client
    gcs_client = storage.Client()

    # Check manifest for previous execution state
    manifest = _read_manifest(gcs_client, base_gcs_path)

    run_pipeline = True
    run_mailer = True
    report_gcs_path = ""
    start_time = datetime.now(timezone.utc)

    if manifest is not None:
        status = manifest.get("status", "")

        if status == "completed":
            logger.info("Manifest status=completed, skipping | process_id=%s", process_id)
            return jsonify({
                "status": "skipped",
                "process_id": process_id,
                "reason": "manifest.txt status=completed — process already executed successfully",
            }), 200

        elif status == "partial_failure":
            # Pipeline succeeded but mailer failed — retry mailer only
            logger.info("Manifest status=partial_failure, retrying mailer only | process_id=%s", process_id)
            run_pipeline = False
            run_mailer = True
            report_gcs_path = manifest.get("report_gcs_path", "")
            if not report_gcs_path:
                # Fallback: reconstruct expected report path
                report_gcs_path = f"{base_gcs_path}/PROCESSING_OUTPUTS/integrated_review_report.pdf"

        elif status == "failed":
            # Pipeline failed — retry everything
            logger.info("Manifest status=failed, retrying full pipeline | process_id=%s", process_id)
            run_pipeline = True
            run_mailer = True

        elif status == "in_progress":
            # A concurrent invocation is (or was) working on this folder.
            # Only retry if the run looks crashed (stale); otherwise stand down
            # so we never run the pipeline/mailer twice in parallel.
            if _is_stale_in_progress(manifest):
                logger.info(
                    "Manifest status=in_progress but stale (>%dm), retrying | process_id=%s",
                    STALE_IN_PROGRESS_MINUTES, process_id,
                )
                run_pipeline = True
                run_mailer = True
            else:
                logger.info(
                    "Manifest status=in_progress and fresh, skipping (already being processed) | process_id=%s",
                    process_id,
                )
                return jsonify({
                    "status": "skipped",
                    "process_id": process_id,
                    "reason": "manifest.txt status=in_progress — another invocation is already processing this folder",
                }), 200

    initial_manifest = (
        f"process_id: {process_id}\n"
        f"status: in_progress\n"
        f"started_at: {start_time.isoformat()}\n"
        f"run_pipeline: {run_pipeline}\n"
        f"run_mailer: {run_mailer}\n"
    )

    if manifest is None:
        # First invocation for this folder: claim atomically so that two
        # near-simultaneous events cannot both proceed. The loser stands down.
        if not _claim_manifest(gcs_client, base_gcs_path, initial_manifest):
            logger.info(
                "Lost manifest claim race, skipping | process_id=%s", process_id
            )
            return jsonify({
                "status": "skipped",
                "process_id": process_id,
                "reason": "concurrent invocation already claimed this folder",
            }), 200
    else:
        # Manifest already existed (retry path): plain overwrite is fine.
        _write_manifest(gcs_client, base_gcs_path, initial_manifest)

    # --- Pipeline step ---
    if run_pipeline:
        try:
            original_gcs, revised_gcs = _resolve_pdf_paths(gcs_client, base_gcs_path)
        except ValueError as e:
            logger.error("Failed to resolve PDF paths: %s", e)
            end_time = datetime.now(timezone.utc)
            fail_manifest = (
                f"process_id: {process_id}\n"
                f"status: failed\n"
                f"started_at: {start_time.isoformat()}\n"
                f"ended_at: {end_time.isoformat()}\n"
                f"duration_seconds: {(end_time - start_time).total_seconds():.1f}\n"
                f"stage: resolve_pdfs\n"
                f"error: {str(e)}\n"
            )
            _write_manifest(gcs_client, base_gcs_path, fail_manifest)
            return jsonify({
                "status": "error",
                "process_id": process_id,
                "stage": "resolve_pdfs",
                "error": str(e),
            }), 500

        pipeline_payload = {
            "process_id": process_id,
            "base_gcs_path": base_gcs_path,
            "revised_pdf_gcs_path": revised_gcs,
            "mode": "comparison" if original_gcs else "single",
        }
        if original_gcs:
            pipeline_payload["original_pdf_gcs_path"] = original_gcs

        try:
            logger.info("Calling pipeline | process_id=%s", process_id)
            pipeline_response = _call_function(pipeline_url, pipeline_payload, timeout=900)
            report_gcs_path = pipeline_response.get("report_gcs_path", "")
            logger.info("Pipeline succeeded | report=%s", report_gcs_path)
        except Exception as e:
            logger.exception("Pipeline call failed | process_id=%s", process_id)
            end_time = datetime.now(timezone.utc)
            fail_manifest = (
                f"process_id: {process_id}\n"
                f"status: failed\n"
                f"started_at: {start_time.isoformat()}\n"
                f"ended_at: {end_time.isoformat()}\n"
                f"duration_seconds: {(end_time - start_time).total_seconds():.1f}\n"
                f"stage: pipeline\n"
                f"pipeline: failed\n"
                f"error: {str(e)}\n"
            )
            _write_manifest(gcs_client, base_gcs_path, fail_manifest)
            return jsonify({
                "status": "error",
                "process_id": process_id,
                "stage": "pipeline",
                "error": str(e),
            }), 500

    # --- Mailer step ---
    if run_mailer:
        mailer_payload = {
            "process_id": process_id,
            "report_gcs_path": report_gcs_path,
        }
        if recipients:
            mailer_payload["recipients"] = recipients

        try:
            logger.info("Calling mailer | process_id=%s", process_id)
            mailer_response = _call_function(mailer_url, mailer_payload, timeout=120)
            logger.info("Mailer succeeded | process_id=%s", process_id)
        except Exception as e:
            logger.exception("Mailer call failed | process_id=%s", process_id)
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

    # --- Success: write final manifest ---
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
    )
    _write_manifest(gcs_client, base_gcs_path, final_manifest)

    logger.info("Orchestrator completed | process_id=%s | duration=%.1fs", process_id, duration)

    return jsonify({
        "status": "success",
        "process_id": process_id,
        "duration_seconds": round(duration, 1),
        "report_gcs_path": report_gcs_path,
    }), 200


def _base_gcs_path_from_pdf(bucket: str, object_name: str) -> str | None:
    """Derive the CT base GCS path from an uploaded PDF object.

    Returns the base path (gs://bucket/.../CT_xxx) if the object is a PDF under
    ``<TRIGGER_PREFIX>/.../ResultingObjects/``, else None (event ignored).
    """
    # Scope: only objects under the configured prefix are considered.
    if not object_name.startswith(f"{TRIGGER_PREFIX}/") and object_name != TRIGGER_PREFIX:
        return None
    # Only PDFs trigger processing.
    if not object_name.lower().endswith(".pdf"):
        return None
    # Must live inside a ResultingObjects/ subfolder.
    if RESULTING_OBJECTS_MARKER not in object_name:
        return None
    # The CT folder is everything before "/ResultingObjects/".
    base_blob = object_name.split(RESULTING_OBJECTS_MARKER, 1)[0]
    if not base_blob:
        return None
    return f"gs://{bucket}/{base_blob}"


def _handle_cloud_event(request: Request):
    """Parse a GCS finalize CloudEvent and route to _run if it's a scoped PDF."""
    try:
        event = from_http(request.headers, request.get_data())
    except Exception as e:
        logger.error("Failed to parse CloudEvent: %s", e)
        return jsonify({"error": "Invalid CloudEvent"}), 400

    data = event.data or {}
    bucket = data.get("bucket", "")
    object_name = data.get("name", "")

    if not bucket or not object_name:
        logger.info("CloudEvent missing bucket/name, ignoring | data=%s", data)
        return jsonify({"status": "ignored", "reason": "missing bucket or name"}), 200

    base_gcs_path = _base_gcs_path_from_pdf(bucket, object_name)
    if base_gcs_path is None:
        logger.info("Object out of scope, ignoring | object=%s", object_name)
        return jsonify({
            "status": "ignored",
            "object": object_name,
            "reason": (
                f"not a PDF under {TRIGGER_PREFIX}/.../ResultingObjects/"
            ),
        }), 200

    logger.info(
        "Scoped PDF detected | object=%s | base_gcs_path=%s",
        object_name, base_gcs_path,
    )

    # Settle delay: if a second PDF is still in flight, wait for it to land so
    # _resolve_pdf_paths sees the full set and picks comparison vs single mode
    # correctly. The atomic manifest claim in _run handles the duplicate event.
    if SETTLE_DELAY_SECONDS > 0:
        logger.info("Settling %ds before resolving PDFs | %s", SETTLE_DELAY_SECONDS, base_gcs_path)
        time.sleep(SETTLE_DELAY_SECONDS)

    return _run(base_gcs_path)


def _handle_direct_json(request: Request):
    """Handle a direct JSON invocation carrying base_gcs_path."""
    try:
        payload = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON payload"}), 400

    if not payload or "base_gcs_path" not in payload:
        return jsonify({"error": "Missing required field: base_gcs_path"}), 400

    return _run(payload["base_gcs_path"], recipients=payload.get("recipients"))


# Flask app for Cloud Run deployment
app = Flask(__name__)


@app.route("/", methods=["POST"])
def handle_request():
    # Eventarc delivers CloudEvents with ce-* headers; a manual call sends plain JSON.
    if "ce-id" in flask_request.headers or "ce-type" in flask_request.headers:
        return _handle_cloud_event(flask_request)
    return _handle_direct_json(flask_request)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200
