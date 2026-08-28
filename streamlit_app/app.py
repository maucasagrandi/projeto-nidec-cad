"""CAD Review Test Interface.

Streamlit app for testing the CAD Review pipeline.
Uploads PDFs to GCS, triggers the orchestrator, waits for the report, and
returns the generated PDF to the user.

Authentication:
    Uses Application Default Credentials (ADC). Run `gcloud auth login`
    and `gcloud auth application-default login` on the host before starting.

Environment variables (optional, have sensible defaults):
    GCS_BUCKET          - Target GCS bucket (default: acim-global-data-lake-sandbox-temp)
    GCS_BASE_PATH       - Base path prefix (default: temp/Windchill/cadreview)
    ORCHESTRATOR_URL    - Orchestrator Cloud Run URL
"""

import io
import os
import time

import google.auth
import google.auth.transport.requests
import google.oauth2.id_token
import requests
import streamlit as st
from google.cloud import storage

# ── Configuration ─────────────────────────────────────────────────────────────
GCS_BUCKET = os.getenv("GCS_BUCKET", "acim-global-data-lake-sandbox-temp")
GCS_BASE_PATH = os.getenv("GCS_BASE_PATH", "temp/Windchill/cadreview")
ORCHESTRATOR_URL = os.getenv(
    "ORCHESTRATOR_URL",
    "https://cad-review-orchestrator-1085633511117.us-central1.run.app",
)


# ── Helper functions ──────────────────────────────────────────────────────────
@st.cache_resource
def get_gcs_client():
    """Initialize GCS client using ADC."""
    return storage.Client()


def upload_pdf_to_gcs(client: storage.Client, file_bytes: bytes, gcs_path: str) -> str:
    """Upload a PDF file to GCS and return the gs:// path."""
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(gcs_path)
    blob.upload_from_string(file_bytes, content_type="application/pdf")
    return f"gs://{GCS_BUCKET}/{gcs_path}"


def delete_manifest(client: storage.Client, ct_code: str) -> None:
    """Delete manifest.txt if it exists (allow re-processing)."""
    blob_path = f"{GCS_BASE_PATH}/{ct_code}/manifest.txt"
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(blob_path)
    if blob.exists():
        blob.delete()


def get_id_token(audience: str) -> str:
    """Get an ID token for authenticating to Cloud Run."""
    auth_req = google.auth.transport.requests.Request()
    token = google.oauth2.id_token.fetch_id_token(auth_req, audience)
    return token


def trigger_orchestrator(base_gcs_path: str) -> dict:
    """Call the orchestrator Cloud Run service."""
    token = get_id_token(ORCHESTRATOR_URL)
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {"base_gcs_path": base_gcs_path}
    response = requests.post(ORCHESTRATOR_URL, json=payload, headers=headers, timeout=1200)
    return {"status_code": response.status_code, "body": response.json()}


def wait_for_report(client: storage.Client, ct_code: str, timeout: int = 900) -> bytes | None:
    """Poll GCS for the report PDF until it appears or timeout."""
    report_path = f"{GCS_BASE_PATH}/{ct_code}/PROCESSING_OUTPUTS/integrated_review_report.pdf"
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(report_path)

    start = time.time()
    while time.time() - start < timeout:
        if blob.exists():
            return blob.download_as_bytes()
        time.sleep(5)
    return None


def download_report(client: storage.Client, ct_code: str) -> bytes | None:
    """Download the report PDF from GCS (no waiting)."""
    report_path = f"{GCS_BASE_PATH}/{ct_code}/PROCESSING_OUTPUTS/integrated_review_report.pdf"
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(report_path)
    if blob.exists():
        return blob.download_as_bytes()
    return None


# ── Streamlit UI ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="CAD Review Test", page_icon="📐", layout="centered")
st.title("📐 CAD Review — Test Interface")
st.markdown("Upload 1 or 2 PDF drawings to generate the integrated review report.")

# Input fields
ct_code = st.text_input(
    "Change Task Code",
    placeholder="CT_219344",
    help="The CT folder name (e.g. CT_219344, CT_13358002)",
)

uploaded_files = st.file_uploader(
    "PDF Drawings (1 for single analysis, 2 for comparison)",
    type=["pdf"],
    accept_multiple_files=True,
)

# Validation
if uploaded_files and len(uploaded_files) > 2:
    st.error("Maximum 2 PDF files allowed.")
    st.stop()

# Run button
if st.button("Run CAD Review", type="primary", disabled=not ct_code or not uploaded_files):
    if not ct_code.strip():
        st.error("Please enter a Change Task code.")
        st.stop()

    ct_code = ct_code.strip()
    num_pdfs = len(uploaded_files)
    mode = "comparison" if num_pdfs == 2 else "single"

    st.info(f"Mode: **{mode}** ({num_pdfs} PDF{'s' if num_pdfs > 1 else ''})")

    client = get_gcs_client()

    # Step 1: Upload PDFs to GCS
    with st.status("Uploading PDFs to GCS...", expanded=True) as status:
        for uploaded_file in uploaded_files:
            file_bytes = uploaded_file.read()
            gcs_path = f"{GCS_BASE_PATH}/{ct_code}/ResultingObjects/{uploaded_file.name}"
            full_path = upload_pdf_to_gcs(client, file_bytes, gcs_path)
            st.write(f"Uploaded: `{full_path}`")
        status.update(label="PDFs uploaded", state="complete")

    # Step 2: Delete existing manifest (allow re-run)
    with st.status("Preparing execution...", expanded=False) as status:
        delete_manifest(client, ct_code)
        status.update(label="Ready to execute", state="complete")

    # Step 3: Trigger orchestrator
    base_gcs_path = f"gs://{GCS_BUCKET}/{GCS_BASE_PATH}/{ct_code}"

    with st.status("Running CAD Review pipeline...", expanded=True) as status:
        st.write(f"Calling orchestrator for `{ct_code}`...")
        st.write(f"Base path: `{base_gcs_path}`")

        try:
            result = trigger_orchestrator(base_gcs_path)
            status_code = result["status_code"]
            body = result["body"]

            if status_code == 200 and body.get("status") == "success":
                st.write(f"Pipeline completed in **{body.get('duration_seconds', '?')}s**")
                status.update(label="Pipeline completed", state="complete")
            elif body.get("status") == "error" and body.get("stage") == "mailer":
                # Pipeline OK, mailer failed (expected — SMTP issue)
                st.write(f"Pipeline succeeded. Mailer skipped (SMTP not configured).")
                st.write(f"Report: `{body.get('report_gcs_path', '')}`")
                status.update(label="Pipeline completed (mailer skipped)", state="complete")
            else:
                st.error(f"Orchestrator returned {status_code}: {body}")
                status.update(label="Pipeline failed", state="error")
                st.stop()
        except requests.exceptions.Timeout:
            st.warning("Request timed out. Waiting for report in GCS...")
            status.update(label="Waiting for report...", state="running")
        except Exception as e:
            st.error(f"Error calling orchestrator: {e}")
            status.update(label="Error", state="error")
            st.stop()

    # Step 4 & 5: Download report
    with st.status("Downloading report...", expanded=False) as status:
        report_bytes = download_report(client, ct_code)

        if report_bytes is None:
            st.write("Report not immediately available, polling GCS...")
            report_bytes = wait_for_report(client, ct_code, timeout=300)

        if report_bytes:
            status.update(label="Report ready", state="complete")
        else:
            status.update(label="Report not found", state="error")
            st.error("Timed out waiting for the report PDF.")
            st.stop()

    # Step 6: Offer download
    st.success(f"Report generated successfully ({len(report_bytes) / 1024:.1f} KB)")
    st.download_button(
        label="Download Report PDF",
        data=report_bytes,
        file_name=f"{ct_code}_review_report.pdf",
        mime="application/pdf",
        type="primary",
    )
