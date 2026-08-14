"""Cloud Run Function 3: Mailer.

Sends an email with the CAD review report PDF attached.
SMTP credentials are retrieved from Google Secret Manager.

Expected JSON payload:
{
    "process_id": "abc-123",
    "report_gcs_path": "gs://bucket/process/abc-123/PROCESSING_OUTPUTS/integrated_review_report.pdf",
    "subject": "Optional custom subject (uses default if omitted)",
    "body": "Optional custom body (uses default if omitted)"
}

Environment variables:
    SMTP_HOST               - SMTP server address (e.g. smtp.gmail.com)
    SMTP_PORT               - SMTP server port (default: 587)
    SECRET_PROJECT_ID       - GCP project ID where secrets are stored
    SECRET_SMTP_USER        - Secret Manager secret name for the SMTP username
    SECRET_SMTP_PASSWORD    - Secret Manager secret name for the SMTP password
    MAIL_RECIPIENTS         - Comma-separated list of recipient email addresses
    MAIL_SENDER             - Sender email address (From header)
"""

from __future__ import annotations

import logging
import os
import smtplib
from email import encoders
from email.mime.base64 import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import functions_framework
from flask import Request, jsonify
from google.cloud import secretmanager, storage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Defaults
DEFAULT_SUBJECT = "[CAD Review] Relatório de revisão disponível"
DEFAULT_BODY = (
    "Prezado(a),\n\n"
    "O relatório de revisão de CAD foi gerado com sucesso.\n"
    "Segue em anexo o PDF do relatório integrado.\n\n"
    "Atenciosamente,\n"
    "Sistema de Revisão CAD"
)


def _get_secret(project_id: str, secret_name: str, version: str = "latest") -> str:
    """Retrieve a secret value from Google Secret Manager."""
    client = secretmanager.SecretManagerServiceClient()
    resource_name = f"projects/{project_id}/secrets/{secret_name}/versions/{version}"
    response = client.access_secret_version(request={"name": resource_name})
    return response.payload.data.decode("UTF-8")


def _parse_gcs_path(gcs_path: str) -> tuple[str, str]:
    """Parse a gs://bucket/path string into (bucket_name, blob_path)."""
    if not gcs_path.startswith("gs://"):
        raise ValueError(f"Invalid GCS path (must start with gs://): {gcs_path}")
    parts = gcs_path[5:].split("/", 1)
    if len(parts) < 2:
        raise ValueError(f"Invalid GCS path (missing object path): {gcs_path}")
    return parts[0], parts[1]


def _download_blob(gcs_path: str) -> bytes:
    """Download a GCS object and return its bytes."""
    bucket_name, blob_path = _parse_gcs_path(gcs_path)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    return blob.download_as_bytes()


def _send_email(
    smtp_host: str,
    smtp_port: int,
    smtp_user: str,
    smtp_password: str,
    sender: str,
    recipients: list[str],
    subject: str,
    body: str,
    attachment_bytes: bytes,
    attachment_filename: str,
) -> None:
    """Compose and send an email with a PDF attachment via SMTP/TLS."""
    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = subject

    # Body
    msg.attach(MIMEText(body, "plain", "utf-8"))

    # PDF attachment
    part = MIMEBase("application", "pdf")
    part.set_payload(attachment_bytes)
    encoders.encode_base64(part)
    part.add_header("Content-Disposition", f'attachment; filename="{attachment_filename}"')
    msg.attach(part)

    # Send via SMTP with STARTTLS
    logger.info("Connecting to SMTP %s:%d", smtp_host, smtp_port)
    with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(smtp_user, smtp_password)
        server.sendmail(sender, recipients, msg.as_string())

    logger.info("Email sent to %d recipients", len(recipients))


@functions_framework.http
def mailer(request: Request):
    """HTTP Cloud Run function entry point for the mailer."""
    # Parse request
    try:
        payload = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON payload"}), 400

    if "report_gcs_path" not in payload:
        return jsonify({"error": "Missing required field: report_gcs_path"}), 400

    process_id = payload.get("process_id", "unknown")
    report_gcs_path = payload["report_gcs_path"]
    subject = payload.get("subject", DEFAULT_SUBJECT)
    body = payload.get("body", DEFAULT_BODY)

    logger.info("Mailer started | process_id=%s", process_id)

    # Read environment configuration
    smtp_host = os.environ.get("SMTP_HOST", "")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    secret_project_id = os.environ.get("SECRET_PROJECT_ID", "")
    secret_smtp_user = os.environ.get("SECRET_SMTP_USER", "")
    secret_smtp_password = os.environ.get("SECRET_SMTP_PASSWORD", "")
    mail_recipients_raw = os.environ.get("MAIL_RECIPIENTS", "")
    mail_sender = os.environ.get("MAIL_SENDER", "")

    # Validate env vars
    missing_env = []
    if not smtp_host:
        missing_env.append("SMTP_HOST")
    if not secret_project_id:
        missing_env.append("SECRET_PROJECT_ID")
    if not secret_smtp_user:
        missing_env.append("SECRET_SMTP_USER")
    if not secret_smtp_password:
        missing_env.append("SECRET_SMTP_PASSWORD")
    if not mail_recipients_raw:
        missing_env.append("MAIL_RECIPIENTS")
    if not mail_sender:
        missing_env.append("MAIL_SENDER")

    if missing_env:
        return jsonify({"error": f"Missing environment variables: {missing_env}"}), 500

    recipients = [r.strip() for r in mail_recipients_raw.split(",") if r.strip()]
    if not recipients:
        return jsonify({"error": "MAIL_RECIPIENTS is empty after parsing"}), 500

    try:
        # Retrieve SMTP credentials from Secret Manager
        logger.info("Retrieving SMTP credentials from Secret Manager")
        smtp_user = _get_secret(secret_project_id, secret_smtp_user)
        smtp_password = _get_secret(secret_project_id, secret_smtp_password)

        # Download the report PDF from GCS
        logger.info("Downloading report PDF: %s", report_gcs_path)
        report_pdf_bytes = _download_blob(report_gcs_path)
        attachment_filename = report_gcs_path.rsplit("/", 1)[-1]

        # Send the email
        _send_email(
            smtp_host=smtp_host,
            smtp_port=smtp_port,
            smtp_user=smtp_user,
            smtp_password=smtp_password,
            sender=mail_sender,
            recipients=recipients,
            subject=subject,
            body=body,
            attachment_bytes=report_pdf_bytes,
            attachment_filename=attachment_filename,
        )

        logger.info("Mailer completed | process_id=%s", process_id)
        return jsonify({
            "status": "success",
            "process_id": process_id,
            "recipients": recipients,
        }), 200

    except Exception as e:
        logger.exception("Mailer failed | process_id=%s", process_id)
        return jsonify({
            "status": "error",
            "process_id": process_id,
            "error": str(e),
        }), 500
