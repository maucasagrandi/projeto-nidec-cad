"""Cloud Run Service: Mailer.

Sends an email with the CAD review report PDF attached.
SMTP credentials are provided via environment variables.
The SMTP password is injected from a Secret Manager secret via Cloud Run's
native secretKeyRef mechanism (already resolved as an env var at runtime).

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
    SECRET_SMTP_USER        - SMTP username (plain value, e.g. it.apps@nidec-ga.com)
    SECRET_SMTP_PASSWORD    - SMTP password (injected from Secret Manager via secretKeyRef)
    MAIL_RECIPIENTS         - Comma-separated list of recipient email addresses
    MAIL_SENDER             - Sender email address (From header)
"""

from __future__ import annotations

import logging
import os
import shutil
import smtplib
import subprocess
import tempfile
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from flask import Flask, Request, jsonify, request as flask_request
from google.cloud import storage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Maximum attachment size (bytes) we will inline into the email. Gmail rejects
# messages larger than ~25 MB and base64 encoding inflates the payload ~33%, so
# we cap the raw PDF well below that. Reports above this size are recompressed
# with Ghostscript before attaching.
MAX_ATTACHMENT_BYTES = int(os.environ.get("MAX_ATTACHMENT_BYTES", str(18 * 1024 * 1024)))

# Defaults
def _default_subject(process_id: str) -> str:
    """Build the default subject, including the CT code when available."""
    if process_id and process_id != "unknown":
        return f"[CAD Review] {process_id}: Relatório de revisão disponível"
    return "[CAD Review] Relatório de revisão disponível"


DEFAULT_BODY = (
    "Prezado(a),\n\n"
    "O relatório de revisão de CAD foi gerado com sucesso.\n"
    "Segue em anexo o PDF do relatório integrado.\n\n"
    "Atenciosamente,\n"
    "Sistema de Revisão CAD"
)


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


def _compress_pdf(pdf_bytes: bytes) -> bytes | None:
    """Recompress a PDF with Ghostscript to shrink it for email attachment.

    Downsamples embedded images to ~300 DPI (Ghostscript ``/printer`` preset),
    which reduces the size of high-DPI CAD report images while keeping text and
    line work crisp. Returns the compressed bytes, or None if Ghostscript is
    unavailable or the run fails (caller keeps the original).
    """
    gs_bin = shutil.which("gs")
    if not gs_bin:
        logger.warning("Ghostscript (gs) not found on PATH — cannot compress PDF")
        return None
    try:
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "in.pdf")
            dst = os.path.join(tmp, "out.pdf")
            with open(src, "wb") as f:
                f.write(pdf_bytes)
            cmd = [
                gs_bin,
                "-sDEVICE=pdfwrite",
                "-dCompatibilityLevel=1.5",
                "-dPDFSETTINGS=/printer",  # ~300 DPI image downsampling
                "-dNOPAUSE",
                "-dBATCH",
                "-dQUIET",
                "-dDetectDuplicateImages=true",
                f"-sOutputFile={dst}",
                src,
            ]
            subprocess.run(cmd, check=True, timeout=90, capture_output=True)
            with open(dst, "rb") as f:
                return f.read()
    except Exception as e:  # noqa: BLE001 - best-effort; caller keeps original
        logger.warning("PDF compression failed: %s", e)
        return None


def _send_email(
    smtp_host: str,
    smtp_port: int,
    smtp_user: str,
    smtp_password: str,
    sender: str,
    recipients: list[str],
    subject: str,
    body: str,
    attachment_bytes: bytes | None,
    attachment_filename: str,
) -> None:
    """Compose and send an email via SMTP.

    When ``attachment_bytes`` is provided the PDF is attached; otherwise the
    message is sent body-only (used when the report is too large to attach and
    a download link is embedded in the body instead).
    """
    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = subject

    # Body
    msg.attach(MIMEText(body, "plain", "utf-8"))

    # PDF attachment (optional)
    if attachment_bytes is not None:
        part = MIMEBase("application", "pdf")
        part.set_payload(attachment_bytes)
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f'attachment; filename="{attachment_filename}"')
        msg.attach(part)

    # Send via SMTP — use SSL for port 465, STARTTLS for port 587
    logger.info("Connecting to SMTP %s:%d", smtp_host, smtp_port)

    if smtp_port == 465:
        with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=30) as server:
            if smtp_user and smtp_password:
                server.login(smtp_user, smtp_password)
            server.sendmail(sender, recipients, msg.as_string())
    else:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=60) as server:
            server.ehlo()
            # Attempt STARTTLS if available
            if server.has_extn("STARTTLS"):
                server.starttls()
                server.ehlo()
                logger.info("STARTTLS established")
            else:
                logger.info("STARTTLS not available, continuing without encryption")
            # Log in only when the relay advertises AUTH. The Google SMTP relay
            # alternates between IP-based auth (no AUTH offered) and credential
            # auth depending on config; when AUTH is absent we rely on IP-based
            # authorization and send without logging in.
            if smtp_user and smtp_password and server.has_extn("AUTH"):
                logger.info("AUTH available, logging in as %s", smtp_user)
                server.login(smtp_user, smtp_password)
            else:
                logger.info("AUTH not offered, sending without login (IP-based auth)")
            server.sendmail(sender, recipients, msg.as_string())

    logger.info("Email sent to %d recipients", len(recipients))


def mailer(request: Request):
    """HTTP Cloud Run entry point for the mailer."""
    # Parse request
    try:
        payload = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON payload"}), 400

    if "report_gcs_path" not in payload:
        return jsonify({"error": "Missing required field: report_gcs_path"}), 400

    process_id = payload.get("process_id", "unknown")
    report_gcs_path = payload["report_gcs_path"]
    subject = payload.get("subject", _default_subject(process_id))
    body = payload.get("body", DEFAULT_BODY)
    # Optional per-request recipient override (list or comma-separated string).
    # Falls back to the MAIL_RECIPIENTS env var when absent.
    recipients_override = payload.get("recipients")

    logger.info("Mailer started | process_id=%s", process_id)

    # Read environment configuration — all values are plain env vars
    smtp_host = os.environ.get("SMTP_HOST", "")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_user = os.environ.get("SECRET_SMTP_USER", "")
    smtp_password = os.environ.get("SECRET_SMTP_PASSWORD", "")
    mail_recipients_raw = os.environ.get("MAIL_RECIPIENTS", "")
    mail_sender = os.environ.get("MAIL_SENDER", "")

    # Resolve recipients: payload override takes precedence over env var
    if recipients_override:
        if isinstance(recipients_override, str):
            recipients = [r.strip() for r in recipients_override.split(",") if r.strip()]
        else:
            recipients = [str(r).strip() for r in recipients_override if str(r).strip()]
    else:
        recipients = [r.strip() for r in mail_recipients_raw.split(",") if r.strip()]

    # Validate env vars (recipients validated separately below)
    missing_env = []
    if not smtp_host:
        missing_env.append("SMTP_HOST")
    if not smtp_user:
        missing_env.append("SECRET_SMTP_USER")
    if not smtp_password:
        missing_env.append("SECRET_SMTP_PASSWORD")
    if not mail_sender:
        missing_env.append("MAIL_SENDER")

    if missing_env:
        return jsonify({"error": f"Missing environment variables: {missing_env}"}), 500

    if not recipients:
        return jsonify({
            "error": "No recipients: provide 'recipients' in payload or set MAIL_RECIPIENTS env var",
        }), 500

    logger.info("Recipients: %s (%s)", recipients, "override" if recipients_override else "env")

    try:
        # Download the report PDF from GCS
        logger.info("Downloading report PDF: %s", report_gcs_path)
        report_pdf_bytes = _download_blob(report_gcs_path)
        report_size = len(report_pdf_bytes)
        base_filename = report_gcs_path.rsplit("/", 1)[-1]
        # Prefix the attachment with the CT code so it is identifiable
        if process_id and process_id != "unknown" and not base_filename.startswith(process_id):
            attachment_filename = f"{process_id}_{base_filename}"
        else:
            attachment_filename = base_filename

        # If the report is too large to attach (Gmail ~25 MB cap, base64 inflates
        # the payload), recompress it with Ghostscript before attaching.
        delivery = "attachment"
        send_bytes = report_pdf_bytes
        if report_size > MAX_ATTACHMENT_BYTES:
            logger.info(
                "Report %d bytes exceeds attachment cap %d — compressing",
                report_size, MAX_ATTACHMENT_BYTES,
            )
            compressed = _compress_pdf(report_pdf_bytes)
            if compressed and len(compressed) < report_size:
                logger.info(
                    "Compressed report %d -> %d bytes", report_size, len(compressed)
                )
                send_bytes = compressed
                delivery = "attachment_compressed"

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
            attachment_bytes=send_bytes,
            attachment_filename=attachment_filename,
        )

        logger.info("Mailer completed | process_id=%s | delivery=%s", process_id, delivery)
        return jsonify({
            "status": "success",
            "process_id": process_id,
            "recipients": recipients,
            "delivery": delivery,
            "attachment_bytes": len(send_bytes),
        }), 200

    except Exception as e:
        logger.exception("Mailer failed | process_id=%s", process_id)
        return jsonify({
            "status": "error",
            "process_id": process_id,
            "error": str(e),
        }), 500


# Flask app for Cloud Run deployment
app = Flask(__name__)


@app.route("/", methods=["POST"])
def handle_request():
    return mailer(flask_request)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200
