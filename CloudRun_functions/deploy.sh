#!/usr/bin/env bash
# ==============================================================================
# Deploy all three Cloud Run functions for the CAD Review pipeline.
#
# Prerequisites:
#   - gcloud CLI authenticated and configured
#   - GCP project set (or pass --project flag)
#   - Required secrets already created in Secret Manager
#
# Usage:
#   ./deploy.sh <GCP_PROJECT_ID> <GCP_REGION>
#
# Example:
#   ./deploy.sh acim-global-data-lake-sandbox us-east5
# ==============================================================================

set -euo pipefail

PROJECT_ID="${1:?Usage: ./deploy.sh <GCP_PROJECT_ID> <GCP_REGION>}"
REGION="${2:?Usage: ./deploy.sh <GCP_PROJECT_ID> <GCP_REGION>}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Deploying CAD Review Cloud Run Functions ==="
echo "Project: ${PROJECT_ID}"
echo "Region:  ${REGION}"
echo ""

# ------------------------------------------------------------------------------
# Function 2: Pipeline (Docker-based Cloud Run service)
# Deployed as a Cloud Run service because it bundles the full project source
# (src/, prompts.py, assets/) which exceeds simple function source limits.
# ------------------------------------------------------------------------------
echo "--- Deploying: pipeline (Cloud Run service with Dockerfile) ---"
gcloud run deploy cad-review-pipeline \
    --region="${REGION}" \
    --project="${PROJECT_ID}" \
    --source="${SCRIPT_DIR}/pipeline" \
    --no-allow-unauthenticated \
    --memory=4Gi \
    --timeout=900 \
    --cpu=2 \
    --min-instances=0 \
    --max-instances=5 \
    --set-env-vars="GCP_PROJECT_ID=${PROJECT_ID},GCP_REGION=${REGION}" \
    --quiet

PIPELINE_URL=$(gcloud run services describe cad-review-pipeline \
    --region="${REGION}" --project="${PROJECT_ID}" \
    --format="value(status.url)")

echo "Pipeline URL: ${PIPELINE_URL}"
echo ""

# ------------------------------------------------------------------------------
# Function 3: Mailer
# ------------------------------------------------------------------------------
echo "--- Deploying: mailer ---"
gcloud functions deploy cad-review-mailer \
    --gen2 \
    --region="${REGION}" \
    --project="${PROJECT_ID}" \
    --runtime=python311 \
    --source="${SCRIPT_DIR}/mailer" \
    --entry-point=mailer \
    --trigger-http \
    --no-allow-unauthenticated \
    --memory=512Mi \
    --timeout=120s \
    --set-env-vars="SMTP_HOST=<SMTP_HOST>,SMTP_PORT=587,SECRET_PROJECT_ID=${PROJECT_ID},SECRET_SMTP_USER=smtp-user,SECRET_SMTP_PASSWORD=smtp-password,MAIL_RECIPIENTS=<RECIPIENTS>,MAIL_SENDER=<SENDER>"

MAILER_URL=$(gcloud functions describe cad-review-mailer \
    --gen2 --region="${REGION}" --project="${PROJECT_ID}" \
    --format="value(serviceConfig.uri)")

echo "Mailer URL: ${MAILER_URL}"
echo ""

# ------------------------------------------------------------------------------
# Function 1: Orchestrator
# ------------------------------------------------------------------------------
echo "--- Deploying: orchestrator ---"
gcloud functions deploy cad-review-orchestrator \
    --gen2 \
    --region="${REGION}" \
    --project="${PROJECT_ID}" \
    --runtime=python311 \
    --source="${SCRIPT_DIR}/orchestrator" \
    --entry-point=orchestrator \
    --trigger-http \
    --no-allow-unauthenticated \
    --memory=512Mi \
    --timeout=1200s \
    --set-env-vars="PIPELINE_FUNCTION_URL=${PIPELINE_URL},MAILER_FUNCTION_URL=${MAILER_URL}"

ORCHESTRATOR_URL=$(gcloud functions describe cad-review-orchestrator \
    --gen2 --region="${REGION}" --project="${PROJECT_ID}" \
    --format="value(serviceConfig.uri)")

echo ""
echo "=== Deployment Complete ==="
echo "Orchestrator URL: ${ORCHESTRATOR_URL}"
echo "Pipeline URL:     ${PIPELINE_URL}"
echo "Mailer URL:       ${MAILER_URL}"
echo ""
echo "NOTE: Update the mailer env vars (SMTP_HOST, MAIL_RECIPIENTS, MAIL_SENDER)"
echo "      with actual values using:"
echo "  gcloud functions deploy cad-review-mailer --gen2 --region=${REGION} --update-env-vars=..."
