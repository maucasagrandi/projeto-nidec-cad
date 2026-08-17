#!/usr/bin/env bash
# ==============================================================================
# Deploy all three Cloud Run services for the CAD Review pipeline.
#
# Builds Docker images locally, pushes to Artifact Registry, and deploys
# to Cloud Run. No Cloud Build permissions required.
#
# Prerequisites:
#   - gcloud CLI authenticated and configured
#   - Docker installed and accessible (via sudo)
#   - GCP project set (or pass --project flag)
#   - Artifact Registry repo exists in AR_REGION
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

# Artifact Registry configuration
AR_REGION="us-central1"
AR_REPO="cloud-run-source-deploy"
AR_BASE="${AR_REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}"

echo "=== Deploying CAD Review Cloud Run Services ==="
echo "Project:  ${PROJECT_ID}"
echo "Region:   ${REGION}"
echo "Registry: ${AR_BASE}"
echo ""

# ------------------------------------------------------------------------------
# Authenticate Docker with Artifact Registry
# ------------------------------------------------------------------------------
echo "--- Authenticating Docker with Artifact Registry ---"
gcloud auth print-access-token | sudo docker login -u oauth2accesstoken --password-stdin "${AR_REGION}-docker.pkg.dev"
echo ""

# ------------------------------------------------------------------------------
# Service 1: Pipeline
# ------------------------------------------------------------------------------
PIPELINE_IMAGE="${AR_BASE}/cad-review-pipeline:latest"

echo "--- Building: pipeline ---"
sudo docker build -t "${PIPELINE_IMAGE}" "${SCRIPT_DIR}/pipeline"

echo "--- Pushing: pipeline ---"
sudo docker push "${PIPELINE_IMAGE}"

echo "--- Deploying: pipeline (Cloud Run service) ---"
gcloud run deploy cad-review-pipeline \
    --region="${REGION}" \
    --project="${PROJECT_ID}" \
    --image="${PIPELINE_IMAGE}" \
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
# Service 2: Mailer
# ------------------------------------------------------------------------------
MAILER_IMAGE="${AR_BASE}/cad-review-mailer:latest"

echo "--- Building: mailer ---"
sudo docker build -t "${MAILER_IMAGE}" "${SCRIPT_DIR}/mailer"

echo "--- Pushing: mailer ---"
sudo docker push "${MAILER_IMAGE}"

echo "--- Deploying: mailer (Cloud Run service) ---"
gcloud run deploy cad-review-mailer \
    --region="${REGION}" \
    --project="${PROJECT_ID}" \
    --image="${MAILER_IMAGE}" \
    --no-allow-unauthenticated \
    --memory=512Mi \
    --timeout=120 \
    --cpu=1 \
    --min-instances=0 \
    --max-instances=3 \
    --set-env-vars="SMTP_HOST=<SMTP_HOST>,SMTP_PORT=587,SECRET_PROJECT_ID=${PROJECT_ID},SECRET_SMTP_USER=smtp-user,SECRET_SMTP_PASSWORD=smtp-password,MAIL_RECIPIENTS=<RECIPIENTS>,MAIL_SENDER=<SENDER>" \
    --quiet

MAILER_URL=$(gcloud run services describe cad-review-mailer \
    --region="${REGION}" --project="${PROJECT_ID}" \
    --format="value(status.url)")

echo "Mailer URL: ${MAILER_URL}"
echo ""

# ------------------------------------------------------------------------------
# Service 3: Orchestrator
# ------------------------------------------------------------------------------
ORCHESTRATOR_IMAGE="${AR_BASE}/cad-review-orchestrator:latest"

echo "--- Building: orchestrator ---"
sudo docker build -t "${ORCHESTRATOR_IMAGE}" "${SCRIPT_DIR}/orchestrator"

echo "--- Pushing: orchestrator ---"
sudo docker push "${ORCHESTRATOR_IMAGE}"

echo "--- Deploying: orchestrator (Cloud Run service) ---"
gcloud run deploy cad-review-orchestrator \
    --region="${REGION}" \
    --project="${PROJECT_ID}" \
    --image="${ORCHESTRATOR_IMAGE}" \
    --no-allow-unauthenticated \
    --memory=512Mi \
    --timeout=1200 \
    --cpu=1 \
    --min-instances=0 \
    --max-instances=3 \
    --set-env-vars="PIPELINE_FUNCTION_URL=${PIPELINE_URL},MAILER_FUNCTION_URL=${MAILER_URL}" \
    --quiet

ORCHESTRATOR_URL=$(gcloud run services describe cad-review-orchestrator \
    --region="${REGION}" --project="${PROJECT_ID}" \
    --format="value(status.url)")

echo ""
echo "=== Deployment Complete ==="
echo "Orchestrator URL: ${ORCHESTRATOR_URL}"
echo "Pipeline URL:     ${PIPELINE_URL}"
echo "Mailer URL:       ${MAILER_URL}"
echo ""
echo "NOTE: Update the mailer env vars (SMTP_HOST, MAIL_RECIPIENTS, MAIL_SENDER)"
echo "      with actual values using:"
echo "  gcloud run services update cad-review-mailer --region=${REGION} --update-env-vars=SMTP_HOST=...,MAIL_RECIPIENTS=...,MAIL_SENDER=..."
