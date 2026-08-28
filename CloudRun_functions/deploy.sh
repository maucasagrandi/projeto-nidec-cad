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
#   ./deploy.sh <GCP_PROJECT_ID> <GCP_REGION> <TRIGGER_BUCKET> <TRIGGER_LOCATION> [TRIGGER_SA]
#
#   TRIGGER_LOCATION must match the bucket location: a multi-region value like
#   'us' or 'eu' for a multi-region bucket, or a region like 'us-central1' for
#   a regional bucket.
#
# Example (multi-region US bucket, services in us-central1):
#   ./deploy.sh acim-global-data-lake-sandbox us-central1 my-windchill-bucket us
# ==============================================================================

set -euo pipefail

USAGE="Usage: ./deploy.sh <GCP_PROJECT_ID> <GCP_REGION> <TRIGGER_BUCKET> <TRIGGER_LOCATION> [TRIGGER_SA]"

PROJECT_ID="${1:?${USAGE}}"
REGION="${2:?${USAGE}}"
TRIGGER_BUCKET="${3:?${USAGE}}"
# Eventarc trigger location — MUST match the bucket's location. For a
# multi-region bucket use the multi-region value (e.g. 'us' or 'eu'); for a
# regional bucket use its region (e.g. 'us-central1'). Cloud Storage event
# triggers support single- and multi-region locations.
TRIGGER_LOCATION="${4:?${USAGE}}"
# Service account Eventarc uses to invoke the orchestrator. Defaults to the
# project's Compute Engine default SA if not supplied.
PROJECT_NUMBER="$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')"
TRIGGER_SA="${5:-${PROJECT_NUMBER}-compute@developer.gserviceaccount.com}"

# Only objects under this prefix trigger the orchestrator (filtered in code,
# since Eventarc GCS triggers cannot filter by object prefix).
TRIGGER_PREFIX="temp/Windchill/cadreview"
# Seconds to wait after a PDF event before resolving the PDF set, so a second
# PDF arriving moments later is included (comparison vs single-PDF accuracy).
SETTLE_DELAY_SECONDS="5"

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
    --memory=8Gi \
    --timeout=900 \
    --cpu=4 \
    --min-instances=0 \
    --max-instances=5 \
    --set-env-vars="GCP_PROJECT_ID=${PROJECT_ID},GCP_REGION=global,GDT_WORKERS=4" \
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
    --set-env-vars="SMTP_HOST=smtp.gmail.com,SMTP_PORT=587,SECRET_SMTP_USER=it.apps@nidec-ga.com,MAIL_RECIPIENTS=elvis.cantelli@madeinweb.com.br,MAIL_SENDER=do-not-reply@nidec-ga.com" \
    --set-secrets="SECRET_SMTP_PASSWORD=airflow-config-smtp-password:latest" \
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
    --set-env-vars="PIPELINE_FUNCTION_URL=${PIPELINE_URL},MAILER_FUNCTION_URL=${MAILER_URL},TRIGGER_PREFIX=${TRIGGER_PREFIX},SETTLE_DELAY_SECONDS=${SETTLE_DELAY_SECONDS}" \
    --quiet

ORCHESTRATOR_URL=$(gcloud run services describe cad-review-orchestrator \
    --region="${REGION}" --project="${PROJECT_ID}" \
    --format="value(status.url)")

echo "Orchestrator URL: ${ORCHESTRATOR_URL}"
echo ""

# ------------------------------------------------------------------------------
# Eventarc trigger: fire the orchestrator when the sentinel marker is uploaded
# ------------------------------------------------------------------------------
# GCS event filters only support 'type' and 'bucket' (no path/prefix filtering),
# so the orchestrator filters in code to react only to PDFs under
# '${TRIGGER_PREFIX}/.../ResultingObjects/'. Two PDFs produce two events; the
# orchestrator's atomic manifest claim ensures only one invocation proceeds.
#
# Requirements (grant once, idempotent):
#   - The GCS service agent needs roles/pubsub.publisher.
#   - ${TRIGGER_SA} needs roles/run.invoker on the orchestrator and
#     roles/eventarc.eventReceiver on the project.
echo "--- Configuring Eventarc trigger prerequisites ---"

GCS_SA="$(gcloud storage service-agent --project="${PROJECT_ID}")"
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${GCS_SA}" \
    --role="roles/pubsub.publisher" \
    --condition=None --quiet >/dev/null

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${TRIGGER_SA}" \
    --role="roles/eventarc.eventReceiver" \
    --condition=None --quiet >/dev/null

gcloud run services add-iam-policy-binding cad-review-orchestrator \
    --region="${REGION}" --project="${PROJECT_ID}" \
    --member="serviceAccount:${TRIGGER_SA}" \
    --role="roles/run.invoker" --quiet >/dev/null

echo "--- Creating Eventarc trigger (idempotent) ---"
# Trigger location must match the bucket location (multi-region 'us'/'eu' or a
# single region). The destination Cloud Run service can live in a different
# region (${REGION}).
if gcloud eventarc triggers describe cad-review-gcs-trigger \
    --location="${TRIGGER_LOCATION}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
    echo "Trigger cad-review-gcs-trigger already exists in ${TRIGGER_LOCATION} — skipping creation."
else
    gcloud eventarc triggers create cad-review-gcs-trigger \
        --location="${TRIGGER_LOCATION}" \
        --project="${PROJECT_ID}" \
        --destination-run-service=cad-review-orchestrator \
        --destination-run-region="${REGION}" \
        --event-filters="type=google.cloud.storage.object.v1.finalized" \
        --event-filters="bucket=${TRIGGER_BUCKET}" \
        --service-account="${TRIGGER_SA}" \
        --quiet
fi

echo ""
echo "=== Deployment Complete ==="
echo "Orchestrator URL: ${ORCHESTRATOR_URL}"
echo "Pipeline URL:     ${PIPELINE_URL}"
echo "Mailer URL:       ${MAILER_URL}"
echo "Trigger:          cad-review-gcs-trigger (location=${TRIGGER_LOCATION}, bucket=${TRIGGER_BUCKET}, scope=${TRIGGER_PREFIX})"
echo ""
echo "NOTE: The orchestrator reacts to PDFs uploaded under"
echo "      '${TRIGGER_PREFIX}/.../ResultingObjects/'. It waits ${SETTLE_DELAY_SECONDS}s for a"
echo "      possible second PDF, then de-duplicates concurrent events via an atomic"
echo "      manifest claim so each CT folder is processed exactly once."
echo ""
echo "NOTE: Update the mailer env vars (SMTP_HOST, MAIL_RECIPIENTS, MAIL_SENDER)"
echo "      with actual values using:"
echo "  gcloud run services update cad-review-mailer --region=${REGION} --update-env-vars=SMTP_HOST=...,MAIL_RECIPIENTS=...,MAIL_SENDER=..."
