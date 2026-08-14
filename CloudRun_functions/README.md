# CloudRun Functions — CAD Review Pipeline

Three Cloud Run functions that orchestrate the CAD PDF review workflow.

## Architecture

```
[Change Request Trigger]
         │
         ▼
┌─────────────────────┐
│   1. Orchestrator   │  ← checks manifest, coordinates workflow
│                     │
│  manifest.txt guard │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│    2. Pipeline      │  ← executes CAD review (OpenCV + LLM)
│                     │
│  GCS in → GCS out   │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│     3. Mailer       │  ← sends report PDF via SMTP
│                     │
│  Secret Manager     │
└─────────────────────┘
```

## Trigger Payload

The orchestrator is called with:

```json
{
    "process_id": "abc-123",
    "base_gcs_path": "gs://bucket/process/abc-123",
    "original_pdf_gcs_path": "gs://bucket/process/abc-123/drawing_rev5.pdf",
    "revised_pdf_gcs_path": "gs://bucket/process/abc-123/drawing_rev6.pdf"
}
```

## Functions

### 1. Orchestrator (`orchestrator/`)

- **Entry point:** `orchestrator`
- **Role:** Idempotency guard + workflow coordinator
- **Flow:**
  1. Check if `manifest.txt` exists at `base_gcs_path` → if yes, stop
  2. Create `manifest.txt` with start timestamp
  3. Call Pipeline function
  4. Call Mailer function
  5. Write execution summary to `manifest.txt`

**Env vars:**
| Variable | Description |
|----------|-------------|
| `PIPELINE_FUNCTION_URL` | URL of the deployed pipeline function |
| `MAILER_FUNCTION_URL` | URL of the deployed mailer function |

### 2. Pipeline (`pipeline/`)

- **Entry point:** `pipeline`
- **Role:** Executes the full CAD review (classification, GD&T, OpenCV comparison, PDF report)
- **Inputs:** Two PDFs from GCS
- **Outputs:** Results uploaded to `{base_gcs_path}/PROCESSING_OUTPUTS/`

**Env vars:**
| Variable | Description |
|----------|-------------|
| `GCP_PROJECT_ID` | Google Cloud project ID |
| `GCP_REGION` | Vertex AI region (e.g. `us-east5`) |

### 3. Mailer (`mailer/`)

- **Entry point:** `mailer`
- **Role:** Downloads report PDF from GCS, sends email with attachment via SMTP

**Env vars:**
| Variable | Description |
|----------|-------------|
| `SMTP_HOST` | SMTP server address |
| `SMTP_PORT` | SMTP port (default: 587) |
| `SECRET_PROJECT_ID` | GCP project for Secret Manager |
| `SECRET_SMTP_USER` | Secret name for SMTP username |
| `SECRET_SMTP_PASSWORD` | Secret name for SMTP password |
| `MAIL_RECIPIENTS` | Comma-separated recipient emails |
| `MAIL_SENDER` | Sender email address |

## Deployment

### Prerequisites

1. GCP project with Cloud Functions, Cloud Storage, Secret Manager, and Vertex AI APIs enabled
2. Secrets created in Secret Manager for SMTP credentials:
   ```bash
   echo -n "smtp_user@domain.com" | gcloud secrets create smtp-user --data-file=-
   echo -n "your-password" | gcloud secrets create smtp-password --data-file=-
   ```
3. Service account with roles:
   - `roles/cloudfunctions.invoker` (for service-to-service calls)
   - `roles/storage.objectAdmin` (for GCS read/write)
   - `roles/secretmanager.secretAccessor` (for mailer)
   - `roles/aiplatform.user` (for Vertex AI / Gemini)

### Deploy

```bash
chmod +x deploy.sh
./deploy.sh <GCP_PROJECT_ID> <GCP_REGION>
```

After deployment, update the mailer's placeholder env vars with actual SMTP/recipient values.

## Local Testing

Each function can be tested locally with `functions-framework`:

```bash
cd orchestrator
pip install -r requirements.txt
functions-framework --target=orchestrator --port=8080

# In another terminal:
curl -X POST http://localhost:8080 \
  -H "Content-Type: application/json" \
  -d '{"process_id":"test-1","base_gcs_path":"gs://my-bucket/test","original_pdf_gcs_path":"gs://my-bucket/test/orig.pdf","revised_pdf_gcs_path":"gs://my-bucket/test/rev.pdf"}'
```

## Pipeline Source Code

The pipeline function imports the project's `src/` package. When deploying, the
source code from `src/`, `prompts.py`, and `assets/` must be available in the
pipeline function's container. Options:

1. **Docker-based deploy:** Build a custom container that includes the full repo
2. **Source bundling:** Copy `src/`, `prompts.py`, and `assets/gdt/templates/` into `pipeline/` before deploying
