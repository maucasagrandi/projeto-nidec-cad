# CAD Review — Streamlit Test Interface

Interface de teste para o pipeline de revisão CAD.

## Funcionalidade

1. Usuário informa o código do Change Task (ex: `CT_219344`)
2. Faz upload de 1 ou 2 PDFs
3. App sobe os PDFs para o GCS no caminho correto
4. Dispara o orquestrador Cloud Run
5. Aguarda o relatório ficar pronto
6. Disponibiliza o PDF do relatório para download

## Execução Local

```bash
# Pré-requisitos: autenticação GCP
gcloud auth login
gcloud auth application-default login

# Instalar dependências
pip install -r requirements.txt

# Rodar
streamlit run app.py
```

## Deploy em outra conta GCP (VM de testes)

```bash
# Na VM de testes, autenticar com suas credenciais pessoais
gcloud auth login
gcloud auth application-default login

# Instalar e rodar
pip install -r requirements.txt
streamlit run app.py --server.port=8080
```

### Com Docker

```bash
docker build -t cad-review-streamlit .
docker run -p 8080:8080 \
  -v ~/.config/gcloud:/root/.config/gcloud:ro \
  -e GOOGLE_APPLICATION_CREDENTIALS=/root/.config/gcloud/application_default_credentials.json \
  cad-review-streamlit
```

## Variáveis de Ambiente

| Variável | Default | Descrição |
|----------|---------|-----------|
| `GCS_BUCKET` | `acim-global-data-lake-sandbox-temp` | Bucket GCS destino |
| `GCS_BASE_PATH` | `temp/Windchill/cadreview` | Prefixo do caminho no bucket |
| `ORCHESTRATOR_URL` | `https://cad-review-orchestrator-...us-central1.run.app` | URL do orquestrador |

## Autenticação Cross-Account

A app usa Application Default Credentials (ADC). Como está em outra conta GCP:

1. O usuário precisa ter `gcloud auth application-default login` feito com uma conta que tenha:
   - Permissão de escrita no bucket `acim-global-data-lake-sandbox-temp`
   - Permissão de invocar o Cloud Run (`roles/run.invoker`) no projeto `acim-global-data-lake-sandbox`

2. As credenciais ADC são montadas na VM/container e o app as usa transparentemente.
