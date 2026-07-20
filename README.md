# CAD Comparison with Gemini

Aplicação Streamlit para comparar desenhos CAD (PDF) usando IA Gemini via Google Cloud Vertex AI.

## O que faz

- 📄 Upload de dois PDFs (versão original e revisada)
- 🔍 Análise de divergências via Gemini 2.5 Flash Image
- 📊 Rastreamento de tokens, latência e custos (salvo em `custos.csv`)
- 💰 Cálculo automático de custo por análise

## Setup

```bash
# Instalar dependências
pip install -r requirements.txt

# Configurar .env
cp .env.example .env
# Adicione suas credenciais GCP_PROJECT_ID e GCP_REGION
```

## Executar

```bash
streamlit run front.py
```

## Estrutura

```
.
├── front.py                      # App principal Streamlit
├── prompts.py                    # Prompt do Gemini (JSON)
├── custos.csv                    # Log de custos (gerado automaticamente)
├── src/
│   ├── modeling/
│   │   └── llm_models.py        # Cliente Vertex AI + compare_cad_pages
│   └── utils/
│       ├── cost_logger.py       # Logger CSV com cálculo de custos
│       └── helper_func.py       # Utilitários PDF/imagem
└── requirements.txt              # Dependências
```

## Variáveis de Ambiente

- `GCP_PROJECT_ID` - ID do projeto GCP
- `GCP_REGION` - Região (padrão: us-east5)



