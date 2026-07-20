# Especificação Técnica — CAD Review

## 1. Visão Geral

O **CAD Review** é uma aplicação web que compara duas revisões de desenhos técnicos CAD (em PDF) utilizando visão computacional (OpenCV) e inteligência artificial generativa (Google Gemini via Vertex AI). O sistema identifica diferenças visuais entre revisões e gera relatórios técnicos de divergências.

**Stack principal:** Python 3.10+ · Streamlit · Google Gemini (Vertex AI) · OpenCV · PyMuPDF · Pillow

---

## 2. Estrutura do Projeto

```
nidec-cad-review/
├── front.py                    # Interface Streamlit (entrada principal)
├── prompts.py                  # Prompt de sistema para o LLM
├── logo.png                    # Logo exibido na sidebar
├── custos.csv                  # Log de custos/tokens (gerado em runtime)
├── .env                        # Variáveis de ambiente (não versionado)
├── .env.example                # Template de variáveis de ambiente
├── pyproject.toml              # Metadados do projeto e dependências
├── requirements.txt            # Dependências (formato pip)
├── testes.ipynb                # Notebook de testes exploratórios
├── cad_docs_examples/          # PDFs de exemplo para testes manuais
├── CAD_Review_Test_Battery_V1/ # Bateria de testes (single + comparison)
└── src/
    ├── __init__.py
    ├── modeling/
    │   ├── __init__.py
    │   └── llm_models.py       # Cliente Gemini + função de análise
    └── utils/
        ├── __init__.py
        ├── helper_func.py      # Conversão PDF→imagem, diff visual, compressão
        └── cost_logger.py      # Logger de custos e tokens em CSV
```

---

## 3. Fluxo de Processamento

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌──────────────┐
│  Upload PDF │────▶│ Rasterização     │────▶│ Pré-filtragem   │────▶│ Análise LLM  │
│  (2 arquivos)│     │ (PyMuPDF)        │     │ (OpenCV)        │     │ (Gemini)     │
└─────────────┘     └──────────────────┘     └─────────────────┘     └──────────────┘
                           │                        │                        │
                           ▼                        ▼                        ▼
                    200 DPI → LLM            count_diff_regions()     Relatório Markdown
                    300 DPI → Diff visual    Filtra páginas iguais    Tabela de diferenças
```

### 3.1 Etapas detalhadas

| Etapa | Descrição | DPI | Formato |
|-------|-----------|-----|---------|
| 1. Upload | Usuário envia 2 PDFs (original + revisado) | — | PDF |
| 2. Preview | Renderiza 1ª página para preview | 100 | PIL Image |
| 3. Rasterização LLM | Converte todas as páginas para envio ao modelo | 200 | PNG base64 |
| 4. Rasterização Diff | Converte todas as páginas para análise visual | 300 | PIL Image |
| 5. Compressão | Otimiza PNGs com `compress_level=9` para reduzir tokens | — | PNG base64 |
| 6. Pré-filtragem | Identifica páginas com diferenças via OpenCV | — | int (regiões) |
| 7. Diff visual | Gera imagem com contornos vermelhos nas diferenças | 300 | PIL Image |
| 8. Análise LLM | Envia par de imagens ao Gemini para relatório | — | Markdown |
| 9. Download | Exporta imagem diff em PDF | 300 | PDF |

---

## 4. Módulos

### 4.1 `front.py` — Interface Web (Streamlit)

Responsabilidades:
- Autenticação por usuário/senha (via `.env`)
- Upload de 2 PDFs com preview da primeira página
- Orquestração do pipeline de comparação
- Exibição side-by-side: Original | Revisado | Diferenças
- Exibição do relatório de divergências do LLM
- Métricas de tokens, latência e custo
- Botão de download do diff em PDF (300 DPI)
- Sumário final com custos acumulados

**Componentes visuais:**
- `streamlit-image-zoom` para zoom interativo nas imagens
- Layout em 3 colunas para comparação visual
- Métricas em cards (`st.metric`)

---

### 4.2 `prompts.py` — Prompt de Sistema

O prompt instrui o Gemini como um especialista em CAD para:
1. Listar **todas** as diferenças entre as revisões
2. Indicar **localização** no desenho (quadrante/região)
3. Classificar o **tipo de mudança** (dimensional, estrutural, anotação, etc.)
4. Avaliar **impacto potencial**

**Formato de saída:** Tabela Markdown com colunas:
`Item | Diferença Encontrada | Localização (Quadrante) | Tipo de Mudança | Impacto Potencial`

**Regras de formatação:** Proibido HTML. Múltiplos itens em uma célula separados por `;`.

---

### 4.3 `src/modeling/llm_models.py` — Cliente LLM

| Componente | Descrição |
|------------|-----------|
| **Modelo** | `gemini-3.5-flash` via Vertex AI |
| **Cliente** | `google.genai.Client` com `vertexai=True` |
| **Região** | Configurável via `GCP_REGION` (padrão: `global`) |
| **Max tokens** | 32.768 |
| **Entrada** | 2 imagens inline (PNG base64) + prompt de texto |
| **Saída** | Texto Markdown + `AnalysisMetadata` (tokens, latência) |

**Função principal:** `compare_cad_pages(image1_base64, image2_base64, system_prompt)`

O conteúdo enviado ao modelo segue esta estrutura:
```
[imagem1_png] [imagem2_png] [system_prompt + instrução contextual]
```

A instrução contextual fixa:
> "A primeira imagem é o desenho CAD ORIGINAL. A segunda imagem é o desenho CAD REVISADO. Identifique e liste todas as divergências entre eles."

---

### 4.4 `src/utils/helper_func.py` — Processamento de Imagem

#### Funções de conversão PDF → Imagem

| Função | Entrada | Saída | Uso |
|--------|---------|-------|-----|
| `pdf_to_images_base64(pdf_bytes, dpi)` | bytes do PDF | `list[str]` (base64) | Envio ao LLM |
| `pdf_to_pil_images(pdf_bytes, dpi)` | bytes do PDF | `list[PIL.Image]` | Diff visual |

Ambas usam **PyMuPDF** (`fitz`) com `fitz.Matrix(dpi/72, dpi/72)` para controlar a resolução de rasterização.

#### Funções de análise visual (OpenCV)

| Função | Descrição |
|--------|-----------|
| `compute_visual_diff(img1, img2)` | Calcula diferença, aplica morfologia moderada, overlay rosa translúcido |
| `count_diff_regions(img1, img2)` | Conta regiões com diferença (usado na pré-filtragem) |

**Pipeline OpenCV (`compute_visual_diff`):**
1. Conversão RGB → BGR
2. Resize para equalizar dimensões (se necessário)
3. `cv2.absdiff()` — diferença absoluta pixel a pixel
4. Conversão para grayscale
5. `GaussianBlur(5,5)` — suaviza ruído de renderização
6. `threshold(15)` — binariza (limiar sensível para CAD)
7. `morphologyEx(MORPH_CLOSE, kernel=7x7, iter=2)` — fecha gaps pequenos sem unir regiões distantes
8. `dilate(kernel=7x7, iter=2)` — expande levemente as detecções
9. `findContours` — identifica regiões conectadas
10. Filtra contornos com `area > 50` (remove artefatos mínimos)
11. **Overlay rosa semi-transparente** (`alpha=0.30`, cor BGR `(200, 210, 255)`) — sem bordas

**Estilo visual:** Fundo rosa translúcido sobre as regiões alteradas (estilo analista humano). Detecção granular e precisa — marca apenas os pontos exatos de diferença sem fundir regiões distantes. Sem retângulos com bordas vermelhas.

#### Funções de otimização

| Função | Descrição |
|--------|-----------|
| `compress_png_for_llm(img)` | PNG com `optimize=True` + `compress_level=9`. Reduz ~30-40% tokens |
| `pil_to_base64(img)` | Conversão simples PIL → base64 |

---

### 4.5 `src/utils/cost_logger.py` — Logger de Custos

Registra cada análise em `custos.csv` com:
- Timestamp, modelo, tokens (prompt/completion/total), latência, custo USD

**Cálculo de custo (Gemini 2.5 Flash):**
- Entrada: $0.075 / 1M tokens
- Saída: $0.30 / 1M tokens

---

## 5. Configuração e Ambiente

### Variáveis de ambiente (`.env`)

| Variável | Descrição | Padrão |
|----------|-----------|--------|
| `GCP_PROJECT_ID` | Projeto GCP | `acim-global-data-lake-sandbox` |
| `GCP_REGION` | Região Vertex AI | `us-east5` |
| `APP_USERNAME` | Login da aplicação | — |
| `APP_PASSWORD` | Senha da aplicação | — |

### Autenticação GCP

Usa **Application Default Credentials (ADC)** via `gcloud auth login`. Permissões necessárias:
- Vertex AI User
- Service Usage Consumer

---

## 6. Dependências

| Pacote | Versão | Propósito |
|--------|--------|-----------|
| `streamlit` | 1.38.0 | Interface web |
| `streamlit-image-zoom` | 0.0.4 | Zoom interativo em imagens |
| `pymupdf` | 1.25.1 | Rasterização de PDF |
| `Pillow` | 10.4.0 | Manipulação de imagens |
| `opencv-python` | ≥4.8.0 | Diff visual (contornos, morfologia) |
| `numpy` | ≥1.24.0 | Arrays para OpenCV |
| `google-genai` | ≥1.0.0 | Cliente Gemini (Vertex AI) |
| `google-auth` | 2.35.0 | Autenticação GCP |
| `google-cloud-aiplatform` | 1.70.0 | SDK Vertex AI |
| `python-dotenv` | 1.0.1 | Carregamento de `.env` |

**Gerenciador de pacotes:** `uv` (com `uv.lock` para reprodutibilidade)

---

## 7. Como Executar

```bash
# 1. Instalar dependências
uv sync

# 2. Configurar ambiente
cp .env.example .env
# Editar .env com credenciais

# 3. Autenticar no GCP
gcloud auth login
gcloud config set project acim-global-data-lake-sandbox

# 4. Rodar a aplicação
streamlit run front.py
```

---

## 8. Decisões Técnicas

| Decisão | Justificativa |
|---------|---------------|
| 200 DPI para LLM | Equilíbrio entre qualidade e consumo de tokens |
| 300 DPI para diff visual | Alta resolução para detecção precisa de diferenças |
| Threshold 15 no OpenCV | Sensível para captar mudanças sutis em desenhos CAD |
| Morfologia moderada (7x7) | Fecha gaps sem fundir regiões distantes — mantém precisão granular |
| Overlay rosa sem bordas | Estilo limpo igual ao analista humano — não obstrui o conteúdo |
| Pré-filtragem por OpenCV | Evita enviar páginas sem alteração ao LLM (economia de tokens/custo) |
| Compressão PNG level 9 | Reduz ~30-40% do payload sem perda visual (lossless) |
| Gemini 3.5 Flash | Modelo multimodal rápido com boa capacidade de análise visual |
| Saída em tabela Markdown | Formato estruturado, legível e renderizável pelo Streamlit |
