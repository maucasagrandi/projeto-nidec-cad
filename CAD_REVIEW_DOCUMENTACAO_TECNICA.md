# CAD Review — Documentação Técnica

> Módulo de comparação automática entre duas revisões de um desenho técnico CAD (PDF), com detecção visual de diferenças e análise de engenharia via IA generativa (Gemini / GCP Vertex AI).

Este documento explica **como cada parte do pipeline funciona**, na ordem em que o dado passa por elas. A ideia é servir de roteiro para uma apresentação: cada seção abaixo pode virar um slide (ou bloco de slides), com o código-fonte e as decisões de projeto já justificadas.

---

## 1. Visão geral do fluxo

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌───────────────────┐
│  Upload dos  │ --> │  Rasterização │ --> │  Pré-filtro de    │ --> │  Análise por IA   │ --> │  Anotação visual + │
│  2 PDFs      │     │  (PDF→imagem) │     │  páginas mudadas  │     │  (Gemini/Vertex)  │     │  relatório em PDF  │
└──────────────┘     └──────────────┘     └──────────────────┘     └──────────────────┘     └───────────────────┘
```

Entrada: dois arquivos PDF (versão **original** e versão **revisada** do mesmo desenho).
Saída: para cada página com diferença, uma tabela técnica (Item / Diferença / Localização / Status / Ação Recomendada), a imagem de diff visual, a imagem com quadrantes marcados, e relatórios em PDF para download.

Todo o fluxo está implementado em `front.py` (interface Streamlit) e se apoia em quatro módulos utilitários:

| Módulo | Responsabilidade |
|---|---|
| `src/utils/helper_func.py` | Conversão PDF → imagem, cálculo do diff visual (OpenCV), compressão de imagem |
| `src/modeling/llm_models.py` | Chamada ao modelo Gemini via Vertex AI (`compare_cad_pages`) |
| `src/utils/cad_quadrant_paint.py` | Extração da grade de zoneamento do PDF e pintura dos quadrantes reportados pela IA |
| `src/utils/cost_logger.py` | Registro de tokens, latência e custo estimado de cada chamada ao LLM |

---

## 2. Upload e preview

**Onde:** `front.py`, seção `CAD Review Mode`.

O usuário sobe dois arquivos PDF:

- **Original** — revisão anterior do desenho.
- **Revisado** — revisão atual.

```python
pdf1 = st.file_uploader("Upload original PDF", type=["pdf"], key="pdf1")
pdf2 = st.file_uploader("Upload revised PDF", type=["pdf"], key="pdf2")
```

Ao carregar qualquer um dos dois, a primeira página de cada é renderizada em baixa resolução (100 DPI) só para o usuário confirmar visualmente que subiu o arquivo certo antes de disparar o processamento pesado. Essa é uma escolha deliberada de UX: **preview rápido e barato antes do processamento caro**.

---

## 3. Rasterização dos PDFs

**Onde:** `src/utils/helper_func.py` → `pdf_to_pil_images()` e `pdf_to_images_base64()`.

PDFs são documentos vetoriais; para comparar visualmente e para enviar ao modelo de visão, cada página é convertida em imagem (raster) usando **PyMuPDF (fitz)**:

```python
matrix = fitz.Matrix(dpi / 72, dpi / 72)   # 72 pt = 1 polegada (unidade nativa do PDF)
pix = page.get_pixmap(matrix=matrix)
```

O sistema rasteriza **três vezes, em três resoluções diferentes**, cada uma otimizada para um uso:

| Resolução | Uso | Por quê |
|---|---|---|
| 200 DPI | Envio ao LLM (`pages*_b64`) | Suficiente para o modelo ler texto e símbolos; mantém o payload menor |
| 300 DPI | Diff visual e exibição na tela (`pages*_pil`) | Nitidez necessária para detectar mudanças pequenas via OpenCV e para o usuário conseguir dar zoom |
| 150 DPI | Blocos de detalhe "por ID" no relatório PDF (`pages*_pil_150`) | Resolução intermediária: legível, mas evita gerar um PDF de relatório gigante |

Esse escalonamento de resolução por finalidade é uma otimização direta de custo (tokens) e desempenho (tamanho de arquivo).

---

## 4. Otimização de imagem antes do envio ao LLM

**Onde:** `src/utils/helper_func.py` → `compress_png_for_llm()`.

Antes de enviar as imagens ao Gemini, elas passam por uma compressão PNG sem perda de conteúdo relevante:

```python
img.save(buffered, format="PNG", optimize=True, compress_level=9)
```

Isso remove metadados e aplica compressão máxima do PNG (que é sem perdas — não afeta a leitura de texto/símbolos pela IA), reduzindo o tamanho do payload em ~30–40%. Menos bytes de imagem → menos tokens de entrada → menor custo e latência por chamada.

---

## 5. Pré-filtro: quais páginas realmente mudaram?

**Onde:** `src/utils/helper_func.py` → `count_diff_regions()`.

Antes de gastar uma chamada de LLM (a parte mais cara e lenta do pipeline), o sistema decide **quais páginas vale a pena analisar**, usando visão computacional pura (OpenCV), sem IA:

1. Converte as duas imagens da mesma página (original vs. revisada) para escala de cinza.
2. Calcula a diferença absoluta pixel a pixel (`cv2.absdiff`).
3. Aplica um leve `GaussianBlur` para ignorar ruído de renderização (anti-aliasing, jpeg artifacts).
4. Binariza por threshold (`cv2.threshold`) — só sobra o que realmente mudou.
5. Fecha pequenos buracos com morfologia (`MORPH_CLOSE` + `dilate`), sem juntar regiões distantes.
6. Conta os contornos resultantes com área mínima (> 30px) — esse número é o `n_regions`.

Página com `n_regions == 0` é descartada: nenhuma diferença visual relevante, não vale a pena mandar para o LLM. Isso economiza chamadas em documentos grandes (multi-página) onde só uma minoria das páginas de fato mudou.

> **Ponto de apresentação:** esse é o motivo pelo qual o sistema escala bem — o custo de IA cresce só com o número de páginas *que efetivamente mudaram*, não com o total de páginas do PDF.

---

## 6. Diff visual (a imagem "rosa" de sobreposição)

**Onde:** `src/utils/helper_func.py` → `compute_visual_diff()`.

Para as páginas que passaram no pré-filtro, o sistema gera uma imagem de "raio-X" da mudança:

1. Mesmo pipeline de diff do passo anterior (absdiff → blur → threshold → morfologia), mas com kernel um pouco mais permissivo para desenhar retângulos mais "generosos" ao redor da mudança (padding proporcional ao tamanho da imagem).
2. Para cada contorno relevante, desenha um **retângulo preenchido com overlay rosa semi-transparente (alpha 0.6)** sobre a imagem revisada — sem bordas vermelhas duras, para não poluir visualmente o desenho técnico.
3. O resultado é devolvido como uma nova imagem PIL, que é uma das colunas exibidas na tela (coluna "Differences") e também disponível para download em PDF (`⬇️ Download Diff (PDF)`).

Essa imagem é **puramente geométrica** — não tem nenhuma informação semântica (ainda não sabe se a diferença é "normal" ou "problema"). Essa camada de julgamento vem no próximo passo.

---

## 7. Análise semântica pela IA (Gemini via Vertex AI)

**Onde:** `src/modeling/llm_models.py` → `compare_cad_pages()`. Prompt em `prompts.py` → `system_prompt`.

Esta é a etapa central de valor do produto. As duas imagens da página (original + revisada, em 200 DPI) são enviadas ao modelo **Gemini** através do SDK `google.genai`, configurado para usar a infraestrutura **GCP Vertex AI**:

```python
client = genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)

response = client.models.generate_content(
    model=model,
    contents=[
        types.Part.from_bytes(data=image1_data, mime_type="image/png"),
        types.Part.from_bytes(data=image2_data, mime_type="image/png"),
        types.Part.from_text(text=f"{system_prompt}\n\nPrimeira: ORIGINAL. Segunda: REVISADA."),
    ],
)
```

### 7.1 O que o prompt pede exatamente

O `system_prompt` instrui o modelo a agir como um **especialista sênior em análise de documentos CAD** e reportar cada diferença encontrada segundo quatro critérios fixos:

| Coluna | Conteúdo |
|---|---|
| **Diferença Encontrada** | Descrição concisa da mudança (múltiplos pontos separados por `;`) |
| **Localização (Quadrante)** | Onde a mudança ocorre, referenciando o sistema de coordenadas do próprio desenho (ex: `D4 a E7`, `A1, B1-C3`) |
| **Status IA** | Classificação de risco em 3 níveis fixos (ver abaixo) |
| **Ação Recomendada** | O que fazer a respeito |

### 7.2 Os três níveis de status (o coração da lógica de negócio)

- 🟢 **Aprovado** — mudança correta, intencional, sem dúvida técnica.
- 🟡 **Aprovado com Observação** — mudança provavelmente intencional, mas com algo que merece checagem humana (ex: troca de norma referenciada, novo requisito de processo, alteração de símbolo de identificação).
- 🔴 **Requer Correção** — erro claro, inconsistência técnica, ou omissão.

Essa taxonomia de 3 níveis é o que transforma "detectar uma diferença" em "dar um parecer de engenharia" — é a diferença entre um diff genérico de imagem e uma ferramenta de revisão técnica.

### 7.3 Regras de sensibilidade máxima

O prompt reforça explicitamente que o modelo deve reportar **toda** diferença, por menor que seja: valores numéricos (cotas, tolerâncias, ângulos), textos que aparecem/desaparecem (inclusive dentro de tabelas de variantes), troca/adição/remoção de símbolos de GD&T (nomeando o símbolo, nunca de forma genérica), geometrias e notas técnicas alteradas. Essa exigência de "nomear o símbolo" (ex: "símbolo de cilindricidade ⌭ substituído por símbolo de circularidade ○") existe porque um revisor humano precisa saber *exatamente* o que trocou, não só que "algo" trocou.

### 7.4 Formato de saída

O modelo responde em **Markdown puro**, com uma tabela de 5 colunas (`Item | Diferença Encontrada | Localização (Quadrante) | Status IA | Ação Recomendada`). Não é pedido JSON estruturado nesta etapa — a escolha por Markdown facilita a renderização direta na tela (`st.markdown`) e o parsing posterior para o PDF e para a pintura de quadrantes (próximas seções).

### 7.5 Metadados de uso

Cada chamada retorna `usage_metadata` (tokens de entrada, saída, total) e a latência é medida no código (`time.time()` antes/depois). Isso alimenta o `CostLogger` (seção 10).

---

## 8. Localização automática das mudanças no desenho (quadrantes)

**Onde:** `src/utils/cad_quadrant_paint.py`.

Esta é a parte mais sofisticada do pipeline e resolve um problema específico: **a IA já disse *onde* está a mudança em texto livre** (ex: `"Central (D4 a E7)"`), mas isso precisa ser transformado em um **retângulo desenhado sobre a imagem**. Importante: **nenhuma chamada adicional de IA é feita aqui** — é 100% determinístico, reaproveitando o texto que o modelo já produziu na tabela.

O processo tem duas etapas independentes:

### 8.1 Extração da grade de zoneamento (`extract_grid`)

Desenhos técnicos de engenharia normalmente têm, nas bordas da folha, uma grade de referência: números ao longo do topo/base, letras ao longo das laterais (como um mapa geográfico). O sistema lê essa grade **direto do PDF vetorial** (não da imagem rasterizada) usando PyMuPDF:

1. Varre os blocos de texto perto das quatro bordas da página (`EDGE_FRACTION = 13%` da largura/altura).
2. Candidatos numéricos próximos às bordas horizontais → candidatos a **coluna**; candidatos alfabéticos próximos às bordas verticais → candidatos a **linha**.
3. Como o texto extraído pode conter ruído (outros números/letras que não são da grade), o algoritmo (`_best_progression`) busca o **maior subconjunto de rótulos igualmente espaçados** — ou seja, ele valida que "1, 2, 3, 4..." realmente formam uma progressão aritmética antes de aceitar como grade real.
4. O resultado (`GridInfo`) mapeia cada rótulo (`"D"`, `"4"`) para uma coordenada real em pontos PDF, junto com o espaçamento (`column_step`, `row_step`) entre células.

Se a página não tiver uma grade detectável com confiança (menos de 3 rótulos consistentes), a função retorna `None` — e o pipeline simplesmente não pinta quadrantes para essa página (falha de forma segura, sem quebrar o restante da análise).

### 8.2 Parsing do texto livre de localização (`parse_quadrant_text`)

O texto que a IA escreve na coluna "Localização" é linguagem natural com alguma estrutura, não um formato rígido. O parser usa expressões regulares para reconhecer três padrões:

- **Célula única**: `"A1"` ou `"1A"` (ordem tanto letra-número quanto número-letra).
- **Intervalo**: `"D4 a E7"` ou `"D4-E7"` (conectado por `"-"` ou pela palavra `"a"`).
- **Lista**: múltiplas células separadas por vírgula, cada uma tratada como grupo independente.

Textos sem nenhum token reconhecível (ex: `"Central / Vistas de Detalhe"`) simplesmente não geram nenhum grupo — sem erro, apenas sem retângulo.

### 8.3 Pintura sobre a imagem (`paint_quadrants`)

Com a grade e os grupos de célula resolvidos, cada retângulo é convertido de pontos PDF para pixels na resolução de rasterização (`bbox_pt_to_px`), e desenhado sobre a imagem revisada:

- Preenchimento semi-transparente (alpha ~70/255) recortado **apenas na região do próprio retângulo** — não é alocado um overlay do tamanho da página inteira, o que evitava `MemoryError` em folhas grandes (A0/A1) rasterizadas em alta resolução.
- Borda colorida sólida ao redor do retângulo.
- Número do item centralizado dentro do quadrante, com contorno branco fino para se destacar sobre as linhas do desenho técnico.
- A cor é escolhida por índice do item em uma paleta fixa de 6 cores distintas (`_CORES`), garantindo que itens próximos no desenho não fiquem visualmente confundidos.

Cada `PaintedRegion` guarda se a resolução teve sucesso (`resolvido: bool`) — usado na tela para avisar o usuário quando "N itens não puderam ser localizados na grade" (texto de localização sem quadrante identificável).

### 8.4 Pintura individual por item (`paint_single_item`)

Variante de `paint_quadrants` para desenhar **um único item por vez** — usada na seção "Detalhamento por ID" do relatório PDF, onde cada linha da tabela ganha um par de imagens (original + revisada) com só aquele item destacado, facilitando a leitura ponto a ponto.

---

## 9. Exibição dos resultados na interface

**Onde:** `front.py`, bloco `Display of results`.

Para cada página analisada, a tela mostra:

1. **Grade de imagens** (3 ou 4 colunas, com zoom): Original | Revisado | Diferenças | *(Revisado com Quadrantes, se a grade foi detectada)*.
2. **Botões de download**: Diff em PDF, Relatório de IA em PDF, Revisado com Quadrantes em PDF.
3. **Métricas da chamada de IA**: Input Tokens, Output Tokens, Total Tokens, Latência.
4. **Tabela de divergências** renderizada como Markdown (`st.markdown`), diretamente a partir da resposta do modelo.

---

## 10. Geração do relatório técnico em PDF

**Onde:** `front.py`, dentro do botão "Download AI Report (PDF)", usando **ReportLab**.

O texto Markdown devolvido pelo modelo é reprocessado para gerar um documento PDF formal, formatado em A4 paisagem, com:

- Título e cabeçalho institucional (`author="CAD Review - Nidec"`).
- **Parsing manual da tabela Markdown** (linhas que começam e terminam com `|`, descartando a linha separadora de `---`), reconstruída como uma `Table` do ReportLab.
- **Coloração condicional da coluna Status IA**: verde para "Aprovado", âmbar para "Aprovado com Observação", vermelho para "Requer Correção" — tanto no texto quanto no fundo da célula.
- **Bullet points automáticos**: células com `;` são quebradas em lista com marcador `•`.
- **Seção "Details by ID"**: para cada linha da tabela, um bloco dedicado com a descrição da diferença e as duas imagens (original/revisada) anotadas somente com aquele item, lado a lado — usando exatamente as funções de pintura da seção 8.

Esse relatório é o artefato final que pode ser compartilhado com um engenheiro que não tem acesso à ferramenta — a "prova documental" da análise.

---

## 11. Rastreamento de custo e performance

**Onde:** `src/utils/cost_logger.py`.

Toda chamada ao LLM é registrada em `custos.csv` (timestamp, modelo, tokens de entrada/saída/total, latência, custo estimado em USD). O custo é calculado com uma tabela de preço por milhão de tokens (parametrizável conforme o modelo/região usados no Vertex AI). Esse log persiste entre sessões e serve tanto para acompanhamento operacional (quanto a ferramenta está custando) quanto para benchmarking de latência.

---

## 12. Resumo dos módulos e onde encontrar cada trecho de código

| Etapa do pipeline | Arquivo | Função/Trecho |
|---|---|---|
| Upload e preview | `front.py` | seção `PDF Upload` |
| Rasterização PDF→imagem | `src/utils/helper_func.py` | `pdf_to_pil_images`, `pdf_to_images_base64` |
| Compressão para LLM | `src/utils/helper_func.py` | `compress_png_for_llm` |
| Pré-filtro de páginas mudadas | `src/utils/helper_func.py` | `count_diff_regions` |
| Diff visual (overlay rosa) | `src/utils/helper_func.py` | `compute_visual_diff` |
| Chamada ao Gemini/Vertex AI | `src/modeling/llm_models.py` | `compare_cad_pages` |
| Prompt de instrução ao modelo | `prompts.py` | `system_prompt` |
| Extração da grade do PDF | `src/utils/cad_quadrant_paint.py` | `extract_grid`, `extract_grid_from_page` |
| Parsing do texto de localização | `src/utils/cad_quadrant_paint.py` | `parse_quadrant_text` |
| Pintura dos quadrantes | `src/utils/cad_quadrant_paint.py` | `paint_quadrants`, `paint_single_item` |
| Parsing da tabela Markdown | `src/utils/cad_quadrant_paint.py` | `parse_markdown_table`, `encontrar_coluna` |
| Geração do relatório PDF | `front.py` | bloco `Download AI Report (PDF)` (ReportLab) |
| Log de custo/latência | `src/utils/cost_logger.py` | `CostLogger.log_analysis`, `get_summary` |

---

## 13. Pontos de destaque para a apresentação

Sugestões de "mensagens-chave" para cada etapa, caso queira montar slides a partir deste documento:

1. **Pipeline em camadas** — visão computacional barata filtra o que vale a pena mandar para IA cara. Isso é uma decisão de custo/arquitetura, não só técnica.
2. **A IA não substitui o revisor, ela prioriza o trabalho dele** — os 3 níveis de status (Aprovado / Observação / Correção) transformam a saída em uma fila de prioridade de revisão humana, não em uma decisão automática e cega.
3. **Rastreabilidade visual** — a etapa de quadrantes (seção 8) é 100% determinística e reaproveita a saída da IA sem custo adicional, transformando texto solto ("Central, D4 a E7") em um retângulo desenhado exatamente no lugar certo do desenho.
4. **Saída acionável, não só um relatório** — cada divergência gera um par de imagens anotadas (original + revisada) e vira uma linha de PDF formal, pronta para anexar a um processo de engenharia.
5. **Controle de custo desde o design** — compressão de imagem, resolução escalonada por finalidade, e pré-filtro de páginas mudadas existem especificamente para reduzir tokens/latência antes que a chamada de IA aconteça.
