# Part Classification — Documentação Técnica

> Módulo de análise individual de um desenho técnico CAD (PDF): classificação da peça, normas citadas, detecção de quadros GD&T (feature control frames) e detecção de datums, com avaliação de conformidade contra ISO 1101 / ISO 5459 como baseline de referência.

Este documento explica **como cada parte do pipeline funciona**, na ordem em que o dado passa por elas — mesmo formato do `CAD_REVIEW_DOCUMENTACAO_TECNICA.md`, para servir de roteiro de apresentação. Cada seção pode virar um bloco de slides, com código-fonte e decisões de projeto já justificadas.

---

## 1. Visão geral do fluxo

```
┌──────────────┐   ┌──────────────┐   ┌──────────────────┐   ┌──────────────────┐   ┌────────────────────┐   ┌───────────────────┐
│  Upload de   │→  │  Extração de │→  │  Classificação    │→  │  Detecção         │→  │  Parsing +          │→  │  Agregação final +  │
│  1 PDF       │   │  texto vetor.│   │  da peça (LLM)    │   │  GD&T + Datums    │   │  avaliação ISO      │   │  evidência visual   │
└──────────────┘   └──────────────┘   └──────────────────┘   └──────────────────┘   └────────────────────┘   └───────────────────┘
```

Entrada: **um** arquivo PDF (a peça a ser analisada, diferente do CAD Review que compara dois).
Saída: classificação da peça com evidência textual, normas citadas no desenho, quadros GD&T detectados com tolerância/datums lidos, datums definidos no desenho, findings de conformidade ISO 1101/ISO 5459, e imagens anotadas (por página) prontas para download.

Todo o fluxo é implementado em `pages/classification.py` (interface Streamlit) e delega o processamento pesado para **um único ponto de entrada determinístico + 1 chamada de LLM**: `process_cad_pdf()`, em `src/cad_review/folder_pipeline.py`.

| Módulo | Responsabilidade |
|---|---|
| `src/cad_review/folder_pipeline.py` | Orquestra o pipeline inteiro: chama classificação, detecção GD&T, parsing, checks ISO, gera artefatos e grava `result.json` |
| `src/cad_review/orchestrator.py` | Chamada ao LLM de classificação + consulta determinística de normas aplicáveis |
| `src/cad_review/compliance_engine.py` | Agrega tudo em um resultado único, gera a lista de `findings` e o `summary` |
| `src/gdt/detector.py` | Detecção geométrica dos quadros GD&T (feature control frames) a partir do PDF vetorial |
| `src/gdt/symbol_classifier.py` | Classificação (ranking) do símbolo da primeira célula por comparação com um catálogo de templates |
| `src/gdt/frame_parser.py` | Leitura estrutural do quadro: tolerância numérica, diâmetro, datums referenciados |
| `src/gdt/datum_feature.py` | Detecção de indicadores de datum (caixa + triângulo preenchido) no desenho |
| `src/gdt/iso1101.py` / `src/gdt/iso1101_reference.py` | Resolução de edição ISO 1101 e avaliação de exigência de datum por característica |
| `src/gdt/datum_consistency.py` | Verifica se cada datum referenciado tem uma definição correspondente (ISO 5459) |
| `src/cad_review/visual_output.py` | Gera as imagens anotadas (GD&T, datums, combinada) por página |
| `src/utils/standards_applicability.py` | Consulta determinística de normas aplicáveis via `Normas.xlsx` |

---

## 2. Upload e preview

**Onde:** `pages/classification.py`.

```python
pdf_file = st.file_uploader("Upload PDF", type=["pdf"], key="pdf_classification")
```

Assim como no CAD Review, a primeira página é renderizada em baixa resolução (100 DPI, via `pdf_to_pil_images`) só para confirmação visual antes do processamento pesado — mesma lógica de "preview barato antes do custo real" descrita no documento do CAD Review.

Antes de disparar a análise, a página valida que os três arquivos de configuração obrigatórios do pipeline existem (`Normas.xlsx`, `assets/gdt/templates/`, `validation/gdt/configs/iso1101_2017_reference_rules.json`). Se algum faltar, a UI mostra um erro explícito em vez de deixar o pipeline falhar com um `FileNotFoundError` genérico no meio do processamento.

---

## 3. Extração de texto vetorial do PDF

**Onde:** `src/modeling/llm_models.py` → `extract_text_from_pdf()`, chamada dentro de `run_part_classification_branch()`.

Diferente do CAD Review (que envia **imagens** ao modelo), a classificação da peça é feita a partir do **texto vetorial** do PDF — extraído diretamente com PyMuPDF (`page.get_text()`), sem rasterização nem OCR. Isso é possível porque a informação que a classificação precisa (nome da peça, material, normas citadas) normalmente está em blocos de texto reais do PDF (bloco de título, notas), não em desenho puramente gráfico.

Extrair texto vetorial em vez de mandar imagem para o LLM é uma escolha deliberada de custo: texto é ordens de magnitude mais barato em tokens do que imagem, e para este problema específico (ler campos de um bloco de título) não há ganho em usar visão computacional do modelo.

---

## 4. Classificação da peça pela IA (LLM)

**Onde:** `src/cad_review/orchestrator.py` → `run_part_classification_branch()`. Chamada real do modelo em `src/modeling/llm_models.py` → `classify_cad_enriched()`. Prompt em `prompts.py` → `classificacao_enriquecida_prompt`.

Esta é a **única chamada de LLM** de todo o pipeline de Part Classification — todo o resto (GD&T, datums, ISO) é 100% determinístico. O texto extraído no passo anterior é injetado no prompt (placeholder `{{texto_extraido}}`) e enviado ao Gemini com **saída estruturada obrigatória** (`response_schema=CadClassificationEnriched`), garantindo que a resposta sempre seja um JSON validado por Pydantic, nunca texto livre.

### 4.1 O que é extraído

| Campo | Descrição |
|---|---|
| `document_type` | Tipo do documento (`product_drawing`, `assembly_drawing`, `process_sheet`, `technical_specification`) |
| `component` | Nome da peça (ex: "Connecting Rod"), lido do campo PART NAME/DESCRIPTION ou título |
| `material_family` | Família do material (ex: `sintered_metal`, `gray_cast_iron`) |
| `compressor_series` | Série do compressor — **somente se houver menção explícita** no texto |
| `cited_standards` | Lista de normas citadas literalmente no texto (com o trecho de evidência) |

Cada campo (exceto `cited_standards`) é retornado como um objeto `{value, evidence, confidence}` — nunca só o valor sozinho. Isso é o mesmo padrão de "responda com evidência, não com afirmação" usado no resto do projeto.

### 4.2 Regras de precisão máxima do prompt

O prompt é explícito em proibir invenção: se um campo não está escrito no texto, o modelo deve retornar `null` em vez de deduzir. A regra mais rígida é sobre `compressor_series` — o prompt reforça três vezes que a série **nunca** deve ser deduzida a partir de código de peça ou outros campos, só aceita menção literal (`"SERIES F"`, `"SÉRIE EG"`). Isso existe porque a série do compressor determina qual conjunto de normas é aplicável (seção 5), então um valor inventado propagaria um erro silencioso pipeline abaixo.

Da mesma forma, códigos de norma devem ser preservados **exatamente como aparecem** no texto (`"TSS002611"`, `"TSS-002611"`, `"TSS 002611"` continuam distintos nesta etapa — a normalização vem depois, na seção 5).

### 4.3 Política temporária de série do compressor

**Onde:** `run_part_classification_branch()`, constante `DEFAULT_COMPRESSOR_SERIES_CONTEXT = "ALL"`.

A série real do compressor deveria vir de um sistema externo (Windchill), que ainda não está integrado. Enquanto isso, o pipeline usa `"ALL"` como contexto de revisão (`review_context.compressor_series`) — deliberadamente **separado** do campo `compressor_series` que o LLM extraiu do próprio CAD (que continua sendo o que foi lido do desenho, normalmente `None`). Isso evita que o código confunda "não sei a série" com "a série é ALL".

---

## 5. Normas: aplicabilidade determinística e comparação

**Onde:** `src/utils/standards_applicability.py` → `StandardsApplicabilityEngine`, `compare_standards()`. Orquestrado em `run_part_classification_branch()`.

Depois do LLM extrair `component` e as normas citadas, uma segunda etapa **totalmente determinística** (sem LLM) consulta a planilha `Normas.xlsx` para descobrir quais normas *deveriam* estar no desenho:

1. **Match fuzzy do componente** (similaridade de Jaccard sobre tokens normalizados) contra a aba `Parts` → normas obrigatórias para aquele tipo de peça.
2. **Normas da aba `Notes`** cuja `Applicability` contém o componente (ou é `"All"`) e cuja `Compressor_Series` é compatível com o contexto atual.
3. **Match por família de material** (se fornecida) contra normas de categoria `Material`.

O resultado é comparado deterministicamente com as normas citadas pelo LLM via `compare_standards()`:

```python
matching   = expected ∩ cited
missing    = expected - cited
unexpected = cited - expected
```

> **Nota sobre a política temporária "ALL series":** como a série real ainda não é conhecida, o pipeline mantém apenas as linhas de aplicabilidade originadas do caminho `component_match` (aba `Notes`, que já filtra por série) e **descarta** as linhas agregadas da aba `Parts` (que historicamente misturam normas de todas as séries em uma lista única por peça) — evitando declarar uma norma como aplicável quando na verdade ela só vale para uma série diferente da real.

Este bloco de comparação (`standards_comparison`, `applicable_standards`) continua sendo calculado e persistido no `result.json` mesmo que a interface atual do Streamlit não o exiba como aba dedicada — ele alimenta os *findings* de domínio `"standards"` (seção 8) e fica disponível na aba **Full JSON**.

---

## 6. Detecção de datums definidos no desenho

**Onde:** `src/gdt/datum_feature.py` → `detect_datum_feature_indicators()`. Chamada em `process_cad_pdf()` **antes** da detecção de GD&T, em uma primeira passada sobre todas as páginas.

Um datum é considerado "definido" no desenho apenas quando três sinais independentes coincidem — uma letra isolada (ex: só o texto `"A"`) **não** é suficiente:

1. um token de texto PDF de exatamente uma letra maiúscula (`^[A-Z]$`);
2. um contorno retangular pequeno e quase-quadrado envolvendo esse texto (detectado via OpenCV sobre a página rasterizada em binário);
3. um marcador triangular preenchido nas proximidades, conectado à caixa por um "corredor" de tinta contínuo (`stem_coverage`) — o traço que liga o balão do datum à seta triangular.

```python
def detect_datum_feature_indicators(
    pdf_bytes: bytes, *, page_index: int = 0, raster_dpi: int = 200,
    min_box_size_pt: float = 7.0, max_box_size_pt: float = 24.0, ...
) -> list[DatumFeatureIndicatorCandidate]:
```

Essa exigência de 3 sinais simultâneos existe para não confundir uma letra qualquer do desenho (cota, nota, revisão) com um datum de verdade. Cada candidato aceito guarda `marker_side` (de qual lado da caixa está o marcador), `stem_coverage` e `box_rectangularity` — métricas de confiança geométrica, não uma probabilidade calibrada.

A busca roda **em todas as páginas antes de avaliar qualquer referência**, porque um datum referenciado num quadro GD&T da página 1 pode estar definido na página 2 (ex: vista auxiliar em outra folha) — avaliar por página isoladamente perderia esse caso.

---

## 7. Detecção geométrica dos quadros GD&T

**Onde:** `src/gdt/detector.py` (subclasse) e `src/utils/gdt_detector.py` (implementação-base). Chamada em `process_cad_pdf()` via `GdtFrameDetector().detect_frames()`.

Um "feature control frame" (o quadro retangular de GD&T, com o símbolo na primeira célula seguido de tolerância e datums) é reconstruído **a partir dos segmentos de linha vetoriais do PDF** — não da imagem rasterizada. O detector varre os segmentos horizontais/verticais da página e agrupa os que formam um retângulo subdividido em células, dentro de faixas de tamanho/proporção plausíveis para um FCF.

A classe em `src/gdt/detector.py` reaproveita o `frame_bbox` (o retângulo externo) exatamente como veio da implementação legada em `src/utils/gdt_detector.py`, mas resegmenta as **divisórias internas de célula** com tolerância mais estrita — isso evita que o próprio traço do símbolo (ex: a barra vertical do símbolo de Position ⌖) seja confundido com uma linha divisória de célula.

Cada candidato (`GdtFrameCandidate`) tem um `candidate_id` explícito no formato `GDT-CAND-P<página>-<sequencial>` — o prefixo `CAND` não é decorativo: reforça em toda a cadeia (UI, relatório, CSV de diagnóstico) que **é um candidato do detector, não um GD&T validado por um humano**, até que alguém confirme.

---

## 8. Classificação do símbolo (ranking contra catálogo de templates)

**Onde:** `src/gdt/symbol_classifier.py` → `load_template_catalog()`, `render_page_gray()`, `score_candidates()`.

Depois de detectar a geometria do quadro, o sistema tenta identificar **qual símbolo GD&T está na primeira célula** (paralelismo, posição, planicidade etc.) comparando o recorte daquela célula contra um catálogo de imagens de referência em `assets/gdt/templates/<classe>/`.

Isso é **puramente determinístico e sem LLM** — nem chamada de IA de texto, nem de visão. O algoritmo:

1. Recorta o interior da célula do símbolo (`crop_cell_interior`), removendo a borda para não comparar linha contra linha.
2. Normaliza contraste/polaridade e gera três representações (`gray`, `binary`, `edges`) do recorte.
3. Compara contra cada template usando correlação de template (`cv2.matchTemplate`) **e** dois descritores de forma global: um descritor estrutural de ocupação/projeção e um HOG (histograma de gradientes orientados) numa grade 3×3.
4. Combina os cinco escores em duas famílias — aparência local (40%) e forma global (60%) — evitando que três variantes da mesma evidência local (`gray`/`binary`/`edges`) sejam contadas como três votos independentes.
5. Retorna a classe de melhor escore, a segunda melhor, e a margem entre elas (`CandidateSymbolScore`).

> **Isto é ranking, não classificação com threshold.** Não existe um corte de aceitação calibrado ("se score > X, é esse símbolo"). O resultado é sempre "dado este catálogo, esta é a melhor correspondência e por quanto ela venceu a segunda" — a decisão final de aceitar ou não fica com quem lê o resultado (revisor humano ou regra downstream).

Se o catálogo de templates estiver incompleto (`symbol_catalog.complete == False`) e `allow_incomplete_symbol_catalog=False` (padrão), essa etapa é **desabilitada por completo** para o run — nenhum candidato recebe classificação de característica, de forma fail-closed (silenciosa, sem crash, mas explícita no resultado).

---

## 9. Parsing estrutural do quadro GD&T

**Onde:** `src/gdt/frame_parser.py` → `parse_feature_control_frame()`.

Com a geometria e (quando disponível) a característica classificada, esta etapa lê o **conteúdo** de cada célula do quadro, texto por texto extraído do PDF vetorial — sem inferência visual nesta versão do pipeline (o parâmetro `visual_evidence` existe na assinatura para fallback visual, mas `process_cad_pdf()` não o preenche hoje):

- **Célula de tolerância** (índice 1): extrai o primeiro número decimal reconhecível (`_extract_first_number`), aceitando tanto `,` quanto `.` como separador decimal, e detecta se há símbolo de diâmetro (`⌀`, `Ø`, `∅`) textualmente presente.
- **Células de datum** (índice 2+): uma célula é aceita como datum **somente se contiver exatamente um único token que seja uma letra maiúscula isolada** (`_extract_structural_datum`) — qualquer outra coisa na célula (múltiplos tokens, número junto, modificador) faz o conteúdo cair em `unresolved_tokens` em vez de ser adivinhado.

O resultado (`ParsedGdtFrame`) documenta explicitamente **de onde** cada campo veio (`field_sources`) e o que **não** foi resolvido (`unresolved_fields`, `unresolved_tokens`) — não existe um campo "tolerance_value" preenchido com um palpite; se o texto não permite extrair um número, o campo fica `None` e a lacuna aparece na lista de não resolvidos.

---

## 10. Avaliação ISO 1101 (exigência de datum por característica)

**Onde:** `src/gdt/iso1101.py` (regras/edição) + `src/gdt/iso1101_reference.py` (finding voltado ao usuário). Chamada em `process_cad_pdf()` via `assess_iso1101_datum_rule()`.

Este é o núcleo da lógica de "conformidade" para GD&T: cada característica geométrica (paralelismo, planicidade, posição etc.) tem uma exigência de datum diferente segundo a ISO 1101:2017 — algumas **exigem** datum de referência, outras **não usam**, outras são **condicionais** (depende do contexto de projeto, não pode ser decidido só pela presença/ausência).

A tabela de regras vem de `validation/gdt/configs/iso1101_2017_reference_rules.json` e é citada literalmente no finding (`source_ref`, ex: `"ISO 1101:2017 Table 1, subclause 18.9"`) — nenhuma regra é hard-coded no código Python; tudo vem de configuração externa e rastreável.

### 10.1 Os quatro resultados possíveis

| `datum_requirement` da regra | Situação encontrada | Resultado |
|---|---|---|
| `required` | Sem datum referenciado | 🟡 `WARNING` — "Potential violation": exige datum e não tem |
| `required` / `none` | Presença/ausência compatível com a regra | 🟢 `PASS` |
| `none` | Datum referenciado mesmo assim | 🟡 `WARNING` — datum presente onde a regra diz que não deveria |
| `conditional` | Qualquer situação | 🔎 `NEEDS_CONTEXT` — presença/ausência de datum por si só não decide nada; precisa de contexto de projeto |

### 10.2 Modo `reference` vs `normative`

O finding é gerado com `mode="reference"` no pipeline atual — ou seja, a redação é sempre **"Potential violation of ISO 1101:2017"**, nunca uma afirmação categórica de não-conformidade (`normative_claim=False`). Isso é intencional: a ferramenta usa a ISO 1101:2017 como **baseline técnico de referência** para levantar pontos de atenção, não como prova de que aquela edição da norma é contratualmente aplicável ao desenho — essa determinação normativa (edição exata, aplicabilidade real) fica fora do escopo desta versão do pipeline (ver `src/gdt/iso1101.py::resolve_iso1101_edition`, que existe justamente para nunca *assumir* uma edição sem citação explícita ou regra de aplicabilidade fornecida).

---

## 11. Avaliação ISO 5459 (consistência de definição de datum)

**Onde:** `src/gdt/datum_consistency.py` → `assess_referenced_datum_definitions()`. Chamada em `process_cad_pdf()` para cada quadro GD&T processado.

Enquanto a seção 10 pergunta "esta característica deveria ter datum?", esta etapa pergunta "o datum que o quadro referencia **existe de fato** no desenho?" — cruzando `referenced_datums` (extraído no parsing, seção 9) contra `datum_definitions` (detectado na seção 6):

```python
if datum in definitions:
    status = "PASS"       # ISO5459_DATUM_DEFINITION_FOUND
else:
    status = "WARNING"     # ISO5459_REFERENCED_DATUM_NOT_DEFINED — "Potential violation of ISO 5459"
```

Cada datum é avaliado **uma única vez por quadro** mesmo que apareça repetido (`seen: set[str]`), e tokens que não são uma única letra maiúscula são silenciosamente ignorados (proteção contra ruído de parsing anterior). Assim como na seção 10, o modo padrão é `reference` — a redação nunca afirma categoricamente uma violação, apenas sinaliza o ponto para revisão humana.

---

## 12. Agregação final: findings e resumo

**Onde:** `src/cad_review/compliance_engine.py` → `build_cad_review_result()`. Contratos em `src/cad_review/types.py`.

Esta etapa **não chama LLM e não reinterpreta nada** — ela só normaliza os resultados das seções 5, 10 e 11 em uma lista única de `CadReviewFinding`, cada um com:

```python
{
  "finding_id": "F-001",
  "domain": "standards" | "iso1101" | "iso5459",
  "status": "PASS" | "WARNING" | "NEEDS_CONTEXT" | "NOT_EVALUATED",
  "severity": "INFO" | "WARNING",   # ERROR nunca é emitido pela lógica atual
  "code": "...",                     # ex: ISO1101_REQUIRED_DATUM_MISSING
  "finding": "texto humano do resultado",
  "recommended_action": "...",
  "candidate_id": "...",             # referencia o quadro GD&T, quando aplicável
  "datum": "...",                    # referencia a letra do datum, quando aplicável
  "normative_claim": false,
}
```

O `summary` é apenas uma contagem por status (`PASS`/`WARNING`/`NEEDS_CONTEXT`/`NOT_EVALUATED`) sobre essa lista — não há nota ou score agregado de "conformidade geral"; a leitura é sempre item a item.

O `gdt_frames` final do `result.json` **não** é o formato reduzido que `build_cad_review_result()` monta internamente (que só guarda characteristic/tolerance/datums) — `process_cad_pdf()` deliberadamente sobrescreve esse campo com a versão rica (`raw_frames`, geometria completa + `symbol_scoring` completo), porque a interface e o relatório visual precisam da geometria (bounding boxes) para desenhar as anotações.

---

## 13. Evidência visual (imagens anotadas por página)

**Onde:** `src/cad_review/visual_output.py` → `render_visual_evidence()`. Chamada ao final de `process_cad_pdf()`.

Para cada página do PDF, três imagens PNG são geradas e salvas em disco (não em memória — a UI lê o arquivo do disco depois):

| Imagem | Conteúdo | Uso na UI |
|---|---|---|
| `page_NNN_annotated.png` | GD&T + datums combinados na mesma imagem | Aba **Marked Drawing** |
| `page_NNN_gdt.png` | Somente os quadros GD&T | Aba **GD&T Evaluation** |
| `page_NNN_datums.png` | Somente os datums | Aba **Datum Definitions** |

A cor de cada retângulo desenhado reflete o **status do finding daquele candidato/datum** (verde = PASS, vermelho = WARNING, laranja = NEEDS_CONTEXT, azul = NOT_EVALUATED) — ou seja, a imagem já embute visualmente a conclusão da avaliação ISO, não é um overlay neutro. Os rótulos são posicionados em "faixas livres" ao redor da geometria de origem (`_find_free_label_position`), com uma linha conectando o rótulo de volta ao retângulo, para reduzir sobreposição em desenhos densos.

Além das três imagens por página, recortes individuais de cada quadro/datum (`crops/GDT-CAND-P01-001_frame.png`, `crops/DATUM-A_001_01.png`) são salvos separadamente — hoje não exibidos na UI simplificada, mas disponíveis no diretório de trabalho e referenciados em `artifacts.visual_evidence.crops`.

---

## 14. Diagnóstico de detecção (não exposto na UI atual)

**Onde:** `src/cad_review/detection_diagnostics.py` → `render_detection_diagnostics()`. Chamada também ao final de `process_cad_pdf()`.

Paralelamente à evidência visual "final", o pipeline gera um segundo conjunto de artefatos pensado para quem está **validando o próprio detector**, não para o usuário final da ferramenta:

- `page_NNN_candidates.png` — só os candidatos de geometria da Fase 1 (antes de qualquer classificação), rotulados apenas com o ID do candidato.
- `page_NNN_symbol_contact_sheet.png` — um cartão por candidato, mostrando o recorte do quadro, o recorte do símbolo e o ranking completo (top-3 classes com score) devolvido pelo classificador.
- `candidate_diagnostics.csv` — uma linha por candidato com geometria, ranking, e **três colunas propositalmente vazias** (`human_is_real_gdt`, `human_true_characteristic`, `human_notes`) para um engenheiro preencher manualmente durante validação.

Essa separação existe para poder responder duas perguntas de forma independente: "a Fase 1 (geometria) propôs o quadro certo?" e, se sim, "a Fase 2 (símbolo) classificou certo?" — sem que um artefato misture as duas camadas de incerteza.

---

## 15. Exibição dos resultados na interface

**Onde:** `pages/classification.py`.

Ao final do processamento, o resultado é organizado em abas (a UI foi deliberadamente simplificada para focar em evidência visual + download, não em tabelas detalhadas de findings):

1. **🏷️ Classification** — componente, família de material, tipo de documento e série do compressor (cada um com valor + evidência textual), e a lista de normas citadas no desenho **filtrando normas ISO** (que são tratadas separadamente pela lógica de GD&T/ISO, não como "norma de peça" citada).
2. **🖼️ Marked Drawing** — a imagem combinada (GD&T + datums) por página, com botão de download em PNG.
3. **📐 GD&T Evaluation** — a imagem só de GD&T por página, com botão de download.
4. **🎯 Datum Definitions** — a imagem só de datums por página, com botão de download.
5. **🗂️ Full JSON** — o `result.json` completo navegável na tela e disponível para download.

Note que o `result.json` subjacente contém muito mais detalhe do que a UI expõe diretamente (comparação de normas, cada finding ISO 1101/5459 individual, scoring completo do classificador de símbolo, geometria de cada candidato) — tudo isso permanece acessível pela aba **Full JSON** para quem precisar de auditoria completa, mesmo que a experiência principal priorize as imagens marcadas.

Cada análise roda em um diretório temporário próprio (`tempfile.mkdtemp(prefix="cad_review_")`), limpo automaticamente antes de uma nova análise ser iniciada.

---

## 16. Resumo dos módulos e onde encontrar cada trecho de código

| Etapa do pipeline | Arquivo | Função/Trecho |
|---|---|---|
| Upload e preview | `pages/classification.py` | seção `Upload do PDF` |
| Extração de texto vetorial | `src/modeling/llm_models.py` | `extract_text_from_pdf` |
| Classificação da peça (LLM) | `src/modeling/llm_models.py` | `classify_cad_enriched` |
| Prompt de classificação | `prompts.py` | `classificacao_enriquecida_prompt` |
| Orquestração classificação + normas | `src/cad_review/orchestrator.py` | `run_part_classification_branch` |
| Consulta de normas aplicáveis | `src/utils/standards_applicability.py` | `StandardsApplicabilityEngine.get_applicable_standards` |
| Comparação determinística de normas | `src/utils/standards_applicability.py` | `compare_standards` |
| Detecção de datums definidos | `src/gdt/datum_feature.py` | `detect_datum_feature_indicators` |
| Detecção geométrica de quadros GD&T | `src/gdt/detector.py`, `src/utils/gdt_detector.py` | `GdtFrameDetector.detect_frames` |
| Classificação do símbolo (ranking) | `src/gdt/symbol_classifier.py` | `load_template_catalog`, `score_candidates` |
| Parsing estrutural do quadro | `src/gdt/frame_parser.py` | `parse_feature_control_frame` |
| Resolução de edição ISO 1101 | `src/gdt/iso1101.py` | `resolve_iso1101_edition`, `assess_datum_reference_semantics` |
| Finding ISO 1101 (datum requirement) | `src/gdt/iso1101_reference.py` | `assess_iso1101_datum_rule` |
| Finding ISO 5459 (consistência de datum) | `src/gdt/datum_consistency.py` | `assess_referenced_datum_definitions` |
| Agregação final (findings + summary) | `src/cad_review/compliance_engine.py` | `build_cad_review_result` |
| Orquestração de tudo + escrita do result.json | `src/cad_review/folder_pipeline.py` | `process_cad_pdf` |
| Evidência visual anotada | `src/cad_review/visual_output.py` | `render_visual_evidence` |
| Diagnóstico de detecção (CSV + contact sheet) | `src/cad_review/detection_diagnostics.py` | `render_detection_diagnostics` |
| Exibição na interface | `pages/classification.py` | `_render_classification_tab`, `_render_combined_overview`, `_render_gdt_tab`, `_render_datums_tab` |

---

## 17. Pontos de destaque para a apresentação

1. **Uma única chamada de IA em todo o pipeline** — só a classificação da peça (seção 4) usa LLM. Toda a parte de GD&T, datums e conformidade ISO (seções 6 a 12) é 100% determinística: mesma entrada, mesma saída, sempre — importante para argumentar auditabilidade e custo previsível.
2. **"Candidato" é uma palavra escolhida a dedo** — o prefixo `GDT-CAND-` e a redação "Potential violation" em todos os findings (seções 10 e 11) existem porque o pipeline nunca afirma ter validado um quadro GD&T ou uma violação normativa contra um humano; ele aponta, com evidência, onde olhar.
3. **Fail-closed, não fail-silent** — quando falta contexto para decidir algo (edição da ISO não citada, catálogo de símbolos incompleto, característica condicional), o sistema devolve explicitamente `NOT_EVALUATED`/`NEEDS_CONTEXT` em vez de arriscar um palpite. Isso é uma escolha de design repetida em pelo menos quatro módulos diferentes (`iso1101.py`, `frame_parser.py`, `symbol_classifier` via `folder_pipeline.py`, `datum_consistency.py`).
4. **Duas camadas de evidência visual, propositalmente separadas** — a imagem "final" (seção 13) mistura geometria + resultado da avaliação; o diagnóstico (seção 14) mostra só a geometria bruta antes de qualquer julgamento. Isso permite validar o detector e a lógica de conformidade de forma independente.
5. **A regra de negócio mora na configuração, não no código** — a tabela de exigência de datum por característica (seção 10) é um JSON externo com a citação exata do trecho da norma; trocar de edição ou ajustar uma regra não exige alterar código Python.
