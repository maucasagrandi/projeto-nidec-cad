# Validação GD&T

Esta pasta contém artefatos de validação. Código de produção fica em `src/gdt`.

## Estrutura

```text
validation/gdt/
  baselines/      # resultados congelados de fases aprovadas
  cases/          # configuração versionada de cada CAD de teste
  review/         # revisão humana/provisória
  ground_truth/   # referência independente e versionada
  scripts/        # executáveis de validação
  outputs/        # resultados locais; ignorados pelo Git
```

## Regra principal

Cada fase é validada isoladamente:

1. Geometria: o quadro foi encontrado?
2. Classificação: o símbolo da primeira célula foi identificado?
3. Generalização: o comportamento se mantém em outros CADs?
4. Interpretação: tolerância e datums referenciados foram lidos?
5. Conformidade: a regra ISO foi aplicada corretamente?

A saída de uma etapa não pode ser usada como verdade de referência da própria etapa.

---

## Fase 1 — geometria

Caso inicial:

```text
validation/gdt/cases/case_41_rev8.json
```

Ground truth independente:

```text
validation/gdt/ground_truth/case_41_rev8.json
```

Baseline aprovada:

```text
validation/gdt/baselines/case_41_rev8.geometry.json
```

Resultado atual da pasta 41:

```text
GT = 6
Candidatos = 11
TP = 6
FN = 0
FP = 5
Recall = 1.0000
Precision = 0.5455
F1 = 0.7059
Status = PASS
```

O matching registra `iou`, `overlap_smallest`, `area_ratio` e `match_reason` para tolerar pequenas imprecisões do ROI manual sem esconder a qualidade do bbox vetorial.

---

## Fase 2 — scoring visual do símbolo

A Fase 2 trabalha somente na primeira célula (`symbol_bbox`) de cada candidato e não altera o `frame_bbox` aprovado na Fase 1.

### Refinamento interno de células

O frame externo precisa de `endpoint_tolerance` permissivo para não perder quadros reais. Essa tolerância não é reutilizada cegamente para divisórias internas: `src/gdt/detector.py` refaz apenas a segmentação das células com `cell_endpoint_tolerance=1.0` por padrão.

Isso impede que traços internos do próprio símbolo Position sejam interpretados como paredes de célula.

### Templates

Estrutura:

```text
assets/gdt/templates/
  position/
  profile/
  negative_controls/
```

Para copiar imagens locais para essa estrutura:

```powershell
python validation/gdt/scripts/register_templates.py --position "CAMINHO_POSITION.png" --profile "CAMINHO_PROFILE.png" --negative "CAMINHO_NEGATIVO.png"
```

Cada argumento pode ser repetido para registrar mais de um template da mesma classe.

### Executar a Fase 2 em um comando

```powershell
python validation/gdt/scripts/run_phase2.py --case validation/gdt/cases/case_41_rev8.json
```

Saídas:

```text
validation/gdt/outputs/case_41_rev8/
  symbol_scores.json
  symbol_contact_sheet.png
  symbol_evaluation.json
  symbol_crops/
```

O scorer compara `gray`, `binary` e `edges` e registra `class_scores`, melhor/segunda classe, score, margem e scores por template/representação.

### Resultado diagnóstico do caso 41

Baseline-resumo:

```text
validation/gdt/baselines/case_41_rev8.symbol_summary.json
```

Resultado reproduzido localmente:

```text
quadros reais = 6
ranking correto = 6/6
ranking_accuracy = 1.000
candidatos extras = 5
best_score_gap = +0.046
margin_gap = +0.030
status = PASS_DIAGNOSTIC
```

**Nenhum threshold foi calibrado.** A separação positiva no caso 41 é evidência inicial, não uma regra de produção.

---

## Fase 3 — generalização em outros CADs

### 1. Descobrir casos sem escolher no chute

```powershell
python validation/gdt/scripts/discover_phase3_cases.py
```

O script percorre desenhos `*_draw_*.pdf` da seção Comparison Analysis em DPI baixo apenas para economizar renderização. A geometria continua vindo dos vetores do PDF.

Saída:

```text
validation/gdt/outputs/phase3_discovery.json
```

Ele informa candidatos por CAD e sugere até 5 casos variados. A sugestão não usa classe prevista nem score visual, para evitar selecionar somente desenhos parecidos com os templates atuais.

### 2. Inicializar um caso escolhido

```powershell
python validation/gdt/scripts/init_case.py --case-id case_NOME --pdf "CAMINHO_DO_DRAWING.pdf"
```

O arquivo nasce com `expected=null`: quantidade e classes não são inventadas antes da anotação.

### 3. Anotar ground truth independente

```powershell
python validation/gdt/scripts/annotate_ground_truth.py --case validation/gdt/cases/case_NOME.json
```

### 4. Medir geometria e rodar scoring

Use `validate_geometry.py` para congelar a associação geométrica e depois `run_phase2.py` para produzir `symbol_evaluation.json` daquele caso.

### 5. Agregar vários CADs

```powershell
python validation/gdt/scripts/aggregate_phase3.py \
  validation/gdt/outputs/case_41_rev8/symbol_evaluation.json \
  validation/gdt/outputs/case_NOME/symbol_evaluation.json
```

O agregador calcula ranking global e os gaps entre o menor score/margem dos quadros reais e o maior score/margem dos candidatos extras.

Thresholds `ACCEPTED / AMBIGUOUS / UNKNOWN` só serão discutidos depois de vários CADs independentes.

---

## Princípio metodológico

```text
Detector geométrico
    ↓
candidatos

Ground truth independente
    ↓
métricas de geometria

Primeira célula
    ↓
scores visuais sem threshold

Vários CADs rotulados
    ↓
calibração de decisão

Leitura das demais células + datum
    ↓
regras ISO
    ↓
conformidade
```
