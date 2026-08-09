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
  reference_catalog.json  # manifesto das referências visuais versionadas
```

## Regra principal

Cada fase é validada isoladamente:

1. Geometria: o quadro foi encontrado?
2. Classificação: o símbolo da primeira célula foi identificado?
3. Generalização: o comportamento se mantém em outros CADs?
4. Expansão do catálogo: novas classes não quebram as antigas?
5. Interpretação: tolerância e datums referenciados foram lidos?
6. Conformidade: a regra ISO foi aplicada corretamente?

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

### Templates iniciais

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

### Descoberta

```powershell
python validation/gdt/scripts/discover_phase3_cases.py
```

A descoberta apontou desenhos com alta densidade de candidatos para revisão independente. Esses candidatos não são ground truth.

### Status atual

A anotação manual dos novos CADs foi **adiada**, não aprovada, porque a visualização do anotador atual ficou pequena/ilegível em desenhos densos. Não serão fabricados ground truths a partir dos candidatos do detector.

Consequências metodológicas:

- não afirmar generalização multi-CAD ainda;
- não calibrar `ACCEPTED / AMBIGUOUS / UNKNOWN` com esses casos;
- retomar a Fase 3 quando o annotator tiver zoom/pan ou revisão por crops legíveis.

Os scripts de descoberta, inicialização e agregação permanecem versionados para retomada posterior.

---

## Fase 4 — expansão incremental do catálogo

A Fase 4 amplia as classes visuais sem alterar a geometria nem criar regras de compliance.

### Referências humanas

As imagens fornecidas ficam versionadas em:

```text
cotas/
```

O mapeamento entre filename e classe canônica fica em:

```text
validation/gdt/reference_catalog.json
```

Isso torna a expansão data-driven: novas referências entram pelo manifesto, sem hardcode no classificador.

### Primeiro lote

```text
position       # já existente
profile        # já existente
straightness
flatness
circularity    # referência recebida como Roundness
cylindricity
```

Sincronizar `cotas/` para o catálogo semântico:

```powershell
python validation/gdt/scripts/sync_phase4_templates.py
```

O script aplica apenas normalização de contraste e trim de whitespace externo. Ele não redesenha o símbolo.

### Mudança importante nos controles negativos

O círculo simples usado no experimento inicial como `negative_controls` deixa de ser um controle negativo válido quando `circularity` entra no catálogo, pois o círculo passa a representar uma característica GD&T real.

Por isso a regressão da Fase 4 exclui `negative_controls` da competição visual. Candidatos geométricos extras continuam servindo como população negativa para a futura calibração de aceitação.

### Regressão automática no caso 41

```powershell
python validation/gdt/scripts/run_phase4_regression.py
```

Esse comando:

1. sincroniza as referências ativas do manifesto;
2. pontua o caso 41 com todas as classes válidas;
3. exclui `negative_controls` da competição;
4. avalia contra o ground truth independente já existente;
5. exige que os 3 Position + 3 Profile continuem 6/6 no ranking.

Ele **não valida as novas classes em CAD real** e **não calibra threshold**. É um teste de regressão/competição entre classes.

---

## Próximas fases

Depois da expansão visual e da retomada da generalização:

```text
feature control frame detectado
    ↓
segmentação estrutural das células
    ↓
cell[0] -> característica GD&T
cell[1] -> tolerância/modificadores
cell[2+] -> referências de datum
    ↓
resolver edição ISO aplicável
    ↓
regras determinísticas Table 1 / Table 2
    ↓
detector de datum feature indicator
    ↓
compliance
```

A Table 3 será usada principalmente para orientar o parser estrutural das células. Table 1/Table 2 serão convertidas para regras versionadas; screenshots da norma não serão consultados em runtime.

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

Expansão incremental de classes
    ↓
regressão das classes já suportadas

Vários CADs rotulados
    ↓
calibração de decisão

Leitura das demais células + datum
    ↓
regras ISO
    ↓
conformidade
```
