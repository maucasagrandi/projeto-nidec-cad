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
3. Interpretação: tolerância e datums referenciados foram lidos?
4. Conformidade: a regra ISO foi aplicada corretamente?

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

A Fase 2 trabalha somente na primeira célula (`symbol_bbox`) de cada candidato e não altera a geometria.

### Templates

Estrutura:

```text
assets/gdt/templates/
  position/
  profile/
  negative_controls/
```

Para copiar imagens locais para essa estrutura sem depender de nomes como `cota1`/`cota2`:

```powershell
python validation/gdt/scripts/register_templates.py --position "CAMINHO_POSITION.png" --profile "CAMINHO_PROFILE.png" --negative "CAMINHO_NEGATIVO.png"
```

Cada argumento pode ser repetido para registrar mais de um template da mesma classe.

### Executar a Fase 2 em um comando

```powershell
python validation/gdt/scripts/run_phase2.py --case validation/gdt/cases/case_41_rev8.json
```

Ele executa o scoring e a avaliação diagnóstica contra a baseline da Fase 1.

Saídas:

```text
validation/gdt/outputs/case_41_rev8/
  symbol_scores.json
  symbol_contact_sheet.png
  symbol_evaluation.json
  symbol_crops/
```

O scorer compara três representações:

```text
gray
binary
edges
```

E registra:

```text
class_scores
best_class
best_score
second_best_class
second_best_score
margin
scores por template e representação
```

**Nenhum threshold é aplicado nesta etapa.**

`symbol_evaluation.json` mede apenas o ranking da classe nos quadros reais já associados geometricamente e mostra separadamente o comportamento dos candidatos extras.

Thresholds `ACCEPTED / AMBIGUOUS / UNKNOWN` só serão calibrados depois de observar a distribuição dos scores em mais de um CAD, para evitar overfit no caso 41.

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
