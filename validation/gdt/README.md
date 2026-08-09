# Validação GD&T

Esta pasta contém somente artefatos de validação. Código de produção fica em `src/gdt`.

## Estrutura

```text
validation/gdt/
  cases/          # configuração versionada de cada CAD de teste
  review/         # revisão humana/provisória dos candidatos
  ground_truth/   # referência geométrica independente e versionada
  scripts/        # executáveis de validação
  outputs/        # resultados locais; ignorados pelo Git
```

## Regra principal

Cada fase é validada isoladamente.

1. Geometria: o quadro foi encontrado?
2. Classificação: o símbolo da primeira célula foi identificado?
3. Interpretação: tolerância e datums referenciados foram lidos?
4. Conformidade: a regra ISO foi aplicada corretamente?

Não ajustamos vários blocos ao mesmo tempo.

---

## Fase 1 — geometria

O caso inicial está cadastrado em:

```text
validation/gdt/cases/case_41_rev8.json
```

### 1. Gerar candidatos

```bash
python validation/gdt/scripts/bootstrap_case.py --case validation/gdt/cases/case_41_rev8.json
```

Saídas locais:

```text
validation/gdt/outputs/case_41_rev8/
  candidates.json
  candidates.png
```

A saída atual da pasta 41 contém **candidatos geométricos**. Antes de existir
uma anotação independente, eles não devem ser chamados de TP/FP/FN.

### 2. Ground truth geométrico independente

Para medir recall/precision do detector, os bboxes verdadeiros precisam ser
anotados olhando o CAD original, sem copiar os bboxes de `candidates.json`.
Esses quadros entram como `manual_frames`.

O código ainda permite `accepted_candidates` como ferramenta de bootstrap, mas
qualquer ground truth contendo bbox copiado do detector recebe:

```json
{
  "independent_annotation": false,
  "benchmark_grade": false
}
```

Por padrão, `validate_geometry.py` bloqueia esse tipo de ground truth. O modo
`--allow-candidate-assisted` existe apenas para debug e marca a execução como
`EXPLORATORY`, nunca como benchmark oficial.

### 3. Medir o detector

Com ground truth independente:

```bash
python validation/gdt/scripts/validate_geometry.py \
  --pdf "CAD_Review_Test_Battery_V1/2. Comparison Analysis/41/13358002_REV_8_draw_2.pdf" \
  --ground-truth validation/gdt/ground_truth/case_41_rev8.json \
  --minimum-recall 0.95 \
  --output validation/gdt/outputs/case_41_rev8/geometry_metrics.json
```

Somente aqui surgem oficialmente TP, FP, FN, recall, precision e F1.

---

## Fase 2 — scoring visual do símbolo

A Fase 2 trabalha somente na primeira célula (`symbol_bbox`) de cada candidato.
Ela não altera a geometria.

Templates ficam em:

```text
assets/gdt/templates/
  position/
  profile/
  negative_controls/
```

Execute:

```bash
python validation/gdt/scripts/score_symbols.py \
  --case validation/gdt/cases/case_41_rev8.json \
  --templates assets/gdt/templates
```

Saídas:

```text
validation/gdt/outputs/case_41_rev8/
  symbol_scores.json
  symbol_crops/
    GDT-CAND-P01-001.png
    ...
```

O scorer compara `gray`, `binary` e `edges`, registra todos os scores por
template, agrega o melhor template de cada classe e calcula:

```text
best_class
best_score
second_best_class
second_best_score
margin
```

**Nenhum threshold é aplicado nesta etapa.** Primeiro observamos a distribuição
dos scores; só depois calibramos `ACCEPTED / AMBIGUOUS / UNKNOWN`.

---

## Princípio metodológico

```text
Detector geométrico -> candidatos
Ground truth independente -> métricas de geometria
Primeira célula -> scores visuais
Labels humanos -> métricas de classificação
Regras ISO -> conformidade
```

A saída de uma etapa não pode ser usada como verdade de referência da própria
etapa que está sendo avaliada.
