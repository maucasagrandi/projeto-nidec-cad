# Validação GD&T

Esta pasta contém somente artefatos de validação. Código de produção fica em `src/gdt`.

## Estrutura

```text
validation/gdt/
  cases/          # configuração versionada de cada CAD de teste
  review/         # seleção humana dos candidatos reais
  ground_truth/   # referência congelada após revisão
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

## Fase 1 — geometria

O caso inicial está cadastrado em:

```text
validation/gdt/cases/case_41_rev8.json
```

### 1. Gerar candidatos

```bash
python validation/gdt/scripts/bootstrap_case.py \
  --case validation/gdt/cases/case_41_rev8.json
```

Saídas locais:

```text
validation/gdt/outputs/case_41_rev8/
  candidates.json
  candidates.png
```

A imagem numera os candidatos e o JSON registra bboxes, células, textos e a
configuração geométrica usada pelo detector.

### 2. Revisar os candidatos

Crie:

```text
validation/gdt/review/case_41_rev8.review.json
```

Liste apenas IDs que realmente são quadros GD&T e classifique-os inicialmente
como `position` ou `profile`. Se existir um quadro real que não foi proposto pelo
detector, ele entra em `manual_frames`; isso evita esconder falsos negativos.

### 3. Congelar o ground truth

```bash
python validation/gdt/scripts/build_ground_truth.py \
  --candidates validation/gdt/outputs/case_41_rev8/candidates.json \
  --review validation/gdt/review/case_41_rev8.review.json \
  --output validation/gdt/ground_truth/case_41_rev8.json
```

### 4. Medir o detector

```bash
python validation/gdt/scripts/validate_geometry.py \
  --pdf "CAD_Review_Test_Battery_V1/2. Comparison Analysis/41/13358002_REV_8_draw_2.pdf" \
  --ground-truth validation/gdt/ground_truth/case_41_rev8.json \
  --minimum-recall 0.95 \
  --output validation/gdt/outputs/case_41_rev8/geometry_metrics.json
```

O JSON final registra TP, FN, FP, recall, precision, F1, IoU por quadro e o gate
de recall da Fase 1. Nesta fase, recall tem prioridade; falsos positivos ainda
podem sobreviver para serem filtrados na classificação do símbolo.

## Ground truth

As coordenadas são em pontos PDF, no mesmo sistema do PyMuPDF. O ground truth é
versionado somente depois da revisão visual do CAD original.
