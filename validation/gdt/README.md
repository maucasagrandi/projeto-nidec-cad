# Validação GD&T

Esta pasta contém somente artefatos de validação. Código de produção fica em `src/gdt`.

## Estrutura

```text
validation/gdt/
  ground_truth/   # anotações manuais versionadas
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

Antes de criar o ground truth, rode o bootstrap do caso:

```bash
python validation/gdt/scripts/bootstrap_case.py \
  --pdf "CAD_Review_Test_Battery_V1/2. Comparison Analysis/41/13358002_REV_8_draw_2.pdf" \
  --case-id case_41_rev8
```

Ele gera:

```text
validation/gdt/outputs/case_41_rev8/
  candidates.json
  candidates.png
```

A imagem numera os candidatos. O JSON traz os bboxes e células. Esses artefatos
servem para anotar manualmente os quadros verdadeiros sem inventar coordenadas.

Depois de criar `validation/gdt/ground_truth/case_41_rev8.json`, rode:

```bash
python validation/gdt/scripts/validate_geometry.py \
  --pdf "CAD_Review_Test_Battery_V1/2. Comparison Analysis/41/13358002_REV_8_draw_2.pdf" \
  --ground-truth validation/gdt/ground_truth/case_41_rev8.json \
  --output validation/gdt/outputs/case_41_rev8/geometry_metrics.json
```

Nesta fase, a métrica prioritária é **recall**. Falsos positivos ainda são aceitos,
pois serão filtrados posteriormente pela classificação do símbolo.

## Ground truth

Formato esperado:

```json
{
  "pdf": "arquivo.pdf",
  "page": 1,
  "frames": [
    {
      "id": "GT-001",
      "page": 1,
      "characteristic": "position",
      "bbox": [x0, y0, x1, y1]
    }
  ]
}
```

As coordenadas são em pontos PDF, no mesmo sistema do PyMuPDF.
