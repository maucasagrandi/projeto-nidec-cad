# Validação GD&T

Esta pasta contém somente artefatos de validação. Código de produção fica em `src/gdt`.

## Estrutura

```text
validation/gdt/
  ground_truth/   # anotações manuais versionadas
  scripts/        # executáveis de validação
  outputs/        # resultados locais; não precisam ser commitados
```

## Regra principal

Cada fase é validada isoladamente.

1. Geometria: o quadro foi encontrado?
2. Classificação: o símbolo da primeira célula foi identificado?
3. Interpretação: tolerância e datums referenciados foram lidos?
4. Conformidade: a regra ISO foi aplicada corretamente?

Não ajustamos vários blocos ao mesmo tempo.

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
