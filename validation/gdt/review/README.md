# Revisão humana do bootstrap GD&T

A revisão desta pasta é pequena e deliberadamente separada do ground truth.

## Fluxo

1. Rode `bootstrap_case.py` para gerar `candidates.json` e `candidates.png`.
2. Abra a imagem e compare com o CAD original.
3. Em um arquivo `*.review.json`, liste somente os candidatos que são quadros GD&T reais.
4. Se existir um quadro real sem candidato desenhado sobre ele, coloque-o em `manual_frames`.
5. Rode `build_ground_truth.py`.

Exemplo:

```json
{
  "case_id": "case_41_rev8",
  "accepted_candidates": [
    {
      "candidate_id": "GDT-CAND-P01-002",
      "characteristic": "position"
    }
  ],
  "manual_frames": []
}
```

No primeiro caso oficial esperamos revisar `position` e `profile` apenas.
A lista `manual_frames` é importante: ela impede que um falso negativo suma do
conjunto de referência só porque o detector não o propôs como candidato.
