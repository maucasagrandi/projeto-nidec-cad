# Templates GD&T

Catálogo visual usado pelo classificador da primeira célula do feature control frame.

```text
assets/gdt/templates/
  position/
  profile/
  straightness/
  flatness/
  circularity/
  cylindricity/
  parallelism/
  perpendicularity/
  angularity/
  circular_runout/
  total_runout/
  concentricity_coaxiality/
  symmetry/
  negative_controls/   # legado; não é classe GD&T válida
```

## Regras

- nomes de pasta são semânticos e canônicos;
- uma pasta por classe visual;
- várias imagens da mesma classe podem coexistir;
- screenshots de tabelas ISO inteiras não entram como template;
- `Roundness` é normalizado para `circularity`;
- `Concentricity` e `Coaxiality` são preservados como termos de origem, mas entram na mesma classe visual `concentricity_coaxiality`;
- reconhecer uma classe visual não decide ainda se aquela característica é válida/aplicável para uma edição específica da ISO 1101;
- as imagens humanas/versionadas ficam em `cotas/`; `sync_phase4_templates.py` prepara as cópias usadas pelo scorer.

## Scoring visual

A Fase 4 usa quatro componentes por comparação template/crop:

```text
gray
binary
edges
structure
```

Os três primeiros medem correlação visual local. `structure` compara a forma inteira do símbolo por meio de:

- mapa de ocupação em baixa resolução;
- projeção horizontal dos traços;
- projeção vertical dos traços.

A motivação é evitar que um símbolo simples, como `straightness`, vença por encaixar apenas em um subtrecho de um símbolo mais complexo, como `position`.

O score do template é a média simples desses quatro componentes e a classe usa o melhor template disponível.

**O resultado continua sendo somente ranking diagnóstico.** Não é probabilidade, não é regra ISO e nenhum threshold `ACCEPTED / AMBIGUOUS / UNKNOWN` foi calibrado.

## Importante: círculo não é controle negativo

No primeiro experimento, um círculo simples foi usado como `negative_controls`. A partir da Fase 4 isso deixa de ser válido, porque o círculo é o símbolo de `circularity`/roundness.

Por isso a regressão executa o scorer com:

```text
--exclude-class negative_controls
```

Candidatos geométricos extras continuam sendo a população negativa para calibração futura.

## Fase 4 — catálogo expandido

Classes visuais ativas esperadas na regressão:

- `position`;
- `profile`;
- `straightness`;
- `flatness`;
- `circularity`;
- `cylindricity`;
- `parallelism`;
- `perpendicularity`;
- `angularity`;
- `circular_runout`;
- `total_runout`;
- `concentricity_coaxiality`;
- `symmetry`.

O mapeamento completo filename -> classe canônica está em:

```text
validation/gdt/reference_catalog.json
```

Sincronização:

```powershell
python validation/gdt/scripts/sync_phase4_templates.py
```

O sync também verifica cobertura: se existir uma imagem em `cotas/` não registrada no manifesto, ele falha explicitamente.

Regressão:

```powershell
python validation/gdt/scripts/run_phase4_regression.py
```

A regressão agora imprime os seis quadros reais individualmente e exige que os 3 `Position` + 3 `Profile` continuem 6/6.

Ela **não** valida ainda as classes novas em CAD real e **não** calibra threshold.
