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
- várias imagens da mesma classe podem coexistir (`position_01.png`, `position_02.png`...);
- screenshots de tabelas ISO inteiras não entram aqui: somente símbolos/templates usados para comparação;
- `Roundness` é tratado como alias de entrada de `circularity`;
- `Concentricity` e `Coaxiality` são preservados como termos de origem, mas entram na mesma classe visual `concentricity_coaxiality`;
- reconhecer uma classe visual não decide ainda se aquela característica é válida/aplicável para uma edição específica da ISO 1101;
- as imagens humanas/versionadas ficam em `cotas/`; `sync_phase4_templates.py` prepara as cópias usadas pelo scorer.

## Importante: círculo não é mais controle negativo

No primeiro experimento, um círculo simples foi usado como `negative_controls`. A partir
da Fase 4 isso deixa de ser válido, porque o círculo é o símbolo de
`circularity`/roundness.

Por isso a regressão da Fase 4 executa o scorer com:

```text
--exclude-class negative_controls
```

Candidatos geométricos extras continuam sendo a população negativa para
calibração futura de `ACCEPTED / AMBIGUOUS / UNKNOWN`; não precisamos de uma
classe visual artificial chamada "negative".

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

Sincronização das referências versionadas em `cotas/`:

```powershell
python validation/gdt/scripts/sync_phase4_templates.py
```

O sync também verifica cobertura: se existir uma imagem em `cotas/` que não
esteja registrada no manifesto, ele falha explicitamente.

Regressão no caso 41:

```powershell
python validation/gdt/scripts/run_phase4_regression.py
```

Essa regressão verifica apenas que a expansão do catálogo não rouba as seis
classificações `Position/Profile` já validadas. Ela **não** valida ainda as
classes novas em CAD real e **não** calibra threshold.
