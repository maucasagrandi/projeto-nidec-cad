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
  negative_controls/   # legado; não é classe GD&T válida
```

## Regras

- nomes de pasta são semânticos e canônicos;
- uma pasta por característica;
- várias imagens da mesma classe podem coexistir (`position_01.png`, `position_02.png`...);
- screenshots de tabelas ISO inteiras não entram aqui: somente o símbolo usado para comparação;
- `Roundness` é tratado como alias de entrada de `circularity`; o catálogo usa `circularity`;
- as imagens humanas/versionadas podem ficar em `cotas/`; `sync_phase4_templates.py` prepara as cópias usadas no runtime.

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

## Fase 4 — primeiro lote

Classes ativas esperadas:

- `position`;
- `profile`;
- `straightness`;
- `flatness`;
- `circularity`;
- `cylindricity`.

Sincronização das quatro novas referências já versionadas em `cotas/`:

```powershell
python validation/gdt/scripts/sync_phase4_templates.py
```

Regressão no caso 41:

```powershell
python validation/gdt/scripts/run_phase4_regression.py
```

Essa regressão verifica apenas que a expansão do catálogo não rouba as seis
classificações `Position/Profile` já validadas. Ela **não** valida ainda as
quatro classes novas em CAD real e **não** calibra threshold.
