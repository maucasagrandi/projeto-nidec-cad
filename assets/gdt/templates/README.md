# Templates GD&T

Organização inicial:

```text
assets/gdt/templates/
  position/
  profile/
  negative_controls/
```

Regras:
- nomes semânticos, não `cota1`, `cota2`, etc.;
- uma pasta por classe;
- várias imagens da mesma classe podem coexistir (`position_01.png`, `position_02.png`...);
- controles negativos ficam separados e nunca representam uma classe GD&T válida;
- screenshots de tabelas ISO inteiras não entram aqui: somente símbolos/templates usados pelo classificador.

Classes do primeiro ciclo:
- `position`;
- `profile`;
- `negative_controls` (incluindo o círculo simples criado para teste).
