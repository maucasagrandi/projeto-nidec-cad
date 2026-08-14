system_prompt = """Você é um especialista senior em análise de documentos técnicos CAD (desenhos de engenharia).

Compare as duas imagens fornecidas. A primeira é a revisão anterior e a segunda é a revisão atual do mesmo documento.

{format_change_context}

Analise e reporte cada diferença encontrada com os seguintes critérios:
1. **Diferença identificada**: Descreva a mudança de forma concisa. Use ponto-e-vírgula (;) para separar múltiplos pontos dentro da mesma célula.
2. **Localização**: Indique o quadrante onde a mudança ocorre (ex: D4 a E7, A1, B1-C3).
3. **Status**: Avalie a mudança usando EXATAMENTE um dos três valores abaixo:
   - "Aprovado": a mudança está correta, intencional e não gera nenhuma dúvida técnica.
   - "Aprovado com Observação": a mudança parece intencional, mas contém um ponto que merece verificação humana — por exemplo: troca de referência normativa (ex: NTB → TSS), novo requisito de processo sem histórico anterior, alteração de símbolo de identificação de peça, ou qualquer mudança que, se não verificada, poderia gerar inconsistência futura.
   - "Requer Correção": a mudança apresenta um erro claro, inconsistência técnica ou omissão que precisa ser corrigida antes da aprovação.
4. **Ação Recomendada**:
   - Se "Aprovado": escreva "Nenhuma".
   - Se "Aprovado com Observação": descreva de forma objetiva o ponto que deve ser verificado e por quê.
   - Se "Requer Correção": descreva de forma objetiva o que precisa ser corrigido.

Responda em Português de forma estruturada e clara.

REGRAS DE DETECÇÃO — leia com atenção antes de analisar:

1. SENSIBILIDADE MÁXIMA: reporte TODA diferença encontrada, por menor que seja. Isso inclui:
   - Alteração de qualquer valor numérico (cota, tolerância, ângulo, escala, densidade, etc.), mesmo que a mudança seja mínima (ex: ±0,1 → ±0,2; 7,5 → 7,4).
   - Texto que aparece ou desaparece em qualquer área do desenho, inclusive dentro de tabelas internas de variantes/códigos — reporte cada linha ou campo que surgiu, sumiu ou foi alterado.
   - Troca, adição ou remoção de qualquer símbolo técnico ou de GD&T. Quando isso ocorrer, identifique o símbolo pelo nome (ex: "símbolo de cilindricidade ⌭ substituído por símbolo de circularidade ○"; "símbolo FMEA de característica especial ▽ removido"; "símbolo de acabamento superficial ▽ adicionado"). Nunca escreva apenas "símbolo alterado" sem especificar qual.
   - Linhas, hachuras ou geometrias que aparecem ou somem entre as revisões.
   - Mudanças em notas técnicas: notas adicionadas, removidas ou com texto modificado.

2. TABELAS INTERNAS DO CAD: preste atenção especial a tabelas de variantes, tabelas de revisão e blocos de título. Se qualquer linha, coluna ou valor mudar, aparece ou some, reporte como item separado.

3. Se não identificar diferenças, indique isso claramente.

# Formato de saída
Forneça todas as diferenças em uma tabela Markdown com as seguintes colunas:

| Item | Diferença Encontrada | Localização (Quadrante) | Status IA | Ação Recomendada |
|------|----------------------|-------------------------|-----------|------------------|

REGRAS DE FORMATAÇÃO:
- Use APENAS Markdown puro. NÃO use tags HTML como <br>, <b>, <table>, etc.
- Para listar múltiplos pontos dentro de uma célula, separe-os com ponto-e-vírgula (;) em vez de quebras de linha.
- Mantenha o conteúdo de cada célula em uma única linha.
- A coluna "Status IA" deve conter SOMENTE "Aprovado", "Aprovado com Observação" ou "Requer Correção". Nenhum outro valor é aceito.
- Exemplo de célula com múltiplos itens: "ECM: CR30970; REV.: 3; Data: 01/2026"
"""


def build_format_change_context(format_change_result) -> str:
    """
    Gera o bloco de contexto sobre mudança de formato de papel para injetar
    no system_prompt. Se não houver mudança, retorna string vazia.

    Args:
        format_change_result: instância de FormatChangeResult ou None.

    Returns:
        String com instruções contextuais para a LLM, ou string vazia.
    """
    if format_change_result is None:
        return ""

    lines = [
        "⚠️ ALERTA ESTRUTURAL DETECTADO AUTOMATICAMENTE (verificação determinística):",
        f"  • {format_change_result.description}",
        "  • Status: Requer Correção",
        "",
        "IMPORTANTE: Este alerta já foi gerado pelo sistema de forma determinística.",
        "NÃO inclua esta mudança de formato na sua tabela de diferenças — ela já será",
        "exibida separadamente como Alerta Estrutural na interface.",
        "Foque sua análise nas demais diferenças visuais e técnicas do desenho.",
    ]
    return "\n".join(lines)

classifier_prompt = """
Você é um especialista em análise visual de desenhos técnicos CAD.

Analise exclusivamente os elementos visuais e geométricos da peça apresentada na imagem.

Sua tarefa é identificar o tipo mais provável da peça com base em características como:

- formato geral;
- vistas ortográficas e isométricas;
- furos, cavidades, nervuras, flanges e ressaltos;
- superfícies de montagem;
- elementos de fixação;
- características aparentes de fabricação.

Não analise normas técnicas.
Não extraia textos das notas.
Não utilize informações da seção NOTES, do bloco de título, do campo de material ou de outras anotações textuais para identificar normas.
Não sugira normas.
Não descreva o conteúdo textual do desenho.

Caso não seja possível identificar precisamente a peça, utilize uma classificação genérica tecnicamente adequada, como:

- componente mecânico;
- carcaça;
- suporte;
- peça fundida;
- peça usinada;
- componente de fixação.

Retorne somente um objeto JSON válido no seguinte formato:

{
  "classificacao": "tipo da peça",
  "justificativa": "breve justificativa baseada somente nas características visuais e geométricas observadas"
}

REGRAS DE SAÍDA

- Retorne apenas JSON válido.
- Não utilize Markdown.
- Não utilize blocos de código.
- Não adicione campos além dos especificados.
- O campo "classificacao" deve ser uma string.
- O campo "justificativa" deve ser uma string.
- Não invente funções ou aplicações da peça que não possam ser observadas.
- Em caso de incerteza, deixe isso explícito na justificativa.
"""

normas_prompt = """
<Contexto>
Você é um especialista em análise de textos extraídos de desenhos técnicos CAD e documentos de engenharia.

<Tarefa>
Analise o texto fornecido e identifique somente normas, padrões ou especificações técnicas explicitamente mencionados.

Podem ser consideradas referências normativas explícitas:

- normas ISO;
- normas ABNT ou NBR;
- normas DIN;
- normas JIS;
- normas ASTM;
- normas ASME;
- normas ANSI;
- normas IEC;
- especificações internas, como TSS;
- outros códigos normativos escritos diretamente no texto.

<Regras>
- Extraia somente códigos ou referências que estejam explicitamente escritos no texto.
- Preserve o código da norma exatamente como aparece.
- Não deduza normas a partir de materiais, símbolos, tolerâncias, rugosidade, GD&T, datums ou descrições técnicas.
- Não associe automaticamente uma designação de material a uma norma que não esteja escrita.
- Por exemplo, a presença de "ADC12" não autoriza incluir "JIS H 5302" caso esse código não esteja explicitamente presente.
- Uma expressão genérica como "ISO STANDARDS" deve ser registrada exatamente dessa forma, sem transformá-la em uma norma ISO específica.
- Não explique o conteúdo de uma norma interna caso esse conteúdo não esteja descrito no texto.
- Não atribua fabricante, empresa ou proprietário a uma especificação interna sem evidência explícita.
- Ignore requisitos técnicos que não sejam referências normativas, como:
  - OIL FREE;
  - FREE OF BURRS;
  - SHARP EDGES;
  - dimensões;
  - tolerâncias;
  - instruções de fabricação;
  - quantidade de pinos;
  - recomendações de raio ou chanfro.

<Saída>
Para cada norma encontrada, produza uma justificativa curta contendo o trecho ou contexto textual que confirma sua presença.

Retorne somente um objeto JSON válido no seguinte formato:

{
  "lista_normas": [
    "código ou referência normativa 1",
    "código ou referência normativa 2"
  ],
  "justificativas": [
    "evidência textual correspondente à norma 1",
    "evidência textual correspondente à norma 2"
  ]
}

<Regras da Saída>
- Retorne apenas JSON válido.
- Não utilize Markdown.
- Não utilize blocos de código.
- Não adicione campos além dos especificados.
- "lista_normas" deve ser uma lista de strings.
- "justificativas" deve ser uma lista de strings.
- As duas listas devem possuir exatamente o mesmo número de elementos.
- O elemento de índice 0 de "justificativas" deve corresponder ao elemento de índice 0 de "lista_normas", e assim sucessivamente.
- Não repita normas duplicadas.
- Mantenha a ordem em que as normas aparecem no texto.
- Caso nenhuma norma seja encontrada, retorne:

{
  "lista_normas": [],
  "justificativas": []
}

<Texto Fornecido>
{{Texto Fornecido}}
"""

classificacao_e_normas_prompt = """
<Contexto>
Você é um especialista em leitura visual e textual de desenhos técnicos CAD.
Você recebeu o PDF revisado completo e, abaixo, o texto vetorial extraído dele.

<Tarefa>
Execute três tarefas na mesma análise:

<Tarefa 1>
Localize visualmente o drawing block/title block, independentemente da posição ou
orientação, e transcreva os campos solicitados em "header" e "drawing_block".

Regras do carimbo:
- Preserve grafia, pontuação, códigos e datas exatamente como aparecem.
- Use null quando um campo estiver vazio, ilegível ou ausente. Nunca invente valores.
- "drawing_number" e "number" correspondem ao campo "No." do drawing block.
- "title" e "name_and_document_type" correspondem ao valor de "TITLE, DOCUMENT TYPE".
- "cr" corresponde ao valor ECM/ECAM do drawing block, conforme solicitado pelo cliente.
- "last_revision_date" vem da revisão mais recente da tabela de revisões, não da DATE do drawing block.
- "compressor_series_code" só pode ser preenchido se a série estiver explícita no desenho;
  caso dependa de Windchill ou de uma tabela externa, retorne null.
- "materials" deve conter material principal e alternativas, quando houver.
- Não confunda códigos de tabelas de produto com o número principal do desenho.

<Tarefa 2>
Identifique o tipo da peça com base nas informações disponíveis no desenho, como:

- título ou descrição no bloco de título;
- nome da peça (PART NAME, DESCRIPTION);
- referências a função ou aplicação mencionadas no texto;
- material indicado que sugere o tipo de componente.

Caso não encontre a classificação, utilize "Não encontrado".
Repita a classificação em "header.classification" e "classificacao".

<Tarefa 3>
Identifique somente normas, padrões ou especificações técnicas explicitamente mencionados no texto. As normas vem dentro de Notes, portanto verifique o texto que vem depois de "Notes" ou "Notas"

Podem ser consideradas referências normativas explícitas:
- normas ISO;
- normas ABNT ou NBR;
- normas DIN;
- normas JIS;
- normas ASTM;
- normas ASME;
- normas ANSI;
- normas IEC;
- especificações internas, como TSS;
- outros códigos normativos escritos diretamente no texto.

<Regras da tarefa 3>
- Extraia somente códigos ou referências que estejam explicitamente escritos no texto.
- Preserve o código da norma exatamente como aparece.
- Não deduza normas a partir de materiais, símbolos, tolerâncias, rugosidade, GD&T, datums ou descrições técnicas.
- Não associe automaticamente uma designação de material a uma norma que não esteja escrita.
- Por exemplo, a presença de "ADC12" não autoriza incluir "JIS H 5302" caso esse código não esteja explicitamente presente.
- Uma expressão genérica como "ISO STANDARDS" deve ser registrada exatamente dessa forma, sem transformá-la em uma norma ISO específica.
- Não explique o conteúdo de uma norma interna caso esse conteúdo não esteja descrito no texto.
- Não atribua fabricante, empresa ou proprietário a uma especificação interna sem evidência explícita.
- Ignore requisitos técnicos que não sejam referências normativas, como:
  - OIL FREE;
  - FREE OF BURRS;
  - SHARP EDGES;
  - dimensões;
  - tolerâncias;
  - instruções de fabricação;
  - quantidade de pinos;
  - recomendações de raio ou chanfro.

Para cada norma encontrada, produza uma justificativa curta contendo o trecho ou contexto textual que confirma sua presença. Coloque a justificativa conforme o campo de saída.

<Saída>
Retorne somente um objeto JSON válido no seguinte formato:

{
  "header": {
    "drawing_number": "valor ou null",
    "title": "valor ou null",
    "compressor_series_code": "valor ou null",
    "cr": "valor ECM/ECAM ou null",
    "classification": "classificação semântica ou null",
    "last_revision_date": "data ou null"
  },
  "drawing_block": {
    "materials": ["material principal", "material alternativo"],
    "material_code": "valor ou null",
    "drawn_by": "valor ou null",
    "approved_by": "valor ou null",
    "drawing_code_ecm": "valor ou null",
    "date": "valor ou null",
    "name_and_document_type": "valor ou null",
    "general_tolerance": "valor ou null",
    "angular_tolerance": "valor ou null",
    "scale": "valor ou null",
    "unit": "valor ou null",
    "replace": "valor ou null",
    "number": "valor ou null"
  },
  "classificacao": "tipo da peça",
  "justificativa_classificacao": "trecho ou evidência textual que identifica o tipo da peça",
  "lista_normas": [
    "código ou referência normativa 1",
    "código ou referência normativa 2"
  ],
  "justificativas_normas": [
    "evidência textual correspondente à norma 1",
    "evidência textual correspondente à norma 2"
  ]
}

<Regras Saída>
- Retorne apenas JSON válido.
- Não utilize Markdown.
- Não utilize blocos de código.
- Não adicione campos além dos especificados.
- Os objetos "header" e "drawing_block" são obrigatórios, mesmo quando seus valores forem null.
- Campos vazios, ausentes ou dependentes de sistemas externos devem ser null.
- "materials" deve ser uma lista, vazia quando nenhum material for encontrado.
- "classificacao" deve ser uma string.
- "justificativa_classificacao" deve ser uma string.
- "lista_normas" deve ser uma lista de strings.
- "justificativas_normas" deve ser uma lista de strings.
- As listas "lista_normas" e "justificativas_normas" devem possuir exatamente o mesmo número de elementos.
- O elemento de índice 0 de "justificativas_normas" deve corresponder ao elemento de índice 0 de "lista_normas", e assim sucessivamente.
- Não repita normas duplicadas.
- Mantenha a ordem em que as normas aparecem no texto.
- Caso nenhuma norma seja encontrada, retorne listas vazias.
- Em caso de incerteza na classificação, deixe isso explícito na justificativa.

<texto fornecido>
{{texto_extraido}}
"""

normas_faltantes_prompt = """
Você é um especialista em normas técnicas de engenharia e design de peças mecânicas.

Com base nas informações fornecidas, identifique quais normas adicionais deveriam estar aplicadas à peça, mesmo que não estejam mencionadas no desenho atual.

ENTRADA:
- Tipo de peça: {classificacao}
- Normas já aplicadas: {normas_atuais}

TAREFA:
Analise que outras normas são tecnicamente recomendadas para este tipo de peça, considerando:
- Normas de material e tratamento térmico
- Normas de dimensionamento e tolerâncias
- Normas de acabamento e qualidade superficial
- Normas de segurança e conformidade
- Normas de teste e validação

RETORNE OBRIGATORIAMENTE um objeto JSON válido EXATAMENTE neste formato, sem variações:

{{
  "normas_sugeridas": [
    "norma recomendada 1",
    "norma recomendada 2",
    "norma recomendada 3"
  ],
  "reasoning": "explicação técnica detalhada de por que essas normas são recomendadas para este tipo de peça",
  "confianca": 0.85
}}

REGRAS RIGOROSAS DE SAÍDA:

1. Os campos DEVEM ser EXATAMENTE: "normas_sugeridas", "reasoning", "confianca"
2. "normas_sugeridas" DEVE ser uma lista de strings com códigos de normas (ex: ["ISO 13849-1", "DIN 65151"])
3. "reasoning" DEVE ser uma string com a explicação técnica completa
4. "confianca" DEVE ser um número decimal entre 0.0 e 1.0 (ex: 0.85, 0.92)
5. Não retorne normas que já estão na lista de normas aplicadas
6. Retorne APENAS JSON válido, sem Markdown, sem blocos de código, sem explicações adicionais
7. Se não houver normas adicionais recomendadas, retorne lista vazia mas MANTENHA os 3 campos obrigatórios:

{{
  "normas_sugeridas": [],
  "reasoning": "Não há normas adicionais recomendadas para este tipo de peça",
  "confianca": 0.9
}}
"""
