# Integrated CAD Review

Aplicação Streamlit para revisar desenhos CAD a partir de dois PDFs: o primeiro é
o desenho original e o segundo é o desenho revisado.

## Fluxo

1. Envia **o PDF revisado e seu texto vetorial** em uma chamada multimodal para
   transcrever Header/Drawing Block, classificar a peça e extrair normas citadas.
2. Executa a detecção **determinística de GD&T e datums somente no revisado** e
   produz uma imagem anotada por página.
3. Compara **original e revisado**: OpenCV alinha as páginas e encontra regiões
   candidatas; a LLM valida os candidatos e descreve as mudanças reais.
4. Gera um relatório PDF único com:
   - tabelas Header e Drawing Block Transcription;
   - Applied Standards em bullet points;
   - mapa de diferenças com IDs;
   - tabela consolidada de diferenças;
   - comparação visual Previous/Current para cada ID confirmado;
   - desenho revisado com GD&T e datums marcados.

## Setup

```bash
# Instalar dependências
pip install -r requirements.txt

# Configurar .env
cp .env.example .env
# Adicione suas credenciais GCP_PROJECT_ID e GCP_REGION
```

## Executar

```bash
streamlit run front.py
```

Também existe um ponto de entrada por linha de comando:

```bash
python run_review.py original.pdf revisado.pdf -o REVIEW_RESULTS --gdt-workers 1 --opencv-dpi 150
```

Durante a execução, o terminal mostra o tempo de cada etapa e o tempo acumulado:

```text
INFO:src.cad_review.integrated_review:TEMPO | Part Classification concluída | etapa=00:00:12.3 | acumulado=00:00:12.8
INFO:src.cad_review.integrated_review:TEMPO | Part Comparison concluída | etapa=00:00:31.4 | acumulado=00:01:04.7
INFO:__main__:TEMPO | Execução completa encerrada | total=00:01:08.2
```

As durações do pipeline também são incluídas em `metadata.timings_seconds` no
arquivo `integrated_review.json`.

Para executar apenas o detector determinístico de GD&T/datums:

```bash
python run_gdt.py revisado.pdf -o GDT_RESULTS/revisado
```

## Estrutura

```
.
├── front.py                      # Fluxo unificado no Streamlit
├── run_review.py                 # Fluxo unificado em CLI
├── run_gdt.py                    # GD&T/datums determinístico em CLI
├── compare.py                    # OpenCV + verificação LLM em CLI
├── assets/gdt/templates/         # Templates versionados de símbolos GD&T
├── src/
│   ├── cad_review/
│   │   └── integrated_review.py  # Orquestra os dois PDFs
│   ├── gdt/                      # Detecção, parsing e datum linking
│   ├── modeling/
│   │   └── llm_verify_changes.py # Validação dos crops OpenCV
│   ├── reporting/
│   │   └── unified_cad_report.py # Relatório PDF final
│   └── utils/
│       └── opencv_cad_compare.py # Alinhamento e diferenças visuais
└── requirements.txt              # Dependências
```

## Variáveis de Ambiente

- `GCP_PROJECT_ID` - ID do projeto GCP
- `GCP_REGION` - Região (padrão: us-east5)
- `APP_USERNAME` e `APP_PASSWORD` - autenticação da interface Streamlit
