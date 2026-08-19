# Google Sheets Integration Setup

Este documento descreve como configurar a integração com Google Sheets para carregar a tabela de normas TSS dinamicamente.

## Por que usar Google Sheets?

- ✅ **Atualizações em tempo real**: Cliente atualiza as normas no Sheets e a pipeline usa automaticamente
- ✅ **Sem deploy**: Não precisa atualizar o repositório ou fazer redeploy para adicionar/modificar normas
- ✅ **Colaborativo**: Múltiplos usuários podem editar a planilha
- ✅ **Fallback automático**: Se o Sheets não estiver disponível, usa o `normas.xlsx` local

## Pré-requisitos

- Conta no Google Cloud Platform (GCP)
- Projeto GCP criado (pode ser o mesmo usado para Vertex AI)
- Acesso ao Google Sheets com a planilha de normas

---

## Passo 1: Criar Service Account

1. Acesse o [Google Cloud Console](https://console.cloud.google.com)
2. Selecione seu projeto (ou crie um novo)
3. No menu lateral, vá em **IAM & Admin** → **Service Accounts**
4. Clique em **+ CREATE SERVICE ACCOUNT**
5. Preencha:
   - **Service account name**: `cad-review-sheets-reader`
   - **Service account ID**: `cad-review-sheets-reader` (gerado automaticamente)
   - **Description**: "Service account para leitura da tabela de normas TSS no Google Sheets"
6. Clique em **CREATE AND CONTINUE**
7. **Não precisa adicionar roles** no projeto (pule esta etapa)
8. Clique em **DONE**

---

## Passo 2: Criar chave JSON

1. Na lista de Service Accounts, clique na service account recém-criada (`cad-review-sheets-reader`)
2. Vá na aba **KEYS**
3. Clique em **ADD KEY** → **Create new key**
4. Selecione **JSON**
5. Clique em **CREATE**
6. Um arquivo JSON será baixado automaticamente (ex: `projeto-abc123-xyz456.json`)
7. **Renomeie o arquivo** para `service-account-key.json`
8. **Mova o arquivo** para a raiz do repositório (onde está o `run_review.py`)

⚠️ **IMPORTANTE**: Adicione `service-account-key.json` no `.gitignore` para não comitar credenciais!

---

## Passo 3: Habilitar Google Sheets API

1. No [Google Cloud Console](https://console.cloud.google.com)
2. Vá em **APIs & Services** → **Library**
3. Procure por "Google Sheets API"
4. Clique em **Google Sheets API**
5. Clique em **ENABLE**

---

## Passo 4: Compartilhar a planilha com a Service Account

1. Copie o **email da service account** criada:
   - Formato: `cad-review-sheets-reader@seu-projeto.iam.gserviceaccount.com`
   - Você encontra na lista de Service Accounts ou no arquivo JSON baixado (campo `client_email`)

2. Abra sua **planilha Google Sheets** com as normas TSS

3. Clique no botão **Share** (Compartilhar) no canto superior direito

4. Cole o email da service account no campo "Add people and groups"

5. Defina a permissão como **Viewer** (Leitor)

6. **Desmarque** "Notify people" (não precisa enviar email)

7. Clique em **Share**

✅ Agora a service account tem permissão de leitura na planilha!

---

## Passo 5: Obter o ID da planilha

A URL da sua planilha tem este formato:

```
https://docs.google.com/spreadsheets/d/1ABC123xyz_SPREADSHEET_ID/edit#gid=0
```

O **Spreadsheet ID** é a parte entre `/d/` e `/edit`:

```
1ABC123xyz_SPREADSHEET_ID
```

Copie este ID.

---

## Passo 6: Configurar variáveis de ambiente

Edite o arquivo `.env` na raiz do projeto e adicione:

```bash
# Google Sheets Integration
GOOGLE_SHEETS_CREDENTIALS_PATH=service-account-key.json
GOOGLE_SHEETS_SPREADSHEET_ID=1ABC123xyz_SPREADSHEET_ID
GOOGLE_SHEETS_RANGE=Notes!B2:F
```

**Explicação dos campos:**

- `GOOGLE_SHEETS_CREDENTIALS_PATH`: caminho relativo para o arquivo JSON da chave
- `GOOGLE_SHEETS_SPREADSHEET_ID`: ID da planilha (extraído da URL)
- `GOOGLE_SHEETS_RANGE`: range da planilha no formato `Sheet!StartCell:EndCell`
  - `Notes`: nome da aba/sheet
  - `B2:F`: colunas B até F (Standard, Content, Category, Compressor Series, Applicability), começando na linha 2 (linha 1 é header)

---

## Passo 7: Instalar dependências

Instale as bibliotecas do Google API:

```bash
pip install google-auth google-api-python-client
```

Ou adicione ao `pyproject.toml`:

```toml
[project]
dependencies = [
    # ... suas deps existentes
    "google-auth>=2.0.0",
    "google-api-python-client>=2.0.0",
]
```

E depois:

```bash
pip install -e .
```

---

## Passo 8: Testar

Execute a pipeline normalmente:

```bash
python run_review.py original.pdf revised.pdf -o REVIEW_RESULTS
```

Você deve ver no log:

```
INFO: Loading standards from Google Sheets: 1ABC123xyz_SPREADSHEET_ID (range: Notes!B2:F)
INFO: Loaded 45 standards from Google Sheets
```

---

## Formato esperado da planilha

A aba **Notes** deve ter este layout:

| A (vazio) | B (Standard) | C (Content) | D (Category) | E (Compressor Series) | F (Applicability) |
|-----------|--------------|-------------|--------------|------------------------|-------------------|
| (vazio)   | TSS 001902   | Geometric Requirements | Surface finishing | EM, D1, V2M, VE1, VG5 | Crankshaft, Crankcase, Connecting Rod, Piston, Piston Pin |
| (vazio)   | TSS 003060   | Roundness Harmonics | Surface finishing | E 6Z, EMF, VR5 | Crankshaft, Crankcase, Connecting Rod, Piston, Piston Pin, Rorating |
| ...       | ...          | ...         | ...          | ...                    | ...               |

- **Linha 1**: pode estar vazia ou ter headers (será ignorada)
- **Linha 2+**: dados das normas
- **Coluna A**: sempre vazia (offset para começar em B)
- **Colunas B-F**: Standard, Content, Category, Compressor Series, Applicability

---

## Fallback para normas.xlsx local

Se as variáveis de ambiente do Sheets **não estiverem configuradas** ou se houver **erro ao acessar o Sheets**, a pipeline automaticamente usa o arquivo `normas.xlsx` local como fallback.

Você verá este warning no log:

```
WARNING: Failed to load from Google Sheets (falling back to local xlsx): [error details]
```

Isso garante que a pipeline sempre funcione, mesmo sem internet ou se a service account estiver com problema.

---

## Troubleshooting

### Erro: `FileNotFoundError: service-account-key.json`

- O arquivo JSON não está na raiz do projeto
- Verifique se o caminho em `GOOGLE_SHEETS_CREDENTIALS_PATH` está correto

### Erro: `google.auth.exceptions.RefreshError`

- A chave JSON está inválida ou expirou
- Crie uma nova chave no Google Cloud Console

### Erro: `googleapiclient.errors.HttpError: 403`

- A service account não tem permissão de leitura na planilha
- Verifique se você compartilhou a planilha com o email da service account
- Verifique se a Google Sheets API está habilitada no projeto

### Erro: `No data found in Sheets range`

- O range está incorreto
- Verifique se a aba "Notes" existe
- Verifique se há dados a partir da linha 2

### Pipeline funciona mas não usa o Sheets

- Verifique se as variáveis de ambiente estão definidas no `.env`
- Rode `python -c "import os; print(os.getenv('GOOGLE_SHEETS_SPREADSHEET_ID'))"`
- Se retornar `None`, o `.env` não foi carregado

---

## Segurança

⚠️ **NUNCA comite o arquivo `service-account-key.json` no Git!**

Adicione ao `.gitignore`:

```gitignore
# Google Cloud credentials
service-account-key.json
*.json
!package.json
!tsconfig.json
```

Se você acidentalmente comitou a chave:

1. **Revogue a chave imediatamente** no Google Cloud Console
2. Crie uma nova chave
3. Use `git filter-branch` ou BFG Repo-Cleaner para remover do histórico

---

## Atualização de normas

Quando o cliente atualizar as normas no Google Sheets:

✅ **Não precisa fazer nada!**

A pipeline carrega as normas do Sheets **em tempo de execução**, então qualquer mudança na planilha é automaticamente refletida na próxima execução.

---

## Migração completa para Sheets (opcional)

Se você quiser **remover completamente** o `normas.xlsx` local e usar apenas o Sheets:

1. Configure o Sheets conforme este guia
2. Teste a pipeline para garantir que está carregando do Sheets
3. Delete o arquivo `normas.xlsx` (ou mova para um backup)
4. Remova o fallback em `tss_mapper.py` (opcional)

⚠️ Se você remover o fallback, a pipeline **falhará** se o Sheets estiver indisponível.
