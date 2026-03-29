# JONAH RAG — Checklist de Deploy
## Hugging Face Spaces · FEn / UFPel

---

## Visão geral do que vai ser criado

```
Hugging Face Spaces
└── jonah-rag/                  ← repositório do app
    ├── app_jonah.py
    ├── rag_jonah.py
    ├── ingest_jonah.py
    ├── drive_sync.py
    ├── requirements.txt
    ├── README.md               ← configuração do Space (obrigatório)
    └── data/
        └── indexed_files.json  ← registro incremental (gerado na 1ª indexação)

Google Drive
├── jonah_indice/               ← pasta do índice Chroma zipado
│   └── chroma_index.zip        ← gerado automaticamente pelo app
└── jonah_edicoes/              ← pasta raiz com os 650 PDFs
    ├── v14n1_2024/
    │   ├── artigo1.pdf
    │   └── artigo2.pdf
    └── v13n2_2023/
        └── ...
```

---

## FASE 1 — Google Drive

### 1.1 Organizar as pastas no Drive

- [ ] Confirmar que a pasta raiz dos artigos (`jonah_edicoes/`) tem subpastas por edição
- [ ] Verificar que os nomes das subpastas seguem o padrão `vXXnYY_AAAA` (ex: `v14n1_2024`)
- [ ] Criar uma pasta nova chamada `jonah_indice/` (separada dos PDFs — só para o zip do Chroma)

> **Atenção:** a pasta `jonah_indice/` deve ser **diferente** da pasta `jonah_edicoes/`.
> Nunca misture PDFs e o arquivo `chroma_index.zip` na mesma pasta.

---

### 1.2 Pegar os IDs das pastas do Drive

Para cada pasta, abra no navegador. O ID é o trecho longo da URL:

```
https://drive.google.com/drive/folders/  1AbCdEfGhIjKlMnOpQrStUvWxYz
                                         ↑ este é o ID
```

- [ ] Copiar ID da pasta `jonah_edicoes/`  → salvar como `GDRIVE_DOCS_FOLDER_ID`
- [ ] Copiar ID da pasta `jonah_indice/`   → salvar como `GDRIVE_FOLDER_ID`

---

### 1.3 Compartilhar as pastas com a conta de serviço Google

A conta de serviço (Service Account) precisa ter acesso de **Editor** nas duas pastas.

- [ ] Abrir o arquivo de credenciais JSON do Google Cloud (o mesmo do sistema PET-Saúde)
- [ ] Copiar o campo `client_email` (parece um e-mail como `xxx@projeto.iam.gserviceaccount.com`)
- [ ] No Drive, clicar com botão direito em `jonah_edicoes/` → Compartilhar → colar o e-mail → **Editor**
- [ ] Repetir para `jonah_indice/`

---

## FASE 2 — Hugging Face Spaces

### 2.1 Criar conta e novo Space

- [ ] Acessar [huggingface.co](https://huggingface.co) e criar conta (ou usar conta existente)
- [ ] Ir em **Spaces → Create new Space**
- [ ] Preencher:
  - Name: `jonah-rag` (ou similar)
  - License: `cc-by-nc-4.0` (adequado para uso acadêmico)
  - SDK: **Streamlit**
  - Hardware: **CPU Basic** (grátis — 2 vCPU, 16 GB RAM) ← suficiente para o E5
  - Visibility: **Public** ou **Private** (sua escolha)
- [ ] Clicar em **Create Space**

> O plano **CPU Basic gratuito** já tem 16 GB RAM — muito mais confortável que o
> Streamlit Cloud (1 GB). O modelo `multilingual-e5-base` usa ~1.1 GB RAM ao carregar.

---

### 2.2 Configurar variáveis de ambiente (Secrets)

No Space criado: **Settings → Variables and Secrets → New Secret**

Criar os seguintes secrets (todos como **Secret**, não Variable):

| Nome | Valor |
|------|-------|
| `GROQ_API_KEY` | sua chave da API Groq |
| `GDRIVE_FOLDER_ID` | ID da pasta `jonah_indice/` |
| `GDRIVE_DOCS_FOLDER_ID` | ID da pasta `jonah_edicoes/` |
| `COLLECTION_NAME` | `jonah_journal` |
| `EMBED_MODEL` | `intfloat/multilingual-e5-base` |
| `RELEVANCE_THRESHOLD` | `0.35` |

Para as credenciais do Google Cloud (o JSON inteiro):

- [ ] Criar secret chamado `GCP_SERVICE_ACCOUNT_JSON`
- [ ] Colar o **conteúdo completo** do arquivo `.json` de credenciais como valor

> No `drive_sync.py`, adaptar a leitura do secret se necessário. No HF Spaces,
> os secrets ficam em variáveis de ambiente, então usar:
> ```python
> import json, os
> info = json.loads(os.getenv("GCP_SERVICE_ACCOUNT_JSON", "{}"))
> ```

---

### 2.3 Preparar os arquivos do repositório

Clonar o repositório do Space localmente:

```bash
git clone https://huggingface.co/spaces/SEU_USUARIO/jonah-rag
cd jonah-rag
```

Copiar os arquivos do projeto:

- [ ] `app_jonah.py`       → renomear para `app.py` (o HF Spaces procura `app.py` por padrão)
- [ ] `rag_jonah.py`
- [ ] `ingest_jonah.py`
- [ ] `drive_sync.py`      ← idêntico ao do sistema PET-Saúde

---

### 2.4 Criar o `requirements.txt`

```txt
streamlit>=1.35.0
langchain>=0.2.0
langchain-community>=0.2.0
langchain-chroma>=0.1.2
langchain-huggingface>=0.0.3
langchain-groq>=0.1.6
langchain-core>=0.2.0
langchain-text-splitters>=0.2.0
chromadb>=0.5.0
sentence-transformers>=3.0.0
huggingface-hub>=0.23.0
pypdf>=4.0.0
ftfy>=6.1.0
python-dotenv>=1.0.0
google-auth>=2.29.0
google-api-python-client>=2.130.0
```

- [ ] Criar `requirements.txt` com o conteúdo acima

---

### 2.5 Criar o `README.md` do Space

O HF Spaces usa o bloco YAML no topo do README para configurar o app:

```yaml
---
title: JONAH RAG
emoji: 📖
colorFrom: blue
colorTo: yellow
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
license: cc-by-nc-4.0
---

# JONAH — Journal of Nursing and Health

Consulta semântica ao acervo do periódico científico da
Faculdade de Enfermagem (FEn) da Universidade Federal de Pelotas (UFPel).

Sistema RAG com 650 artigos indexados, busca por relevância semântica
e respostas narrativas com citações automáticas em ABNT.
```

- [ ] Criar `README.md` com o bloco YAML acima

---

### 2.6 Adaptar o `drive_sync.py` para ler credenciais do HF Spaces

Localizar no `drive_sync.py` a função `get_drive_service()` e adicionar leitura
da variável de ambiente `GCP_SERVICE_ACCOUNT_JSON`:

```python
def get_drive_service():
    import json

    # 1. Tenta variável de ambiente (Hugging Face Spaces)
    json_str = os.getenv("GCP_SERVICE_ACCOUNT_JSON", "")
    if json_str:
        info = json.loads(json_str)
        creds = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
        return build('drive', 'v3', credentials=creds)

    # 2. Tenta st.secrets (Streamlit Cloud) — mantém compatibilidade
    try:
        if 'gcp_service_account' in st.secrets:
            info = st.secrets['gcp_service_account']
            creds = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
            return build('drive', 'v3', credentials=creds)
    except Exception:
        pass

    # 3. Tenta arquivo JSON local (desenvolvimento)
    json_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', 'service_account.json')
    if os.path.exists(json_path):
        creds = service_account.Credentials.from_service_account_file(json_path, scopes=SCOPES)
        return build('drive', 'v3', credentials=creds)

    raise RuntimeError("Credenciais Google Cloud não encontradas.")
```

- [ ] Aplicar a modificação acima no `drive_sync.py`

---

### 2.7 Fazer o push para o Space

```bash
git add .
git commit -m "deploy inicial JONAH RAG"
git push
```

- [ ] Push feito — aguardar o build automático no HF Spaces (1–3 minutos)
- [ ] Acessar a URL do Space e verificar se o app abre sem erros

---

## FASE 3 — Primeira indexação

Esta fase é feita **uma única vez** e pode demorar 60–90 minutos.
Após isso, cada nova edição leva apenas 2–5 minutos.

### 3.1 Executar a indexação pelo sidebar do app

- [ ] Acessar o Space no navegador
- [ ] Abrir o **sidebar** (seta à esquerda)
- [ ] Clicar em **"Sincronizar Drive e reindexar novos"**
- [ ] Aguardar — o progresso aparece no terminal do Space (aba Logs)

O que acontece internamente:
1. `drive_sync.sync_folder()` baixa os PDFs do Drive para `/tmp/raw_docs/`
2. `ingest_jonah.build_index()` processa os PDFs, gera embeddings e salva no Chroma
3. O índice é zipado e enviado de volta ao Drive (`jonah_indice/chroma_index.zip`)
4. O `indexed_files.json` registra todos os arquivos processados

---

### 3.2 Verificar o resultado

- [ ] Nos logs, procurar a linha: `✓ XXXX chunks adicionados ao índice JONAH`
- [ ] Fazer uma consulta de teste no campo de busca
- [ ] Verificar se os cards de referência aparecem com snippet e barra de relevância

---

## FASE 4 — Adicionando novas edições (fluxo recorrente)

Quando sair uma nova edição do JONAH:

### Opção A — Pelo app (recomendado)

1. Fazer upload da pasta da nova edição no Drive dentro de `jonah_edicoes/`
2. No sidebar do app, clicar **"Sincronizar Drive e reindexar novos"**
3. Apenas os PDFs novos serão baixados e indexados (~2–5 min)

### Opção B — Pelo código (avançado)

```python
from ingest_jonah import add_edition
add_edition("data/raw_docs/v15n1_2025", gdrive_folder_id="ID_DA_PASTA_INDICE")
```

---

## FASE 5 — Migração do modelo (uma vez só)

**Esta etapa só é necessária se você já tem um índice gerado com o modelo antigo
(`paraphrase-multilingual-mpnet-base-v2`). Embeddings de modelos diferentes
não podem conviver no mesmo índice Chroma.**

- [ ] No sidebar do app, clicar **"⚠️ Reindexar tudo do zero"**
- [ ] Aguardar a reindexação completa (~60–90 min)
- [ ] O índice antigo será descartado e o novo criado com `multilingual-e5-base`

---

## Referência rápida — URLs e comandos

| Item | Valor |
|------|-------|
| App no HF Spaces | `https://huggingface.co/spaces/SEU_USUARIO/jonah-rag` |
| Logs do Space | Aba **Logs** na página do Space |
| Reiniciar o Space | Settings → **Restart Space** |
| Modelo de embeddings | `intfloat/multilingual-e5-base` |
| Coleção Chroma | `jonah_journal` |
| Arquivo de registro | `data/indexed_files.json` |
| Índice no Drive | `jonah_indice/chroma_index.zip` |

---

## Solução de problemas comuns

**App abre mas não encontra o índice**
→ O índice ainda não foi gerado. Executar a Fase 3.

**Erro de credenciais Google**
→ Verificar se `GCP_SERVICE_ACCOUNT_JSON` está nos Secrets do Space e se a conta
de serviço tem acesso **Editor** nas duas pastas do Drive.

**Indexação trava ou demora muito**
→ Normal na primeira vez (650 PDFs). Verificar os Logs — se parou há mais de
30 min sem progresso, reiniciar o Space e tentar novamente (o registro incremental
garante que o progresso seja retomado de onde parou).

**Resposta em branco ou "índice não encontrado" após reinício do Space**
→ O HF Spaces usa `/tmp/` que é volátil. O app baixa automaticamente o índice do
Drive ao iniciar. Se não baixar, clicar "Sincronizar" no sidebar.

**Qualidade da busca ruim (resultados irrelevantes)**
→ Verificar se os PDFs têm texto extraível (não são scans). PDFs escaneados
precisam de OCR antes da indexação — considerar `pytesseract` ou `ocrmypdf`.
