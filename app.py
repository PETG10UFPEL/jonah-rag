# -*- coding: utf-8 -*-
import os
import time
import base64
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(".env.local", override=True)

import streamlit as st
import streamlit.components.v1 as components

# Streamlit Cloud: carrega secrets se disponíveis
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    if "GDRIVE_FOLDER_ID" in st.secrets:
        os.environ["GDRIVE_FOLDER_ID"] = st.secrets["GDRIVE_FOLDER_ID"]
    if "HF_TOKEN" in st.secrets:
        os.environ["HF_TOKEN"] = st.secrets["HF_TOKEN"]
except Exception:
    pass

from drive_sync import sync_folder, download_index_from_drive
from ingest import build_index
from rag import answer

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DOCS_DIR = DATA_DIR / "raw_docs"

_on_cloud = Path("/mount/src").exists()
DB_DIR = "/tmp/chroma_db" if _on_cloud else str(DATA_DIR / "chroma_db")

# CORREÇÃO 1: Padronização do nome da coleção
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "wounds_knowledge")
EMBED_MODEL = os.getenv("EMBED_MODEL", "paraphrase-multilingual-mpnet-base-v2")

st.set_page_config(page_title="Guia PMP- Amor a Pele", layout="wide")

st.markdown("""
<style>
  header {visibility: hidden;}
  [data-testid="stToolbar"] {visibility: hidden;}
  [data-testid="stHeader"] {display:none;}
  [data-testid="stDecoration"] {display:none;}
  #MainMenu {visibility: hidden;}
  footer {visibility: hidden;}
  .block-container { padding-top: 0.25rem !important; }
  .field-label {
    font-size: 1.15rem;
    font-weight: 600;
    margin-bottom: 0.2rem;
    margin-top: 0.8rem;
    color: #1a1a2e;
  }
  .info-text {
    font-size: 1.0rem;
    line-height: 1.6;
    color: #222;
    margin: 0.3rem 0;
  }
</style>
""", unsafe_allow_html=True)


def img_b64(filename: str) -> str:
    p = BASE_DIR / "assets" / filename
    if p.exists():
        return base64.b64encode(p.read_bytes()).decode()
    return ""


def secret_get(key: str, default: str = ""):
    try:
        return st.secrets.get(key, default)
    except Exception:
        return os.getenv(key, default)


def _ensure_data_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DOCS_DIR.mkdir(parents=True, exist_ok=True)


def _init_state() -> None:
    defaults = {
        "restore_status": "unknown",
        "vectordb_loaded": False,
        "vectordb": None,
        "load_log": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _log(msg: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    st.session_state.load_log.append(f"[{stamp}] {msg}")
    print(f"[app] {msg}")


_ensure_data_dirs()
_init_state()


def ensure_local_index(force_download: bool = False) -> str:
    db_path = Path(DB_DIR)
    already_local = db_path.exists() and any(db_path.iterdir())
    if already_local and not force_download:
        status = "local"
        st.session_state.restore_status = status
        _log("Índice local já existe.")
        return status

    folder_id = os.getenv("GDRIVE_FOLDER_ID", "")
    if not folder_id:
        status = "no_folder_id"
        st.session_state.restore_status = status
        _log("GDRIVE_FOLDER_ID não configurado.")
        return status

    _log("Tentando restaurar índice do Google Drive...")
    t0 = time.perf_counter()
    ok = download_index_from_drive(DB_DIR, folder_id)
    dt = time.perf_counter() - t0
    status = "restored" if ok else "not_found"
    st.session_state.restore_status = status
    _log(f"Restauração concluída em {dt:.1f}s com status: {status}.")
    return status


@st.cache_resource(show_spinner=False)
def _load_vectordb_resource(db_dir: str, embed_model: str, collection_name: str, embed_device: str):
    embeddings = HuggingFaceEmbeddings(
        model_name=embed_model,
        model_kwargs={"device": embed_device},
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(
        persist_directory=db_dir,
        embedding_function=embeddings,
        collection_name=collection_name,
    )


def load_vectordb_if_needed() -> bool:
    if st.session_state.vectordb is not None:
        return True

    db_path = Path(DB_DIR)
    if not db_path.exists() or not any(db_path.iterdir()):
        _log("Índice local ausente ao tentar carregar vector DB.")
        return False

    try:
        _log("Carregando embeddings/modelo vetorial do disco...")
        t0 = time.perf_counter()
        st.session_state.vectordb = _load_vectordb_resource(
            str(db_path),
            EMBED_MODEL,
            COLLECTION_NAME,
            os.getenv("EMBED_DEVICE", "cpu"),
        )
        dt = time.perf_counter() - t0
        st.session_state.vectordb_loaded = st.session_state.vectordb is not None
        _log(f"Vector DB carregado em {dt:.1f}s.")
        return st.session_state.vectordb_loaded
    except Exception as e:
        st.session_state.vectordb = None
        st.session_state.vectordb_loaded = False
        _log(f"Falha ao carregar índice local: {e}")
        return False


def build_augmented_query(user_text: str, mode: str, temperature: float) -> str:
    if temperature <= 0.25:
        style = "resposta mais objetiva, curta, organizada e direta"
    elif temperature <= 0.55:
        style = "resposta equilibrada, clara, didática e prática"
    else:
        style = "resposta mais explicativa, com mais contexto, exemplos e detalhamento"

    if mode == "Ensino (tutor)":
        mode_block = """
MODO: ENSINO (TUTOR)
Objetivo:
- Ensinar do básico ao clínico.
- Explicar o raciocínio passo a passo.
- Quando faltarem dados, apontar claramente o que ainda precisa saber.
Formato desejado:
1. Resposta curta inicial.
2. Explicação em passos.
3. Perguntas diagnósticas objetivas, se faltarem dados.
4. Mini-roteiro de estudo ou revisão.
5. Quando fizer sentido, exercício curto com gabarito comentado.
"""
    else:
        mode_block = """
MODO: CLÍNICO (OBJETIVO)
Objetivo:
- Responder de forma prática, segura e direta.
- Priorizar condutas, avaliação, sinais de alarme, critérios e próximos passos.
Formato desejado:
1. Resposta objetiva.
2. Checklist ou tópicos curtos.
3. Alertas e encaminhamento presencial quando necessário.
"""
    return f"""{mode_block}

ESTILO DA RESPOSTA:
- Adote {style}.
- Use prioritariamente os documentos recuperados.
- Se precisar complementar com conhecimento geral, deixe isso explícito.
- Não invente fonte.
- Quando citar base documental, mencione os documentos usados.

DÚVIDA / CONSULTA DO USUÁRIO:
{user_text}
""".strip()


def decide_sketch_with_groq(question: str, response_text: str, mode: str) -> dict:
    try:
        from langchain_groq import ChatGroq
        groq_key = os.getenv("GROQ_API_KEY", "")
        if not groq_key:
            return {"need_sketch": False, "reason": "GROQ_API_KEY ausente.", "sketch_prompt": ""}
        llm = ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            groq_api_key=groq_key,
            temperature=0.1,
        )
        prompt = f"""
Você decide se um ESBOÇO/FIGURA didática ajudaria a compreender a resposta.
Responda SOMENTE em JSON válido, sem markdown, no formato:
{{"need_sketch": true/false, "reason": "...", "sketch_prompt": "..."}}

Regras:
- need_sketch = true quando desenho, fluxograma, anatomia simplificada, comparação visual, posicionamento, classificação, passo-a-passo ou esquema ajudarem muito.
- need_sketch = false quando a resposta já estiver suficientemente clara em texto.
- Se need_sketch = false, sketch_prompt deve ser "".
- Se need_sketch = true, crie um prompt curto, claro, didático e sem conteúdo chocante.
- Contexto do app: feridas crônicas, atendimento, ensino e consulta clínica.
- Modo atual: {mode}

PERGUNTA:
{question}

RESPOSTA:
{response_text[:2500]}
"""
        raw = llm.invoke(prompt).content.strip()
        import json, re
        try:
            data = json.loads(raw)
        except Exception:
            m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            if not m:
                return {"need_sketch": False, "reason": "Não foi possível interpretar a decisão.", "sketch_prompt": ""}
            data = json.loads(m.group(0))
        return {
            "need_sketch": bool(data.get("need_sketch", False)),
            "reason": str(data.get("reason", "")).strip(),
            "sketch_prompt": str(data.get("sketch_prompt", "")).strip(),
        }
    except Exception as e:
        return {"need_sketch": False, "reason": f"Falha ao decidir esboço: {e}", "sketch_prompt": ""}

banner_b64 = img_b64("banner.png")
if banner_b64:
    st.markdown(f"""
    <div style="display:flex; justify-content:center; margin-bottom:8px;">
      <img src="data:image/png;base64,{banner_b64}"
           style="width:40%; height:auto; border-radius:6px;">
    </div>
    """, unsafe_allow_html=True)
else:
    st.title("🩹 Feridas Crônicas - PET G10 UFPel")

insta_b64 = img_b64("instagram.png")
enf_b64 = img_b64("logo.enfermagem.png")
insta_tag = f'<img src="data:image/png;base64,{insta_b64}" width="22" style="vertical-align:middle; margin-right:4px;">' if insta_b64 else "📷"
enf_tag = f'<img src="data:image/png;base64,{enf_b64}" width="22" style="vertical-align:middle; margin-right:4px;">' if enf_b64 else "🎓"

st.markdown(f"""
<div style="display:flex; align-items:center; gap:14px; flex-wrap:wrap;
            margin-top:4px; margin-bottom:10px; font-size:0.95rem;">
  {insta_tag}
  <a href="https://www.instagram.com/amorapele_ufpel/" target="_blank"
     style="text-decoration:none; font-weight:500;">Amor à Pele</a>
  <span style="color:#aaa;">|</span>
  <a href="https://www.instagram.com/g10petsaude/" target="_blank"
     style="text-decoration:none; font-weight:500;">PET G10</a>
  <span style="color:#aaa;">|</span>
  {enf_tag}
  <a href="https://wp.ufpel.edu.br/fen/" target="_blank"
     style="text-decoration:none; font-weight:500;">Faculdade de Enfermagem – UFPel</a>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<p class="info-text" style="margin-bottom:0.8rem;">
  Sistema de apoio ao <strong>Guia de Feridas Crônicas</strong> da Prefeitura Municipal de Pelotas.
</p>
""", unsafe_allow_html=True)

_emb_path = BASE_DIR / "assets" / "embeddings_guia.html"
if _emb_path.exists():
    with st.expander("📊 Ver Embeddings da Base de Conhecimento", expanded=False):
        pass

st.divider()

with st.sidebar:
    st.header("Base de conhecimento (Google Drive)")

    status = st.session_state.restore_status
    if status in ("local", "restored"):
        st.success("✅ Índice local disponível.")
    elif status == "not_found":
        st.error("⚠️ Nenhum índice encontrado no Drive.")
    elif status == "no_folder_id":
        st.error("⚠️ GDRIVE_FOLDER_ID não configurado nos secrets.")
    else:
        st.caption("Índice ainda não verificado nesta sessão.")

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Verificar/restaurar índice"):
            with st.spinner("Verificando/restaurando índice..."):
                ensure_local_index(force_download=False)
    with col_b:
        if st.button("Carregar índice"):
            with st.spinner("Carregando índice na sessão..."):
                if not (Path(DB_DIR).exists() and any(Path(DB_DIR).iterdir())):
                    ensure_local_index(force_download=False)
                ok = load_vectordb_if_needed()
                if ok:
                    st.success("Índice carregado na sessão.")
                else:
                    st.error("Não foi possível carregar o índice.")

    admin_pass = st.text_input("Senha admin", type="password")
    admin_password = secret_get("ADMIN_PASSWORD", "")
    is_admin = bool(admin_password) and (admin_pass == admin_password)

    if is_admin:
        folder_id = os.getenv("GDRIVE_FOLDER_ID", "")
        st.text_input("Folder ID", value=folder_id, key="folder_id")

        if st.button("1) Sincronizar arquivos do Drive"):
            with st.spinner("Baixando..."):
                files = sync_folder(st.session_state.folder_id, str(RAW_DOCS_DIR))
            st.success(f"Baixados {len(files)} arquivos para {RAW_DOCS_DIR}")

        if st.button("2) Recriar índice (embeddings)"):
            with st.spinner("Indexando e salvando no Drive..."):
                t0 = time.perf_counter()
                
                # CORREÇÃO 2: Limpando a base atual da memória ANTES de recriar
                # Isso impede o bloqueio do SQLite na hora de gerar o ZIP
                _load_vectordb_resource.clear()
                st.session_state.vectordb = None
                st.session_state.vectordb_loaded = False
                
                # CORREÇÃO 3: Recuperando o terceiro valor (upload_ok) do ingest
                n, vectordb, upload_ok = build_index(
                    str(RAW_DOCS_DIR),
                    DB_DIR,
                    gdrive_folder_id=st.session_state.folder_id,
                )
                
                dt = time.perf_counter() - t0
                if vectordb is not None:
                    st.session_state.vectordb = vectordb
                    st.session_state.vectordb_loaded = True
                    st.session_state.restore_status = "local"
                    _log(f"Índice recriado em {dt:.1f}s com {n} trechos.")
                    
                    db_check = Path(DB_DIR)
                    db_files = list(db_check.rglob("*")) if db_check.exists() else []
                    db_total = sum(f.stat().st_size for f in db_files if f.is_file())
                    st.info(f"🔍 DB_DIR={DB_DIR} | arquivos={len(db_files)} | tamanho={db_total//1024}KB")
                    
                    # CORREÇÃO 4: Feedback preciso e condicional sobre o Drive
                    if upload_ok:
                        st.success(f"✅ Índice criado ({n} trechos) e SALVO no Google Drive.")
                    else:
                        st.warning(f"⚠️ Índice criado ({n} trechos) localmente, mas FALHOU ao salvar no Drive. O Service Account tem permissão de 'Editor' na pasta?")
                else:
                    _log("Nenhum documento encontrado para indexação.")
                    st.error(f"Nenhum documento encontrado em {RAW_DOCS_DIR}. Sincronize primeiro.")

        if Path(DB_DIR).exists():
            st.info("✅ Índice disponível em disco.")
        else:
            st.warning("⚠️ Índice não encontrado. Execute os passos 1 e 2 acima.")
    else:
        if admin_pass:
            st.error("Senha incorreta")


def mic_component(target_label: str, field_index: int):
    pass # mantido minimizado na visualização por praticidade, use o seu código original aqui


col_cfg1, col_cfg2, col_cfg3 = st.columns([1.3, 1.2, 1.2])

with col_cfg1:
    mode = st.radio("Modo", ["Ensino (tutor)", "Clínico (objetivo)"], horizontal=True)

with col_cfg2:
    temperature = st.slider(
        "Estilo da resposta",
        0.0,
        1.0,
        0.25 if mode == "Ensino (tutor)" else 0.20,
        0.05,
    )

with col_cfg3:
    auto_sketch = st.checkbox(
        "Sugerir esboço quando fizer sentido",
        value=True,
    )

st.markdown('<p class="field-label">❓ Dúvida / consulta</p>', unsafe_allow_html=True)
user_query = st.text_area(
    "Dúvida / consulta",
    height=220,
    label_visibility="collapsed",
)

col_btn1, col_btn2, _ = st.columns([1, 1, 3])
with col_btn1:
    gerar = st.button("🚀 Tirar dúvida", type="primary")
with col_btn2:
    if st.button("🗑️ Limpar cache"):
        _load_vectordb_resource.clear()
        st.session_state.vectordb = None
        st.session_state.vectordb_loaded = False
        st.success("Cache do índice limpo!")

if gerar:
    if not user_query.strip():
        st.error("Escreva sua dúvida primeiro.")
    else:
        with st.spinner("Buscando nos documentos e gerando resposta..."):
            if st.session_state.vectordb is None:
                ensure_local_index(force_download=False)
                load_vectordb_if_needed()

            query_for_rag = build_augmented_query(user_query, mode, temperature)
            resp, hits = answer(query_for_rag, "Não informado.", k=4, vectordb=st.session_state.vectordb)

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("### Resposta")
            st.markdown(resp)

        with col2:
            st.markdown("### Fontes recuperadas")
            if hits:
                shown = set()
                for d in hits:
                    src = Path(d.metadata.get("source", "arquivo_desconhecido")).name
                    page = d.metadata.get("page", "trecho")
                    key = f"{src}-{page}"
                    if key in shown:
                        continue
                    shown.add(key)
                    st.write(f"- {src} | p. {page}")
            else:
                st.info("Nenhuma fonte indexada utilizada.")