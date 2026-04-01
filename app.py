from __future__ import annotations

import os
import re
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# --------------------------------------------------
# ENV
# --------------------------------------------------

if Path(".env.local").exists():
    load_dotenv(".env.local", override=True)
else:
    load_dotenv(override=True)

for key in ["GROQ_API_KEY", "GDRIVE_FOLDER_ID", "GDRIVE_DOCS_FOLDER_ID"]:
    try:
        val = st.secrets.get(key, "")
        if val:
            os.environ[key] = val
    except Exception:
        pass

_on_cloud = Path("/mount/src").exists()
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
DB_DIR = "/tmp/chroma_db" if _on_cloud else str(BASE_DIR / "data" / "chroma_db")
RAW_DIR = str(BASE_DIR / "data" / "raw_docs")

GDRIVE_ID = os.getenv("GDRIVE_FOLDER_ID", "")
GDRIVE_DOCS = os.getenv("GDRIVE_DOCS_FOLDER_ID", "")

# --------------------------------------------------
# UTILS
# --------------------------------------------------

def img_b64(filename: str) -> str:
    import base64
    p = BASE_DIR / "img" / filename
    if p.exists():
        return base64.b64encode(p.read_bytes()).decode()
    return ""


@st.cache_data(show_spinner=False)
def _html_com_imagens_embutidas(html_path_str: str) -> str:
    import base64

    html_path = Path(html_path_str)
    html = html_path.read_text(encoding="utf-8", errors="replace")
    assets_dir = html_path.parent

    def substituir(match):
        src = match.group(1)
        if src.startswith("http") or src.startswith("data:"):
            return match.group(0)

        img_path = assets_dir / src
        if img_path.exists():
            suffix = src.lower()
            if suffix.endswith(".png"):
                mime = "image/png"
            elif suffix.endswith(".webp"):
                mime = "image/webp"
            elif suffix.endswith(".svg"):
                mime = "image/svg+xml"
            else:
                mime = "image/jpeg"

            b64 = base64.b64encode(img_path.read_bytes()).decode()
            return f'src="data:{mime};base64,{b64}"'

        return match.group(0)

    return re.sub(r'src="([^"]+)"', substituir, html)


def _render_narrative(text: str) -> str:
    return re.sub(r"\[(\d+)\]", r'<span class="cit-pill">[\1]</span>', text)


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="JONAH · Journal of Nursing and Health · UFPel",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------
# CSS
# --------------------------------------------------

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.1rem;
        padding-bottom: 2rem;
        max-width: 1280px;
    }

    .hero-wrap {
        border: 1px solid #dbe5ef;
        border-radius: 18px;
        padding: 1rem 1.1rem;
        background: linear-gradient(180deg, #ffffff 0%, #f8fbfe 100%);
        margin-bottom: 1rem;
    }

    .hero-title {
        font-size: 1.65rem;
        font-weight: 700;
        color: #163652;
        margin-bottom: .15rem;
    }

    .hero-subtitle {
        color: #486173;
        margin-bottom: .6rem;
    }

    .inst-links {
        display: flex;
        align-items: center;
        gap: 14px;
        flex-wrap: wrap;
        margin-top: 6px;
        margin-bottom: 10px;
        font-size: 0.95rem;
    }

    .inst-links a {
        text-decoration: none;
        font-weight: 500;
    }

    .inst-box {
        border-left: 4px solid #1a5f86;
        background: #f5f9fc;
        padding: .85rem .95rem;
        border-radius: 10px;
        margin: .75rem 0 .5rem 0;
        color: #314a5b;
        line-height: 1.65;
    }

    .mini-box {
        background: #fcfdfd;
        border: 1px solid #e4edf5;
        border-radius: 12px;
        padding: .8rem .9rem;
        margin-top: .6rem;
        color: #4a6272;
        line-height: 1.6;
    }

    .search-wrap {
        border: 1px solid #dbe5ef;
        border-radius: 16px;
        padding: .85rem .9rem .25rem .9rem;
        background: #ffffff;
        margin-bottom: .75rem;
    }

    .score-chip-wrap {
        display: flex;
        gap: .55rem;
        flex-wrap: wrap;
        margin: .4rem 0 1rem 0;
    }

    .score-chip {
        display: inline-flex;
        align-items: center;
        gap: .4rem;
        border: 1px solid #d7e3ee;
        border-radius: 999px;
        padding: .35rem .7rem;
        background: #fff;
        font-size: .84rem;
        color: #446072;
    }

    .dot {
        width: 9px;
        height: 9px;
        border-radius: 50%;
        display: inline-block;
    }

    .panel-title {
        font-size: 1rem;
        font-weight: 700;
        color: #163652;
        margin-bottom: .7rem;
        padding-bottom: .35rem;
        border-bottom: 2px solid #d9a441;
    }

    .answer-box {
        border: 1px solid #dbe5ef;
        border-radius: 16px;
        padding: 1rem 1rem .3rem 1rem;
        background: #fff;
        min-height: 240px;
    }

    .answer-box p {
        line-height: 1.85;
        color: #2f4757;
        margin-bottom: .9rem;
    }

    .cit-pill {
        display: inline-block;
        background: #163652;
        color: #fff;
        border-radius: 999px;
        padding: .05rem .35rem;
        font-size: .78rem;
        font-weight: 700;
        vertical-align: baseline;
    }

    .ref-card {
        border: 1px solid #dbe5ef;
        border-radius: 14px;
        overflow: hidden;
        background: #fff;
        margin-bottom: .8rem;
    }

    .ref-head {
        padding: .85rem .95rem .55rem .95rem;
        border-bottom: 1px solid #e9eff5;
    }

    .ref-title {
        font-weight: 700;
        color: #163652;
        line-height: 1.35;
    }

    .ref-meta {
        color: #678094;
        font-size: .82rem;
        margin-top: .2rem;
    }

    .ref-strip {
        padding: .45rem .95rem;
        background: #f3f8fb;
        border-bottom: 1px solid #e9eff5;
        color: #507086;
        font-size: .8rem;
        font-weight: 600;
    }

    .ref-snippet {
        padding: .75rem .95rem;
        background: #fffdf7;
        border-bottom: 1px solid #e9eff5;
        color: #4b6170;
        font-style: italic;
        line-height: 1.6;
        font-size: .9rem;
    }

    .ref-abnt {
        padding: .7rem .95rem;
        color: #5b7487;
        line-height: 1.55;
        font-size: .8rem;
    }

    .empty-box {
        text-align: center;
        padding: 2.2rem 1rem;
        color: #6f8798;
        border: 1px dashed #cfdae5;
        border-radius: 16px;
        background: #fbfdff;
    }

    iframe {
        border-radius: 12px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# HEADER
# --------------------------------------------------

banner_path = BASE_DIR / "img" / "banner_jonah.png"
if banner_path.exists():
    import base64 as _b64
    _banner_b64 = _b64.b64encode(banner_path.read_bytes()).decode()
    st.markdown(
        f'<img src="data:image/png;base64,{_banner_b64}" '
        'style="width:100%;max-width:100%;height:auto;display:block;'
        'object-fit:contain;object-position:center top;'
        'border-radius:12px;margin-bottom:.5rem;" '
        'alt="Banner JONAH">',
        unsafe_allow_html=True,
    )

insta_b64 = img_b64("instagram.png")
enf_b64 = img_b64("logo.enfermagem.png")

insta_tag = (
    f'<img src="data:image/png;base64,{insta_b64}" width="22" style="vertical-align:middle; margin-right:4px;">'
    if insta_b64 else "📷"
)
enf_tag = (
    f'<img src="data:image/png;base64,{enf_b64}" width="22" style="vertical-align:middle; margin-right:4px;">'
    if enf_b64 else "🎓"
)

hero_html = (
    '<div class="hero-wrap">'
    '<div class="hero-title">JONAH \u00b7 Journal of Nursing and Health \u00b7 UFPel</div>'
    '<div class="hero-subtitle">Sistema de consulta sem\u00e2ntica com <strong>RAG + embeddings</strong></div>'
    '<div class="inst-links">'
    + insta_tag
    + '<a href="https://www.instagram.com/amorapele_ufpel/" target="_blank">Amor \u00e0 Pele</a>'
    '<span style="color:#aaa;">|</span>'
    '<a href="https://www.instagram.com/g10petsaude/" target="_blank">PET G10</a>'
    '<span style="color:#aaa;">|</span>'
    + enf_tag
    + '<a href="https://wp.ufpel.edu.br/fen/" target="_blank">Faculdade de Enfermagem \u2013 UFPel</a>'
    '</div>'
    '<div class="inst-box">'
    'Sistema de consulta sem\u00e2ntica ao acervo do <strong>JONAH \u2014 Journal of Nursing and Health</strong>, '
    'peri\u00f3dico cient\u00edfico da Faculdade de Enfermagem (FEn) da Universidade Federal de Pelotas (UFPel), '
    'desenvolvido em parceria entre o <strong>JONAH</strong> e os projetos '
    '<strong>Amor \u00e0 Pele</strong> e <strong>PET G10 UFPel Feridas Cr\u00f4nicas</strong>, '
    'coordenados pela professora '
    '<a href="https://institucional.ufpel.edu.br/servidores/id/2858" target="_blank">Adrize Rutz Porto</a>, '
    'editora do JONAH e docente da Faculdade de Enfermagem \u2013 UFPel.'
    '</div>'
    '<div class="mini-box">'
    '<strong>Retrieval-Augmented Generation (RAG-AI)</strong> melhora a precis\u00e3o dos modelos de linguagem '
    'ao recuperar dados externos para responder perguntas, reduzindo alucina\u00e7\u00f5es.<br><br>'
    '<strong>Embeddings</strong> s\u00e3o representa\u00e7\u00f5es num\u00e9ricas de textos que capturam significado sem\u00e2ntico, '
    'permitindo recuperar conte\u00fado por similaridade de sentido, e n\u00e3o apenas por palavras-chave.'
    '</div>'
    '</div>'
)

st.markdown(hero_html, unsafe_allow_html=True)

# --------------------------------------------------
# EMBEDDINGS VIEW
# --------------------------------------------------

_emb_path = ASSETS_DIR / "embeddings_jonah.html"
if _emb_path.exists():
    import streamlit.components.v1 as components

    with st.expander("📊 Ver embeddings do acervo JONAH", expanded=False):
        html = _html_com_imagens_embutidas(str(_emb_path))
        components.html(
            html,
            height=720,
            scrolling=True,
        )

st.divider()

# --------------------------------------------------
# LOAD VECTOR DB
# --------------------------------------------------

@st.cache_resource(show_spinner="Carregando índice vetorial…")
def load_vectordb():
    db_path = Path(DB_DIR)

    if db_path.exists() and any(db_path.iterdir()):
        from rag import _get_db
        try:
            return _get_db(DB_DIR)
        except Exception as e:
            st.error(f"Erro ao abrir índice existente: {e}")
            return None

    if GDRIVE_ID:
        try:
            from drive_sync import download_index_from_drive
            ok = download_index_from_drive(DB_DIR, GDRIVE_ID)
        except Exception as e:
            st.warning(f"Não foi possível baixar o índice do Drive: {e}")
            ok = False

        if ok and db_path.exists() and any(db_path.iterdir()):
            from rag import _get_db
            try:
                return _get_db(DB_DIR)
            except Exception as e:
                st.error(f"Erro ao abrir índice baixado do Drive: {e}")
                return None

    raw_path = Path(RAW_DIR)
    pdfs = list(raw_path.rglob("*.pdf")) if raw_path.exists() else []

    if pdfs:
        try:
            from ingest import build_index
            _, vectordb, _ = build_index(
                raw_dir=RAW_DIR,
                db_dir=DB_DIR,
                gdrive_folder_id=GDRIVE_ID,
                clear_existing=False,
            )
            return vectordb
        except Exception as e:
            st.error(f"Erro ao reconstruir índice: {e}")
            return None

    return None


try:
    vectordb = load_vectordb()
except Exception as e:
    st.error(f"Erro ao carregar índice: {e}")
    vectordb = None

if vectordb is None:
    st.warning(
        "⚠️ Índice vetorial não disponível. Configure GDRIVE_FOLDER_ID nos Secrets do Space, "
        "ou use a barra lateral para sincronizar e reindexar os documentos."
    )

# --------------------------------------------------
# SEARCH
# --------------------------------------------------

st.markdown('<div class="search-wrap">', unsafe_allow_html=True)
with st.form("search_form", clear_on_submit=False):
    col_q, col_btn = st.columns([5, 1])
    with col_q:
        query = st.text_input(
            "Consulta",
            placeholder="Ex: intervenções de enfermagem para manejo da dor em prematuros",
            label_visibility="collapsed",
        )
    with col_btn:
        submitted = st.form_submit_button("Consultar", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

with st.expander("Filtros de recuperação", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        k_results = st.slider("Artigos recuperados", 3, 10, 5)
    with col2:
        ano_ini = st.number_input("Ano inicial", min_value=2011, max_value=2026, value=2011)
    with col3:
        ano_fim = st.number_input("Ano final", min_value=2011, max_value=2026, value=2026)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.markdown("### Administração do índice")
    st.caption("Use apenas para reindexar ou adicionar novas edições.")

    if st.button("🔄 Sincronizar Drive e reindexar novos"):
        if GDRIVE_DOCS:
            with st.spinner("Sincronizando artigos do Drive…"):
                try:
                    from drive_sync import sync_folder
                    novos = sync_folder(GDRIVE_DOCS, RAW_DIR, recursive=True)
                    st.success(f"{len(novos)} arquivo(s) baixado(s).")
                except Exception as e:
                    st.error(str(e))
        with st.spinner("Indexando arquivos novos…"):
            try:
                from ingest import build_index
                n, _, _ = build_index(
                    raw_dir=RAW_DIR,
                    db_dir=DB_DIR,
                    gdrive_folder_id=GDRIVE_ID,
                    clear_existing=False,
                )
                st.success(f"{n} chunks adicionados.")
                st.cache_resource.clear()
                st.rerun()
            except Exception as e:
                st.error(str(e))

    st.divider()

    if st.button("⚠️ Reindexar tudo do zero", type="secondary"):
        with st.spinner("Reindexando todo o acervo (pode demorar)…"):
            try:
                from ingest import build_index
                n, _, _ = build_index(
                    raw_dir=RAW_DIR,
                    db_dir=DB_DIR,
                    gdrive_folder_id=GDRIVE_ID,
                    clear_existing=True,
                )
                st.success(f"✓ {n} chunks indexados.")
                st.cache_resource.clear()
                st.rerun()
            except Exception as e:
                st.error(str(e))

    st.divider()
    st.caption(f"DB: `{DB_DIR}`")
    st.caption(f"Docs: `{RAW_DIR}`")
    db_ok = Path(DB_DIR).exists() and any(Path(DB_DIR).iterdir())
    st.caption(f"Índice no disco: {'✅' if db_ok else '❌'}")

# --------------------------------------------------
# QUERY EXECUTION
# --------------------------------------------------

if submitted and query.strip():
    if vectordb is None:
        st.warning("Índice não disponível. Use a barra lateral para criar ou restaurar o índice.")
        st.stop()

    with st.spinner("Consultando o acervo JONAH…"):
        from rag import answer_structured
        result = answer_structured(
            query.strip(),
            k=k_results,
            vectordb=vectordb,
            ano_ini=int(ano_ini),
            ano_fim=int(ano_fim),
        )

    st.session_state["last_result"] = result
    st.session_state["last_query"] = query.strip()

if "last_result" in st.session_state and st.session_state["last_result"]:
    result = st.session_state["last_result"]
    narrativa = result["narrativa"]
    cards = result["cards"]
    avg_score = result.get("avg_score", 0)
    total = result.get("total_hits", len(cards))

    st.markdown(
        f"""
        <div class="score-chip-wrap">
          <div class="score-chip"><span class="dot" style="background:#d39a2f"></span>Artigos recuperados <strong>{total}</strong></div>
          <div class="score-chip"><span class="dot" style="background:#1aa5a5"></span>Relevância média <strong>{avg_score}%</strong></div>
          <div class="score-chip"><span class="dot" style="background:#163652"></span>Modelo <strong>multilingual-e5-base</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        st.markdown('<div class="panel-title">Resposta elaborada</div>', unsafe_allow_html=True)

        paragraphs = [p.strip() for p in _render_narrative(narrativa).split("\n\n") if p.strip()]
        html_paragraphs = "".join(f"<p>{p}</p>" for p in paragraphs)

        st.markdown(
            f'<div class="answer-box">{html_paragraphs}</div>',
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown('<div class="panel-title">Fontes recuperadas</div>', unsafe_allow_html=True)

        for card in cards:
            fname = card.get("filename", "arquivo.pdf")
            edicao = card.get("edicao", "")
            vol = card.get("volume", "")
            num_ed = card.get("numero_ed", "")
            ano = card.get("ano", "")
            score_pct = card.get("score_pct", 0)
            snippet = card.get("snippet", "")
            abnt = card.get("abnt", "")
            source = card.get("source", "")
            n_card = card.get("numero", "")
            page = card.get("page", "")

            ed_label = f"v{vol}n{num_ed} · {ano}" if vol else edicao

            st.markdown(
                f"""
                <div class="ref-card">
                  <div class="ref-head">
                    <div class="ref-title">[{n_card}] {fname.replace('.pdf','').replace('_',' ').title()}</div>
                    <div class="ref-meta">{ed_label} · p. {page}</div>
                  </div>
                  <div class="ref-strip">Relevância: {score_pct}%</div>
                  <div class="ref-snippet">"{snippet}"</div>
                  <div class="ref-abnt"><strong>ABNT:</strong> {abnt}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            pdf_path   = Path(source)
            gdrive_link = card.get("gdrive_link", "")

            if pdf_path.exists():
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        label=f"⬇ Baixar PDF — {fname}",
                        data=f,
                        file_name=fname,
                        mime="application/pdf",
                        key=f"dl_{n_card}_{fname}",
                        use_container_width=True,
                    )
            elif gdrive_link:
                _lbl = "⬇ Baixar PDF — " + fname
                _btn = (
                    '<a href="' + gdrive_link + '" target="_blank" rel="noopener" '
                    'style="display:block;text-align:center;padding:.45rem .9rem;'
                    'background:#f0f7ff;border:1px solid #b8d4ef;border-radius:8px;'
                    'color:#185FA5;font-size:.88rem;font-weight:600;text-decoration:none;'
                    'margin-top:.3rem;">' + _lbl + '</a>'
                )
                st.markdown(_btn, unsafe_allow_html=True)

elif "last_result" not in st.session_state:
    st.markdown(
        """
        <div class="empty-box">
          Digite uma pergunta ou tema acima para consultar o acervo JONAH.<br>
          <small>A busca é semântica — não precisa usar palavras exatas.</small>
        </div>
        """,
        unsafe_allow_html=True,
    )
