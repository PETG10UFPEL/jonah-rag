"""
app.py — Interface Streamlit para o JONAH RAG.

Layout:
  - Coluna esquerda (60%): resposta narrativa com citações [N] destacadas
  - Coluna direita (40%): cards de referência com snippet, ABNT e botão PDF

Dependências adicionais em relação ao app.py original:
  - rag.py   (este projeto)
  - ingest.py (este projeto)
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

if Path(".env.local").exists():
    load_dotenv(".env.local", override=True)
else:
    load_dotenv(override=True)

# Segredos Streamlit Cloud
for key in ["GROQ_API_KEY", "GDRIVE_FOLDER_ID", "GDRIVE_DOCS_FOLDER_ID"]:
    try:
        val = st.secrets.get(key, "")
        if val:
            os.environ[key] = val
    except Exception:
        pass

_on_cloud   = Path("/mount/src").exists()
DB_DIR      = "/tmp/chroma_db" if _on_cloud else str(Path(__file__).parent / "data" / "chroma_db")
RAW_DIR     = str(Path(__file__).parent / "data" / "raw_docs")
GDRIVE_ID   = os.getenv("GDRIVE_FOLDER_ID", "")
GDRIVE_DOCS = os.getenv("GDRIVE_DOCS_FOLDER_ID", "")   # pasta com os PDFs dos artigos

# ---------------------------------------------------------------------------
# Configuração da página
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="JONAH · Journal of Nursing and Health · UFPel",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# CSS institucional FEn/UFPel
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,600;1,400&family=Lora:ital,wght@0,400;0,500;1,400&display=swap');

html, body, [class*="css"] { font-family: 'Lora', serif; }

.mast {
    border-top: 4px solid #1a3a5c;
    border-bottom: 3px double #c4d4e0;
    padding: .6rem 0 .5rem;
    margin-bottom: 1rem;
    display: flex; align-items: center; gap: 1rem;
}
.mast-badge {
    background: #1a3a5c; color: #fff;
    font-family: 'Playfair Display', serif;
    font-size: 1.2rem; font-weight: 600;
    padding: .25rem .6rem; letter-spacing: .05em;
}
.mast-badge small { color: #f0d08a; font-size: .6rem; display: block; letter-spacing: .12em; font-family: 'Lora', serif; font-weight: 400; }
.mast-title { font-family: 'Playfair Display', serif; font-style: italic; color: #1a3a5c; font-size: .95rem; }
.mast-right { margin-left: auto; font-size: .72rem; color: #8aacbf; text-align: right; line-height: 1.6; }

.narr-text { font-family: 'Lora', serif; font-size: .95rem; line-height: 1.9; color: inherit; }
.narr-text p { margin-bottom: .9rem; }

.cit-pill {
    display: inline-flex; align-items: center; justify-content: center;
    width: 18px; height: 18px; border-radius: 50%;
    background: #1a3a5c; color: #fff;
    font-size: .6rem; font-weight: 500;
    vertical-align: middle; margin: 0 1px;
    cursor: pointer;
}

.ref-card {
    border: 1px solid #c4d4e0;
    border-radius: 3px;
    margin-bottom: .75rem;
    font-family: 'Lora', serif;
    overflow: hidden;
}
.rc-header { padding: .6rem .75rem; border-bottom: 1px solid #e4eef5; }
.rc-num {
    display: inline-flex; align-items: center; justify-content: center;
    width: 20px; height: 20px; border-radius: 50%;
    background: #1a3a5c; color: #fff;
    font-size: .65rem; margin-right: 6px; float: left; margin-top: 2px;
}
.rc-title { font-size: .82rem; font-weight: 600; color: #1a3a5c; line-height: 1.3; margin-left: 26px; }
.rc-authors { font-style: italic; font-size: .72rem; color: #6a8ea5; margin-top: 2px; margin-left: 26px; }

.rel-strip {
    padding: .35rem .75rem;
    background: #edf4fa;
    display: flex; align-items: center; gap: .5rem;
    border-bottom: 1px solid #dde8f0;
    font-size: .68rem; color: #8aacbf; letter-spacing: .08em; text-transform: uppercase;
}
.rel-bar-outer { flex: 1; height: 5px; background: #c4d4e0; border-radius: 1px; overflow: hidden; }
.rel-bar-inner { height: 100%; background: #c9922a; border-radius: 1px; }
.rel-pct { color: #c9922a; font-weight: 500; }

.rc-snippet {
    padding: .5rem .75rem;
    font-size: .75rem; line-height: 1.6;
    color: #555; font-style: italic;
    border-bottom: 1px solid #e4eef5;
    background: #fdf6e3;
}
.rc-abnt {
    padding: .45rem .75rem;
    font-size: .68rem; line-height: 1.55; color: #5a7a95;
    border-bottom: 1px solid #e4eef5;
}
.abnt-lbl { font-size: .6rem; letter-spacing: .1em; text-transform: uppercase; color: #aabccf; display: block; margin-bottom: 2px; }

.score-chips { display: flex; gap: .5rem; flex-wrap: wrap; margin-bottom: 1rem; }
.chip {
    display: inline-flex; align-items: center; gap: 5px;
    border: 1px solid #c4d4e0; padding: 3px 9px;
    font-size: .72rem; font-family: 'Lora', serif; border-radius: 2px;
}
.chip-dot { width: 8px; height: 8px; border-radius: 50%; }

.panel-head {
    font-family: 'Playfair Display', serif;
    font-size: .85rem; color: #1a3a5c;
    border-bottom: 2px solid #c9922a;
    padding-bottom: .35rem; margin-bottom: .75rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Masthead
# ---------------------------------------------------------------------------

# ─────────────────────────────────────────────────────────────────────────────
# BLOCO INSTITUCIONAL — JONAH
# Substitui as linhas 287–354 do app.py original (PET-Saúde).
# Cole este trecho no app_jonah.py no lugar equivalente,
# logo após o banner e antes do st.divider().
# ─────────────────────────────────────────────────────────────────────────────

# Banner (mesmo mecanismo do app original — só troca o arquivo de imagem)
banner_path = BASE_DIR / "assets" / "banner_jonah.png"
if banner_path.exists():
    st.image(str(banner_path), use_container_width=True)
else:
    st.title("📖 JONAH · Journal of Nursing and Health · UFPel")

# ── Ícones institucionais ────────────────────────────────────────────────────
# Mantém os mesmos ícones do app original.
# Coloque os arquivos em assets/:
#   instagram.png       → mesmo arquivo do PET-Saúde
#   logo.enfermagem.png → mesmo arquivo do PET-Saúde
#   logo.jonah.png      → logo do journal (opcional — se não tiver, usa emoji)

insta_b64 = img_b64("instagram.png")
enf_b64   = img_b64("logo.enfermagem.png")
jonah_b64 = img_b64("logo.jonah.png")

insta_tag = (
    f'<img src="data:image/png;base64,{insta_b64}" width="22" '
    f'style="vertical-align:middle; margin-right:4px;">'
    if insta_b64 else "📷"
)
enf_tag = (
    f'<img src="data:image/png;base64,{enf_b64}" width="22" '
    f'style="vertical-align:middle; margin-right:4px;">'
    if enf_b64 else "🎓"
)
jonah_tag = (
    f'<img src="data:image/png;base64,{jonah_b64}" width="22" '
    f'style="vertical-align:middle; margin-right:4px;">'
    if jonah_b64 else "📖"
)

# ── Links institucionais ─────────────────────────────────────────────────────
st.markdown(f"""
<div style="display:flex; align-items:center; gap:14px; flex-wrap:wrap;
            margin-top:4px; margin-bottom:10px; font-size:0.95rem;">
  {jonah_tag}
  <a href="https://periodicos.ufpel.edu.br/index.php/enfermagem"
     target="_blank" style="text-decoration:none; font-weight:500;">JONAH</a>
  <span style="color:#aaa;">|</span>
  {insta_tag}
  <a href="https://www.instagram.com/amorapele_ufpel/" target="_blank"
     style="text-decoration:none; font-weight:500;">Amor à Pele</a>
  <span style="color:#aaa;">|</span>
  <a href="https://www.instagram.com/g10petsaude/" target="_blank"
     style="text-decoration:none; font-weight:500;">PET G10 Feridas Crônicas</a>
  <span style="color:#aaa;">|</span>
  {enf_tag}
  <a href="https://wp.ufpel.edu.br/fen/" target="_blank"
     style="text-decoration:none; font-weight:500;">Faculdade de Enfermagem – UFPel</a>
</div>
""", unsafe_allow_html=True)

# ── Descrição institucional ──────────────────────────────────────────────────
st.markdown("""
<p class="info-text" style="margin-bottom:0.8rem;">
  Sistema de consulta semântica ao acervo do <strong>JONAH — Journal of Nursing and Health</strong>,
  periódico científico da Faculdade de Enfermagem (FEn) da Universidade Federal de Pelotas (UFPel),
  desenvolvido em parceria entre o <strong>JONAH</strong> e os projetos
  <strong>Amor à Pele</strong> e <strong>PET G10 UFPel Feridas Crônicas</strong>,
  coordenados pela professora
  <a href="https://institucional.ufpel.edu.br/servidores/id/2858"
     target="_blank">Adrize Rutz Porto</a>,
  editora do JONAH e docente da Faculdade de Enfermagem&nbsp;&ndash;&nbsp;UFPel.
</p>
""", unsafe_allow_html=True)

# ── Explicação RAG/Embeddings (idêntica ao app original) ────────────────────
st.markdown("""
<p class="info-text" style="margin-top:0.8rem;">
  <strong>Retrieval-Augmented Generation (RAG-AI)</strong> melhora a precisão dos modelos de linguagem (LLMs)
  ao recuperar dados externos, como documentos e bases de dados, para responder perguntas,
  reduzindo as chamadas alucinações.
</p>
<p class="info-text" style="margin-top:0.2rem;">
  <strong>Embeddings</strong> são representações numéricas (vetores) de textos que capturam seu significado
  semântico, permitindo que a IA encontre informações relevantes por similaridade de sentido,
  e não apenas por palavras-chave.
</p>
<p class="info-text" style="margin-top:0.2rem;">
  Quer visualizar a representação gráfica do conhecimento indexado?
</p>
""", unsafe_allow_html=True)

# ── Visualização de embeddings (opcional — mesmo mecanismo do app original) ──
_emb_path = BASE_DIR / "assets" / "embeddings_jonah.html"
if _emb_path.exists():
    with st.expander("📊 Ver Embeddings do Acervo JONAH", expanded=False):
        _html_content = _html_com_imagens_embutidas(_emb_path)
        components.html(_html_content, height=700, scrolling=True)
else:
    st.caption("_(arquivo embeddings_jonah.html não encontrado em assets/)_")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# O que muda em relação ao app original — resumo para referência:
#
#  LINHA / ELEMENTO          ORIGINAL (PET-Saúde)        JONAH
#  ─────────────────────     ──────────────────────────  ──────────────────────────────────────
#  page_title (set_page_config)  "Guia PMP- Amor a Pele"     "JONAH · Journal of Nursing · UFPel"
#  banner_path               assets/banner.png           assets/banner_jonah.png
#  título fallback           "🩹 Feridas Crônicas…"      "📖 JONAH · Journal…"
#  1º link                   Amor à Pele (instagram)     JONAH (periodicos.ufpel.edu.br)
#  2º link                   PET G10                     Amor à Pele (instagram)
#  3º link                   FEn UFPel                   PET G10 Feridas Crônicas
#  4º link                   —                           FEn UFPel
#  Descrição                 "Sistema de apoio ao Guia   "Sistema de consulta semântica
#                             de Feridas Crônicas…"       ao acervo do JONAH…"
#  Papel da Prof. Adrize     coordenadora                coordenadora + editora do JONAH
#  Arquivo embeddings        embeddings_guia.html        embeddings_jonah.html
#  COLLECTION_NAME           wounds_knowledge            jonah_journal
#  EMBED_MODEL               paraphrase-multilingual-…   intfloat/multilingual-e5-base
# ─────────────────────────────────────────────────────────────────────────────


# ---------------------------------------------------------------------------
# Inicialização do índice
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Carregando índice vetorial…")
def load_vectordb():
    """
    Tenta carregar o índice do disco. Se não existir, tenta baixar do Drive.
    Retorna None se nenhum índice estiver disponível.
    """
    db_path = Path(DB_DIR)
    if not db_path.exists() or not any(db_path.iterdir()):
        if GDRIVE_ID:
            st.info("Baixando índice do Google Drive… aguarde.")
            try:
                from drive_sync import download_index_from_drive
                ok = download_index_from_drive(DB_DIR, GDRIVE_ID)
                if not ok:
                    return None
            except Exception as e:
                st.error(f"Erro ao baixar índice: {e}")
                return None

    db_path = Path(DB_DIR)
    if not db_path.exists() or not any(db_path.iterdir()):
        return None

    from rag import _get_db
    try:
        return _get_db(DB_DIR)
    except Exception as e:
        st.error(f"Erro ao abrir índice: {e}")
        return None


vectordb = load_vectordb()

# ---------------------------------------------------------------------------
# Barra de busca
# ---------------------------------------------------------------------------

with st.form("search_form", clear_on_submit=False):
    col_q, col_btn = st.columns([5, 1])
    with col_q:
        query = st.text_input(
            label="Consulta",
            placeholder="Ex: intervenções de enfermagem para manejo da dor em prematuros",
            label_visibility="collapsed",
        )
    with col_btn:
        submitted = st.form_submit_button("Consultar →", use_container_width=True)

# Filtros rápidos (optional)
with st.expander("Filtros", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        k_results = st.slider("Artigos recuperados", 3, 10, 5)
    with col2:
        ano_ini = st.number_input("Ano inicial", min_value=2011, max_value=2024, value=2011)
    with col3:
        ano_fim = st.number_input("Ano final", min_value=2011, max_value=2024, value=2024)

# ---------------------------------------------------------------------------
# Sidebar — administração do índice
# ---------------------------------------------------------------------------

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
                n, vdb, _ = build_index(
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
                from ingest_jonah import build_index
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

# ---------------------------------------------------------------------------
# Execução da consulta
# ---------------------------------------------------------------------------

def _render_narrative_with_pills(text: str) -> str:
    """Converte [1] [2] no texto em pills HTML coloridas."""
    def replace(m):
        n = m.group(1)
        return (
            f'<span class="cit-pill" title="Ver fonte {n}">{n}</span>'
        )
    return re.sub(r'\[(\d+)\]', replace, text)


if submitted and query.strip():
    if vectordb is None:
        st.warning("Índice não disponível. Use a barra lateral para criar ou restaurar o índice.")
        st.stop()

    with st.spinner("Consultando o acervo JONAH…"):
        from rag_jonah import answer_structured
        result = answer_structured(query.strip(), k=k_results, vectordb=vectordb)

    narrativa = result["narrativa"]
    cards     = result["cards"]
    avg_score = result["avg_score"]
    total     = result["total_hits"]

    # Score chips
    st.markdown(f"""
    <div class="score-chips">
      <div class="chip"><div class="chip-dot" style="background:#c9922a"></div>
        <span style="color:#888">Artigos recuperados</span>&nbsp;<strong>{total}</strong></div>
      <div class="chip"><div class="chip-dot" style="background:#2a8a5a"></div>
        <span style="color:#888">Relevância média</span>&nbsp;<strong>{avg_score}%</strong></div>
      <div class="chip"><div class="chip-dot" style="background:#1a3a5c"></div>
        <span style="color:#888">Modelo</span>&nbsp;<strong>multilingual-e5-base</strong></div>
    </div>
    """, unsafe_allow_html=True)

    col_narr, col_refs = st.columns([3, 2], gap="large")

    # ── Coluna esquerda: narrativa ──
    with col_narr:
        st.markdown('<div class="panel-head">Resposta elaborada</div>', unsafe_allow_html=True)
        narr_html = _render_narrative_with_pills(narrativa)
        # Converte quebras de parágrafo em <p>
        paragraphs = [p.strip() for p in narr_html.split("\n\n") if p.strip()]
        html_paragraphs = "".join(f"<p>{p}</p>" for p in paragraphs)
        st.markdown(
            f'<div class="narr-text">{html_paragraphs}</div>',
            unsafe_allow_html=True,
        )

    # ── Coluna direita: cards de referência ──
    with col_refs:
        st.markdown('<div class="panel-head">Fontes recuperadas</div>', unsafe_allow_html=True)

        for card in cards:
            n          = card["numero"]
            fname      = card["filename"]
            edicao     = card["edicao"]
            vol        = card["volume"]
            num_ed     = card["numero_ed"]
            ano        = card["ano"]
            score_pct  = card["score_pct"]
            snippet    = card["snippet"]
            abnt       = card["abnt"]
            source     = card["source"]

            ed_label = f"v{vol}n{num_ed} · {ano}" if vol else edicao

            # Barra de relevância HTML
            bar_html = f"""
            <div class="ref-card">
              <div class="rc-header">
                <div class="rc-num">{n}</div>
                <div class="rc-title">{fname.replace('.pdf','').replace('_',' ').title()}</div>
                <div class="rc-authors">{ed_label} · p. {card['page']}</div>
              </div>
              <div class="rel-strip">
                Relevância
                <div class="rel-bar-outer">
                  <div class="rel-bar-inner" style="width:{score_pct}%"></div>
                </div>
                <span class="rel-pct">{score_pct}%</span>
              </div>
              <div class="rc-snippet">"{snippet}"</div>
              <div class="rc-abnt">
                <span class="abnt-lbl">ABNT</span>
                {abnt}
              </div>
            </div>
            """
            st.markdown(bar_html, unsafe_allow_html=True)

            # Botão de download do PDF (Streamlit nativo para funcionar na nuvem)
            pdf_path = Path(source)
            if pdf_path.exists():
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        label=f"⬇ Baixar PDF — {fname}",
                        data=f,
                        file_name=fname,
                        mime="application/pdf",
                        key=f"dl_{n}_{fname}",
                        use_container_width=True,
                    )
            else:
                st.caption(f"📄 {fname} · {ed_label}")

elif not submitted:
    # Estado inicial — mostra instrução
    st.markdown("""
    <div style="text-align:center;padding:2.5rem 1rem;color:#8aacbf;font-family:'Lora',serif;font-style:italic">
      Digite uma pergunta ou tema acima para consultar os 650 artigos do acervo JONAH.<br>
      <small style="font-size:.78rem">A busca é semântica — não precisa usar palavras exatas.</small>
    </div>
    """, unsafe_allow_html=True)
