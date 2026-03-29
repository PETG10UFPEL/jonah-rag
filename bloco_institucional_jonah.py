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
