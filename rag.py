"""
Módulo RAG para o JONAH — Journal of Nursing and Health (UFPel / FEn).

Diferenças em relação ao rag.py original (PET-Saúde):
  1. Prefixo "query:" na consulta — par obrigatório com "passage:" do ingest E5.
  2. Prompt estruturado: Groq gera resposta narrativa com citações [1][2]...
     que o front-end mapeia para os cards de referência.
  3. Retorna dict rico com resposta, fontes, scores e metadados de edição.
  4. Função answer_structured() para o app.py montar os cards automaticamente.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv(".env.local", override=True)

try:
    import streamlit as st
    try:
        groq_secret = st.secrets.get("GROQ_API_KEY", "")
        if groq_secret:
            os.environ["GROQ_API_KEY"] = groq_secret
    except Exception:
        pass
except Exception:
    st = None

GROQ_MODEL          = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
BASE_DIR            = Path(__file__).resolve().parent
_on_cloud           = Path("/mount/src").exists()
DB_DIR_DEFAULT      = "/tmp/chroma_db" if _on_cloud else str(BASE_DIR / "data" / "chroma_db")
COLLECTION_NAME     = os.getenv("COLLECTION_NAME", "jonah_journal")
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.35"))
EMBED_MODEL         = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-base")

# ---------------------------------------------------------------------------
# Prompt — instrui o Groq a gerar resposta com citações numeradas
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """Você é o assistente acadêmico do JONAH — Journal of Nursing and Health,
periódico científico da Faculdade de Enfermagem (FEn) da Universidade Federal de Pelotas (UFPel).

Seu papel é responder perguntas de pesquisadores, estudantes e leitores com base
nos artigos do acervo do JONAH, produzindo respostas narrativas, elaboradas e rigorosas.

REGRAS OBRIGATÓRIAS:
1. Redija a resposta em português, em linguagem acadêmica clara e fluida.
2. Use EXCLUSIVAMENTE os trechos fornecidos no CONTEXTO como base factual.
3. Ao usar informação de um trecho, insira imediatamente a citação numérica [N]
   onde N é o número do trecho (1, 2, 3...) conforme listado no contexto.
4. A mesma fonte pode ser citada mais de uma vez com o mesmo número [N].
5. NÃO invente informações. Se o contexto não cobrir algum ponto, diga explicitamente:
   "(informação não encontrada no acervo JONAH)".
6. NÃO use conhecimento geral sem identificar com: "(conhecimento geral — não consta no acervo)".
7. Organize a resposta em parágrafos temáticos, sem listas ou tópicos.
8. Ao final, NÃO repita as referências — elas serão exibidas automaticamente pelo sistema.

FORMATO DA RESPOSTA: apenas o texto narrativo com as citações [N] embutidas.
"""

# ---------------------------------------------------------------------------
# Embeddings e DB
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _get_embeddings() -> Any:
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": os.getenv("EMBED_DEVICE", "cpu")},
        encode_kwargs={"normalize_embeddings": True},
    )


@functools.lru_cache(maxsize=4)
def _get_db(db_dir: str = DB_DIR_DEFAULT) -> Any:
    from langchain_chroma import Chroma
    return Chroma(
        persist_directory=db_dir,
        embedding_function=_get_embeddings(),
        collection_name=COLLECTION_NAME,
    )


# ---------------------------------------------------------------------------
# Montagem do contexto para o prompt
# ---------------------------------------------------------------------------

def _build_context(hits_with_score: List[Tuple[Any, float]]) -> str:
    """
    Monta bloco de contexto numerado para o prompt.
    Remove o prefixo 'passage:' antes de enviar ao LLM.
    """
    parts = []
    for i, (doc, score) in enumerate(hits_with_score, 1):
        source   = os.path.basename(doc.metadata.get("source", "desconhecido"))
        page     = doc.metadata.get("page", "-")
        edicao   = doc.metadata.get("edicao", "")
        volume   = doc.metadata.get("volume", "")
        numero   = doc.metadata.get("numero", "")
        ano      = doc.metadata.get("ano", "")
        filename = doc.metadata.get("filename", source)

        content = doc.page_content
        if content.startswith("passage: "):
            content = content[len("passage: "):]

        ref_str = f"v.{volume} n.{numero} ({ano})" if volume else edicao
        parts.append(
            f"[{i}] FONTE: {filename} | {ref_str} | p. {page} | relevância: {score:.2f}\n"
            f"TRECHO: {content.strip()}"
        )
    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Estrutura de retorno
# ---------------------------------------------------------------------------

def _build_source_cards(hits_with_score: List[Tuple[Any, float]]) -> List[Dict]:
    """
    Monta lista de dicts com tudo que o app.py precisa para exibir os cards.
    """
    cards = []
    for i, (doc, score) in enumerate(hits_with_score, 1):
        meta     = doc.metadata
        filename = meta.get("filename", os.path.basename(meta.get("source", "")))
        edicao   = meta.get("edicao", "")
        volume   = meta.get("volume", "")
        numero   = meta.get("numero", "")
        ano      = meta.get("ano", "")
        page     = meta.get("page", "-")
        source         = meta.get("source", "")
        gdrive_file_id = meta.get("gdrive_file_id", "")
        gdrive_link    = (
            meta.get("gdrive_link", "") or
            (f"https://drive.google.com/file/d/{gdrive_file_id}/view" if gdrive_file_id else "")
        )

        snippet = doc.page_content
        if snippet.startswith("passage: "):
            snippet = snippet[len("passage: "):]
        snippet = snippet.strip()[:350] + ("…" if len(snippet) > 350 else "")

        titulo_proxy = filename.replace(".pdf", "").replace("_", " ").title()
        abnt = (
            f"{titulo_proxy}. "
            f"<em>Journal of Nursing and Health</em>, "
            f"Pelotas"
            + (f", v. {volume}, n. {numero}, p. -, {ano}." if volume else f", {edicao}.")
        )

        cards.append({
            "numero":    i,
            "filename":  filename,
            "edicao":    edicao,
            "volume":    volume,
            "numero_ed": numero,
            "ano":       ano,
            "page":      page,
            "score":     round(score, 3),
            "score_pct": round(score * 100),
            "snippet":   snippet,
            "abnt":      abnt,
            "source":         source,
            "gdrive_file_id": gdrive_file_id,
            "gdrive_link":    gdrive_link,
        })
    return cards


# ---------------------------------------------------------------------------
# Função principal
# ---------------------------------------------------------------------------

def answer_structured(
    question: str,
    k:        int = 5,
    vectordb: Optional[Any] = None,
    ano_ini:  Optional[int] = None,
    ano_fim:  Optional[int] = None,
) -> Dict:
    """
    Busca no índice JONAH e gera resposta narrativa com citações [N].

    Retorna dict:
    {
        "narrativa":  str,          # texto com [1][2]... para o front mapear
        "cards":      List[dict],   # metadados + snippet + abnt por fonte
        "total_hits": int,
        "avg_score":  float,
        "query":      str,
    }
    """
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise RuntimeError("GROQ_API_KEY não encontrada.")

    if vectordb is not None:
        db = vectordb
    else:
        db_path = Path(DB_DIR_DEFAULT)
        if not db_path.exists() or not any(db_path.iterdir()):
            return {
                "narrativa": "⚠️ Índice não encontrado. Recrie o índice antes de consultar.",
                "cards": [], "total_hits": 0, "avg_score": 0.0, "query": question,
            }
        db = _get_db()

    # Prefixo "query:" obrigatório para o multilingual-e5
    query_prefixed = f"query: {question}"
    hits_with_score = db.similarity_search_with_relevance_scores(query_prefixed, k=k)

    def _year_ok(doc) -> bool:
        if ano_ini is None and ano_fim is None:
            return True
        try:
            ano = int(doc.metadata.get("ano", 0))
        except (ValueError, TypeError):
            return True
        if ano == 0:
            return True
        if ano_ini is not None and ano < ano_ini:
            return False
        if ano_fim is not None and ano > ano_fim:
            return False
        return True

    hits_filtered = [(d, s) for d, s in hits_with_score if _year_ok(d)]
    if not hits_filtered:
        hits_filtered = hits_with_score

    relevant = [(d, s) for d, s in hits_filtered if s >= RELEVANCE_THRESHOLD]
    to_use   = relevant if relevant else hits_filtered  # fallback: usa tudo

    if not to_use:
        return {
            "narrativa": "Nenhum trecho relevante encontrado no acervo para esta consulta.",
            "cards": [], "total_hits": 0, "avg_score": 0.0, "query": question,
        }

    context = _build_context(to_use)
    cards   = _build_source_cards(to_use)
    avg_score = round(sum(s for _, s in to_use) / len(to_use) * 100)

    from langchain_groq import ChatGroq
    llm = ChatGroq(model=GROQ_MODEL, groq_api_key=groq_key, temperature=0.15)

    user_msg = f"""PERGUNTA:
{question}

CONTEXTO DOS ARTIGOS JONAH (use os números [N] para citar):
{context}
"""
    try:
        response = llm.invoke([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ])
        narrativa = response.content

    except Exception as e:
        err = str(e).lower()
        import re as _re
        _wait = _re.search(
            r"(?:try again in|please wait|retry after|wait)\s*([\d]+m[\d]+(?:\.[\d]+)?s|[\d]+(?:\.[\d]+)?s|[\d]+m)",
            str(e), _re.I
        )
        _wait_str = _wait.group(1) if _wait else ""

        if any(t in err for t in ["token", "context_length", "context length",
                                   "maximum context", "too long"]):
            narrativa = (
                "A consulta recuperou trechos demais para processar de uma vez. "
                "Tente reduzir o n\u00famero de artigos recuperados (filtro \"Artigos recuperados\") "
                "ou reformule a pergunta de forma mais espec\u00edfica para obter uma resposta mais focada."
            )
        elif any(t in err for t in ["rate_limit", "rate limit", "429"]):
            if _wait_str:
                narrativa = (
                    "O limite de requisi\u00e7\u00f5es foi atingido. "
                    f"O servi\u00e7o estar\u00e1 dispon\u00edvel novamente em {_wait_str}. "
                    "Aguarde e tente novamente."
                )
            else:
                narrativa = (
                    "O limite de requisi\u00e7\u00f5es foi atingido. "
                    "Aguarde alguns instantes e tente novamente."
                )
        elif "groq" in err or "api" in err or "503" in err or "502" in err:
            if _wait_str:
                narrativa = (
                    "O servi\u00e7o de intelig\u00eancia artificial est\u00e1 temporariamente indispon\u00edvel. "
                    f"Previs\u00e3o de retorno em {_wait_str}. "
                    "Tente novamente em instantes."
                )
            else:
                narrativa = (
                    "O servi\u00e7o de intelig\u00eancia artificial est\u00e1 temporariamente indispon\u00edvel. "
                    "Aguarde alguns instantes e tente novamente."
                )
        else:
            narrativa = (
                "N\u00e3o foi poss\u00edvel gerar a resposta devido a um erro inesperado. "
                "Tente novamente em alguns instantes."
            )

    return {
        "narrativa":  narrativa,
        "cards":      cards,
        "total_hits": len(to_use),
        "avg_score":  avg_score,
        "query":      question,
    }


# Mantém compatibilidade com chamadas legadas
def answer(
    question:        str,
    patient_summary: str = "",
    k:               int = 5,
    vectordb:        Optional[Any] = None,
) -> Tuple[str, List[Any]]:
    result = answer_structured(question, k=k, vectordb=vectordb)
    docs   = [type('D', (), {'metadata': c, 'page_content': c['snippet']})()
              for c in result["cards"]]
    return result["narrativa"], docs
