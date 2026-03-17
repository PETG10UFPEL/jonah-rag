"""
Módulo de Recuperação e Resposta (RAG) para o projeto PET-Saúde G10.
Modo híbrido: prioriza documentos indexados, complementa com conhecimento geral.
Embeddings: HuggingFace local (sem API paga).
LLM: Groq (gratuito, rápido, suporte a português).

Versão corrigida para maior robustez no Windows:
- evita imports pesados no carregamento do módulo;
- adia a criação de embeddings/DB/LLM até o momento de uso;
- mantém cache para não recriar objetos desnecessariamente.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any, List, Optional, Tuple

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
    st = None  # type: ignore

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
BASE_DIR = Path(__file__).resolve().parent
DB_DIR_DEFAULT = str(BASE_DIR / "data" / "chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "diet_knowledge")
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.4"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "paraphrase-multilingual-mpnet-base-v2")

SYSTEM_RULES = """Você é um assistente de planejamento alimentar clínico do projeto PET-Saúde G10, \
especialista em nutrição clínica e dietoterapia.

MODO DE OPERAÇÃO HÍBRIDO:
1. PRIORIDADE MÁXIMA: Use os trechos do CONTEXTO DOS DOCUMENTOS como base principal da resposta.
   - Sempre cite as fontes no formato [Nome do Arquivo | Página] quando usar o contexto.
2. COMPLEMENTAÇÃO: Se o contexto não cobrir completamente a situação clínica do paciente,
   complemente com seu conhecimento em nutrição clínica baseado em evidências.
   - Neste caso, indique claramente: "(conhecimento geral - não consta nos documentos indexados)"
3. TRANSPARÊNCIA: Sempre deixe claro o que veio dos documentos e o que veio do conhecimento geral.
4. NUNCA invente fontes ou cite documentos que não estejam no contexto fornecido.
5. Seja objetivo e organize a resposta em tópicos/checklists.
6. Adapte sempre ao perfil específico do paciente informado.
"""


def _import_hf_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings


def _import_chroma():
    from langchain_chroma import Chroma

    return Chroma


def _import_chatgroq():
    from langchain_groq import ChatGroq

    return ChatGroq


@functools.lru_cache(maxsize=1)
def _get_embeddings() -> Any:
    """Embeddings locais — sem API key, sem cota. Cacheado para evitar recarregar o modelo."""
    HuggingFaceEmbeddings = _import_hf_embeddings()
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": os.getenv("EMBED_DEVICE", "cpu")},
        encode_kwargs={"normalize_embeddings": True},
    )


@functools.lru_cache(maxsize=4)
def _get_db_from_disk(db_dir: str = DB_DIR_DEFAULT) -> Any:
    """Carrega um índice já persistido em disco. Cacheado para evitar reabrir a cada chamada."""
    Chroma = _import_chroma()
    return Chroma(
        persist_directory=db_dir,
        embedding_function=_get_embeddings(),
        collection_name=COLLECTION_NAME,
    )


def answer(
    question: str,
    patient_summary: str,
    k: int = 5,
    vectordb: Optional[Any] = None,
) -> Tuple[str, List[Any]]:
    """
    Busca nos documentos e gera resposta híbrida via Groq.

    Parâmetros:
      vectordb  – instância Chroma já criada (session_state). Se None,
                  tenta carregar do disco (modo local).
    Retorna (texto_da_resposta, lista_de_documentos_encontrados).
    """
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise RuntimeError(
            "GROQ_API_KEY não encontrada. Configure em .streamlit/secrets.toml."
        )

    if vectordb is not None:
        db = vectordb
    else:
        db_path = Path(DB_DIR_DEFAULT)
        if not db_path.exists() or not any(db_path.iterdir()):
            return (
                "⚠️ Índice não encontrado. No Streamlit Cloud o índice precisa ser "
                "criado na sessão atual: use a sidebar para **Sincronizar** e depois "
                "**Recriar índice** antes de gerar uma resposta.",
                [],
            )
        db = _get_db_from_disk()

    full_query = f"Paciente: {patient_summary}\nPergunta: {question}"
    hits_with_score = db.similarity_search_with_relevance_scores(full_query, k=k)

    hits = [doc for doc, score in hits_with_score if score >= RELEVANCE_THRESHOLD]
    all_hits = [doc for doc, _ in hits_with_score]

    if hits_with_score:
        context_parts = []
        for doc, score in hits_with_score:
            source = os.path.basename(doc.metadata.get("source", "desconhecido"))
            page = doc.metadata.get("page", "-")
            content = doc.page_content.strip()
            relevance = "alta" if score >= RELEVANCE_THRESHOLD else "baixa"
            context_parts.append(
                f"FONTE: {source} (p. {page}) [relevância: {relevance}]\nCONTEÚDO: {content}"
            )
        context_text = "\n\n---\n\n".join(context_parts)
        context_note = "" if hits else (
            "\n⚠️ NOTA: Os documentos indexados têm baixa relevância para esta consulta. "
            "Use seu conhecimento clínico para complementar, indicando claramente o que é conhecimento geral."
        )
    else:
        context_text = "Nenhum documento relevante encontrado no índice."
        context_note = (
            "\n⚠️ NOTA: Não há documentos indexados para esta consulta. "
            "Responda com seu conhecimento clínico em nutrição, indicando que a resposta é baseada em conhecimento geral."
        )

    ChatGroq = _import_chatgroq()
    llm = ChatGroq(
        model=GROQ_MODEL,
        groq_api_key=groq_key,
        temperature=0.2,
    )

    user_message = f"""DADOS DO PACIENTE:
{patient_summary}

PERGUNTA DO USUÁRIO:
{question}

CONTEXTO DOS DOCUMENTOS:
{context_text}
{context_note}
"""

    response = llm.invoke([
        {"role": "system", "content": SYSTEM_RULES},
        {"role": "user", "content": user_message},
    ])

    return response.content, all_hits
