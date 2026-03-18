"""
Indexador (RAG) para o projeto PET-Saúde G10.
Usa HuggingFace Embeddings locais (sem API paga, sem cota).
Modelo multilíngue: suporta PT, EN, ES e +50 línguas.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, List, Tuple, Optional

import ftfy
from dotenv import load_dotenv
load_dotenv(".env.local", override=True)

BASE_DIR = Path(__file__).resolve().parent

# No Streamlit Cloud /mount/src é read-only — usa /tmp que é gravável
_on_cloud = Path("/mount/src").exists()
RAW_DIR_DEFAULT = str(BASE_DIR / "data" / "raw_docs")
DB_DIR_DEFAULT  = "/tmp/chroma_db" if _on_cloud else str(BASE_DIR / "data" / "chroma_db")

# CORREÇÃO: Padronização do nome da coleção
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "wounds_knowledge")

# Modelo multilíngue — roda local, sem API key, sem limite de cota
EMBED_MODEL = os.getenv("EMBED_MODEL", "paraphrase-multilingual-mpnet-base-v2")


def _get_document_class():
    from langchain_core.documents import Document
    return Document


def _get_pdf_loader():
    from langchain_community.document_loaders import PyPDFLoader
    return PyPDFLoader


def _get_docx_loader():
    try:
        from langchain_community.document_loaders import UnstructuredWordDocumentLoader
        return UnstructuredWordDocumentLoader
    except Exception:
        return None


def _get_splitter():
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    return RecursiveCharacterTextSplitter


def _get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": os.getenv("EMBED_DEVICE", "cpu")},
        encode_kwargs={"normalize_embeddings": True},
    )


def _get_chroma_class():
    from langchain_chroma import Chroma
    return Chroma


def _fix_encoding(docs: List[Any]) -> List[Any]:
    """
    Corrige problemas de encoding em textos extraídos de PDFs e DOCX.
    Usa ftfy para detectar e reparar bytes mal interpretados.
    """
    for doc in docs:
        try:
            doc.page_content = ftfy.fix_text(doc.page_content)
        except Exception:
            try:
                doc.page_content = (
                    doc.page_content.encode("latin-1", errors="replace")
                    .decode("utf-8", errors="replace")
                )
            except Exception:
                pass
    return docs


def _load_pdf(path: Path) -> List[Any]:
    try:
        from pypdf import PdfReader
        from langchain_core.documents import Document

        reader = PdfReader(str(path))
        docs = []

        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = ftfy.fix_text(text)

            if text.strip():
                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": str(path),
                            "page": i + 1,
                            "filename": path.name,
                        },
                    )
                )

        print(f"[PDF] {path.name}: {len(docs)} páginas com texto")
        return docs

    except Exception as e:
        print(f"[AVISO] Erro ao carregar PDF {path}: {e}")
        return []


def _load_docx(path: Path) -> List[Any]:
    UnstructuredWordDocumentLoader = _get_docx_loader()
    if UnstructuredWordDocumentLoader is None:
        print(f"[AVISO] Pulando DOCX (loader indisponível): {path}")
        return []
    try:
        docs = UnstructuredWordDocumentLoader(str(path)).load()
        return _fix_encoding(docs)
    except Exception as e:
        print(f"[AVISO] Erro ao carregar DOCX {path}: {e}")
        return []


def load_file(path: Path) -> List[Any]:
    suf = path.suffix.lower()
    if suf == ".pdf":
        return _load_pdf(path)
    if suf == ".docx":
        return _load_docx(path)
    return []


def load_all_docs(raw_dir: str) -> Tuple[List[Any], List[str]]:
    raw = Path(raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    docs: List[Any] = []
    skipped: List[str] = []
    for p in raw.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in [".pdf", ".docx"]:
            skipped.append(str(p))
            continue
        try:
            docs.extend(load_file(p))
        except Exception as e:
            skipped.append(f"{p} (erro: {e})")
    return docs, skipped


def build_index(
    raw_dir: str = RAW_DIR_DEFAULT,
    db_dir: str = DB_DIR_DEFAULT,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
    gdrive_folder_id: str = "",
    clear_existing: bool = True,
) -> Tuple[int, Optional[Any], bool]:
    """
    Indexa os documentos localmente com HuggingFace (sem API paga, sem cota).
    Suporta documentos em português, inglês e outras línguas simultaneamente.
    Corrige automaticamente problemas de encoding em documentos PT-BR.

    gdrive_folder_id — se informado, faz upload do índice ao Drive após criar,
                       permitindo recuperar após sleep do Streamlit.
    """
    # Garante sempre /tmp/chroma_db na nuvem independente do parametro recebido
    if Path("/mount/src").exists():
        db_dir = "/tmp/chroma_db"

    raw_path = Path(raw_dir)
    db_path = Path(db_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    db_path.mkdir(parents=True, exist_ok=True)  # garante que a pasta existe antes do Chroma

    docs, skipped = load_all_docs(str(raw_path))
    if not docs:
        print(f"AVISO: Nenhum documento válido encontrado em '{raw_dir}'")
        return 0, None, False

    RecursiveCharacterTextSplitter = _get_splitter()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    # Limpa sempre para evitar DB corrompido entre sessoes
    if db_path.exists():
        shutil.rmtree(db_path, ignore_errors=True)
    db_path.mkdir(parents=True, exist_ok=True)

    embeddings = _get_embeddings()

    print(f"[OK] Gerando embeddings para {len(chunks)} chunks (modelo local)...")

    Chroma = _get_chroma_class()
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(db_path),
        collection_name=COLLECTION_NAME,
    )

    try:
        vectordb.persist()
    except Exception:
        pass

    if skipped:
        print(f"Arquivos ignorados/erro: {len(skipped)}")

    print(f"Sucesso! {len(chunks)} trechos indexados.")

    upload_ok = False
    if gdrive_folder_id:
        print("[Drive] Salvando índice no Google Drive...")
        db_path_check = Path(db_dir)
        files_in_db = list(db_path_check.rglob("*")) if db_path_check.exists() else []
        total_size = sum(f.stat().st_size for f in files_in_db if f.is_file())
        print(f"[Drive] db_dir={db_dir} | arquivos={len(files_in_db)} | tamanho={total_size//1024}KB")
        if total_size < 1024:
            print(f"[Drive] AVISO: índice parece vazio ou muito pequeno ({total_size} bytes), abortando upload.")
        else:
            try:
                from drive_sync import upload_index_to_drive
                # CORREÇÃO: Capturando o status real do envio para o Drive
                upload_ok = upload_index_to_drive(db_dir, gdrive_folder_id)
                if upload_ok:
                    print("[Drive] Índice salvo com sucesso.")
                else:
                    print("[Drive] Falha ao salvar índice (verifique permissões).")
            except Exception as e:
                print(f"[Drive] Erro ao salvar índice: {e}")

    return len(chunks), vectordb, upload_ok


if __name__ == "__main__":
    n, _, _ = build_index()
    print(f"{n} trechos indexados.")