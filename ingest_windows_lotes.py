"""
Indexador (RAG) para o projeto PET-Saúde G10.
Versão mais robusta para Windows: embeddings em lotes, logs por etapa
e modelo menor por padrão via .env.local.
"""

from __future__ import annotations

import gc
import os
import shutil
from pathlib import Path
from typing import Any, List, Tuple, Optional

import ftfy
from dotenv import load_dotenv

load_dotenv(".env.local", override=True)

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR_DEFAULT = str(BASE_DIR / "data" / "raw_docs")
DB_DIR_DEFAULT = str(BASE_DIR / "data" / "chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "diet_knowledge")

# Mais leve para Windows; pode sobrescrever no .env.local
EMBED_MODEL = os.getenv(
    "EMBED_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)


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
        PyPDFLoader = _get_pdf_loader()
        docs = PyPDFLoader(str(path)).load()
        return _fix_encoding(docs)
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
            loaded = load_file(p)
            docs.extend(loaded)
            print(f"[LOAD] {p.name}: {len(loaded)} páginas/blocos")
        except Exception as e:
            skipped.append(f"{p} (erro: {e})")
    return docs, skipped


def _batched(items: List[Any], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield i, items[i:i + batch_size]


def build_index(
    raw_dir: str = RAW_DIR_DEFAULT,
    db_dir: str = DB_DIR_DEFAULT,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    gdrive_folder_id: str = "",
    clear_existing: bool = True,
    batch_size: int = 32,
) -> Tuple[int, Optional[Any]]:
    raw_path = Path(raw_dir)
    db_path = Path(db_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INIT] RAW_DIR={raw_path}")
    print(f"[INIT] DB_DIR={db_path}")
    print(f"[INIT] EMBED_MODEL={EMBED_MODEL}")
    print(f"[INIT] EMBED_DEVICE={os.getenv('EMBED_DEVICE', 'cpu')}")

    docs, skipped = load_all_docs(str(raw_path))
    print(f"[LOAD] Documentos carregados: {len(docs)}")
    if not docs:
        print(f"AVISO: Nenhum documento válido encontrado em '{raw_dir}'")
        return 0, None

    RecursiveCharacterTextSplitter = _get_splitter()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"[CHUNK] Total de chunks: {len(chunks)}")

    if clear_existing and db_path.exists():
        print(f"[DB] Limpando índice antigo em {db_path}")
        shutil.rmtree(db_path, ignore_errors=True)

    embeddings = _get_embeddings()
    Chroma = _get_chroma_class()

    print("[EMBED] Modelo carregado.")

    vectordb = Chroma(
        persist_directory=str(db_path),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )

    total_added = 0
    for start, batch in _batched(chunks, batch_size):
        end = start + len(batch)
        print(f"[BATCH] Processando chunks {start + 1}-{end} de {len(chunks)}")
        vectordb.add_documents(batch)
        total_added += len(batch)
        gc.collect()

    try:
        vectordb.persist()
    except Exception:
        pass

    if skipped:
        print(f"[SKIP] Arquivos ignorados/erro: {len(skipped)}")

    print(f"[OK] Sucesso! {total_added} trechos indexados.")

    if gdrive_folder_id:
        print("[Drive] Salvando índice no Google Drive...")
        try:
            from drive_sync import upload_index_to_drive
            ok = upload_index_to_drive(str(db_path), gdrive_folder_id)
            if ok:
                print("[Drive] Índice salvo com sucesso.")
            else:
                print("[Drive] Falha ao salvar índice (verifique permissões).")
        except Exception as e:
            print(f"[Drive] Erro ao salvar índice: {e}")

    return total_added, vectordb


if __name__ == "__main__":
    n, _ = build_index()
    print(f"{n} trechos indexados.")
