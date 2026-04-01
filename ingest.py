"""
Indexador incremental para o JONAH — Journal of Nursing and Health (UFPel / FEn).

Mudanças em relação ao sistema PET-Saúde original:
  1. Modelo: multilingual-e5-base (intfloat) — otimizado para retrieval, não só similaridade.
  2. Prefixo "passage:" em cada chunk — obrigatório para o E5 funcionar corretamente.
  3. Indexação incremental via registro JSON — só processa arquivos novos.
  4. Metadados ricos: edição, volume, número, ano extraídos do nome da pasta.
  5. clear_existing=False por padrão — preserva o índice entre edições.
  6. Função add_edition() para indexar só uma pasta nova.

Estrutura esperada de pastas:
    data/raw_docs/
        v14n1_2024/
            artigo1.pdf
            artigo2.pdf
        v14n2_2024/
            ...
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import ftfy
from dotenv import load_dotenv

load_dotenv(".env.local", override=True)

BASE_DIR = Path(__file__).resolve().parent

_on_cloud       = Path("/mount/src").exists()
RAW_DIR_DEFAULT = str(BASE_DIR / "data" / "raw_docs")
DB_DIR_DEFAULT  = "/tmp/chroma_db" if _on_cloud else str(BASE_DIR / "data" / "chroma_db")
REGISTRY_FILE   = str(BASE_DIR / "data" / "indexed_files.json")

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "jonah_journal")
EMBED_MODEL     = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-base")


# ---------------------------------------------------------------------------
# Registro de arquivos já indexados
# ---------------------------------------------------------------------------

def _load_registry(path: str = REGISTRY_FILE) -> Dict[str, str]:
    p = Path(path)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_registry(registry: Dict[str, str], path: str = REGISTRY_FILE) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")


def _file_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_new_file(path: Path, raw_dir: str, registry: Dict[str, str]) -> bool:
    rel = str(path.relative_to(raw_dir))
    return registry.get(rel) != _file_hash(path)


# ---------------------------------------------------------------------------
# Metadados da estrutura de pastas
# ---------------------------------------------------------------------------

def _parse_edition_folder(folder_name: str) -> Dict[str, str]:
    """
    Extrai volume, número e ano do nome da pasta.
    Suporta: v14n1_2024, v14_n1_2024, 2024_v14n1, v.14 n.1 2024
    """
    meta: Dict[str, str] = {"edicao": folder_name}
    m = re.search(r'v\.?(\d+)[_\s\-]?n\.?(\d+)[_\s\-]?(\d{4})?', folder_name, re.I)
    if m:
        meta["volume"] = m.group(1)
        meta["numero"] = m.group(2)
        if m.group(3):
            meta["ano"] = m.group(3)
    else:
        m_ano = re.search(r'(20\d{2}|19\d{2})', folder_name)
        if m_ano:
            meta["ano"] = m_ano.group(1)
    return meta


# ---------------------------------------------------------------------------
# Carregamento de PDFs com prefixo E5
# ---------------------------------------------------------------------------

def _load_pdf(path: Path, extra_meta: Optional[Dict] = None) -> List[Any]:
    """
    Carrega PDF página a página.
    Prefixa cada chunk com 'passage:' — obrigatório para o modelo E5.
    """
    try:
        from pypdf import PdfReader
        from langchain_core.documents import Document

        reader = PdfReader(str(path))
        docs = []
        gdrive_file_id = (extra_meta or {}).get("gdrive_file_id", "")
        gdrive_link = (
            f"https://drive.google.com/file/d/{gdrive_file_id}/view"
            if gdrive_file_id else ""
        )
        base_meta = {
            "source":         str(path),
            "filename":       path.name,
            "tipo_doc":       "artigo_journal",
            "gdrive_file_id": gdrive_file_id,
            "gdrive_link":    gdrive_link,
            **(extra_meta or {}),
        }

        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = ftfy.fix_text(text)
            if not text.strip():
                continue

            prefixed = f"passage: {text}"

            docs.append(Document(
                page_content=prefixed,
                metadata={**base_meta, "page": i + 1},
            ))

        print(f"  [PDF] {path.name}: {len(docs)} páginas")
        return docs

    except Exception as e:
        print(f"  [AVISO] Erro ao carregar {path.name}: {e}")
        return []


# ---------------------------------------------------------------------------
# Carregamento incremental
# ---------------------------------------------------------------------------

def load_new_docs(
    raw_dir:     str,
    registry:    Dict[str, str],
    file_id_map: Optional[Dict[str, str]] = None,
) -> Tuple[List[Any], List[str], Dict[str, str]]:
    """
    Percorre raw_dir recursivamente.
    Subpastas de nível 1 = edições da revista.
    Só processa arquivos ausentes ou modificados no registro.
    """
    raw = Path(raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    docs: List[Any] = []
    skipped: List[str] = []
    new_registry = dict(registry)

    for p in sorted(raw.rglob("*")):
        if not p.is_file() or p.suffix.lower() != ".pdf":
            if p.is_file():
                skipped.append(str(p))
            continue

        if not _is_new_file(p, raw_dir, registry):
            continue  # já indexado e não mudou

        try:
            rel_parts = p.relative_to(raw).parts
            edition_folder = rel_parts[0] if len(rel_parts) > 1 else "sem_edicao"
        except ValueError:
            edition_folder = "sem_edicao"

        edition_meta = _parse_edition_folder(edition_folder)
        if file_id_map:
            edition_meta["gdrive_file_id"] = file_id_map.get(p.name, "")

        try:
            file_docs = _load_pdf(p, extra_meta=edition_meta)
            docs.extend(file_docs)
            rel_key = str(p.relative_to(raw_dir))
            new_registry[rel_key] = _file_hash(p)
        except Exception as e:
            skipped.append(f"{p} (erro: {e})")

    return docs, skipped, new_registry


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _chunk_docs(
    docs:          List[Any],
    chunk_size:    int = 900,
    chunk_overlap: int = 150,
) -> List[Any]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


# ---------------------------------------------------------------------------
# Embeddings E5
# ---------------------------------------------------------------------------

def _get_embeddings():
    """
    multilingual-e5-base com normalização L2.
    Para ambiente com pouca RAM, troque por 'intfloat/multilingual-e5-small'.
    """
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": os.getenv("EMBED_DEVICE", "cpu")},
        encode_kwargs={"normalize_embeddings": True},
    )


def _open_or_create_db(db_dir: str) -> Any:
    from langchain_chroma import Chroma
    db_path = Path(db_dir)
    db_path.mkdir(parents=True, exist_ok=True)
    return Chroma(
        embedding_function=_get_embeddings(),
        persist_directory=str(db_path),
        collection_name=COLLECTION_NAME,
    )


def _insert_chunks(vectordb: Any, chunks: List[Any], batch_size: int = 100) -> None:
    for i in range(0, len(chunks), batch_size):
        lote = chunks[i: i + batch_size]
        vectordb.add_documents(lote)
        print(f"  [Indexando] {min(i + batch_size, len(chunks))}/{len(chunks)} chunks")


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def build_index(
    raw_dir:          str = RAW_DIR_DEFAULT,
    db_dir:           str = DB_DIR_DEFAULT,
    chunk_size:       int = 900,
    chunk_overlap:    int = 150,
    gdrive_folder_id: str = "",
    clear_existing:   bool = False,
    registry_path:    str = REGISTRY_FILE,
) -> Tuple[int, Optional[Any], bool]:
    """
    Indexa apenas os PDFs novos ou modificados desde a última execução.

    clear_existing=True → apaga tudo e reindexa do zero (use só na migração
    do modelo antigo para o E5, já que embeddings de modelos diferentes
    não podem conviver no mesmo índice).
    """
    if _on_cloud:
        db_dir = "/tmp/chroma_db"

    db_path = Path(db_dir)

    if clear_existing and db_path.exists():
        shutil.rmtree(db_path, ignore_errors=True)
        registry = {}
        print("[AVISO] Índice apagado. Reindexando tudo do zero.")
    else:
        registry = _load_registry(registry_path)
        print(f"[Registro] {len(registry)} arquivos já indexados.")

    file_id_map: Dict[str, str] = {}
    if gdrive_folder_id:
        try:
            from drive_sync import get_file_id_map
            file_id_map = get_file_id_map(gdrive_folder_id)
            print(f"[Drive] {len(file_id_map)} IDs de arquivo obtidos.")
        except Exception as e:
            print(f"[Drive] Nao foi possivel obter IDs: {e}")

    docs, skipped, new_registry = load_new_docs(raw_dir, registry, file_id_map=file_id_map)

    if not docs:
        print("Nenhum arquivo novo. Índice já atualizado.")
        vectordb = _open_or_create_db(db_dir)
        return 0, vectordb, False

    novos = len(new_registry) - len(registry)
    print(f"[Novos] {novos} arquivo(s) → {len(docs)} páginas carregadas.")

    chunks = _chunk_docs(docs, chunk_size, chunk_overlap)
    print(f"[Chunks] {len(chunks)} trechos para indexar.")

    vectordb = _open_or_create_db(db_dir)
    _insert_chunks(vectordb, chunks)

    try:
        vectordb.persist()
    except Exception:
        pass  # persist() removido em versões recentes do chromadb — seguro ignorar

    _save_registry(new_registry, registry_path)
    print(f"[Registro] Salvo: {len(new_registry)} arquivos no total.")

    if skipped:
        print(f"[Pulados] {len(skipped)} arquivos ignorados.")

    upload_ok = False
    if gdrive_folder_id:
        try:
            from drive_sync import upload_index_to_drive
            upload_ok = upload_index_to_drive(db_dir, gdrive_folder_id)
            print(f"[Drive] Upload: {'OK' if upload_ok else 'falhou'}.")
        except Exception as e:
            print(f"[Drive] Erro: {e}")

    print(f"\n✓ {len(chunks)} chunks adicionados ao índice JONAH.")
    return len(chunks), vectordb, upload_ok


def add_edition(
    edition_folder:   str,
    db_dir:           str = DB_DIR_DEFAULT,
    gdrive_folder_id: str = "",
    registry_path:    str = REGISTRY_FILE,
) -> Tuple[int, Optional[Any], bool]:
    """
    Indexa apenas uma edição nova sem reprocessar o acervo inteiro.

    Uso:
        from ingest import add_edition
        add_edition("data/raw_docs/v15n1_2025")
    """
    return build_index(
        raw_dir=edition_folder,
        db_dir=db_dir,
        gdrive_folder_id=gdrive_folder_id,
        clear_existing=False,
        registry_path=registry_path,
    )


if __name__ == "__main__":
    n, _, _ = build_index()
    print(f"{n} chunks adicionados.")
