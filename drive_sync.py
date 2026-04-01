"""
drive_sync.py — Sincronização com Google Drive para o JONAH RAG.
Corrigido para contornar restrição de cota de Service Accounts em contas pessoais.
"""

import os
import json
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload

if Path('.env.local').exists():
    load_dotenv('.env.local', override=True)
else:
    load_dotenv(override=True)

SCOPES = ['https://www.googleapis.com/auth/drive']
FOLDER_MIME    = 'application/vnd.google-apps.folder'
INDEX_ZIP_NAME = 'chroma_index.zip'

EXCLUDED_DIR_NAMES = {'_chroma_index', 'indice_backup', 'index_backup', 'chroma_db', 'chroma_index', '__pycache__'}
ALLOWED_EXTENSIONS = {'.pdf', '.docx'}

def get_drive_service():
    json_str = os.getenv("GCP_SERVICE_ACCOUNT_JSON", "")
    if json_str:
        info = json.loads(json_str)
        creds = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
        return build('drive', 'v3', credentials=creds)
    
    try:
        if 'gcp_service_account' in st.secrets:
            info = st.secrets['gcp_service_account']
            creds = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
            return build('drive', 'v3', credentials=creds)
    except Exception:
        pass

    json_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', 'service_account.json')
    if os.path.exists(json_path):
        creds = service_account.Credentials.from_service_account_file(json_path, scopes=SCOPES)
        return build('drive', 'v3', credentials=creds)

    raise RuntimeError("Credenciais Google Cloud não encontradas.")

def _list_children(service, folder_id: str):
    q = f"'{folder_id}' in parents and trashed = false"
    items = []
    page_token = None
    while True:
        resp = service.files().list(
            q=q,
            fields='nextPageToken, files(id, name, mimeType, size)',
            pageToken=page_token,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
        items.extend(resp.get('files', []))
        page_token = resp.get('nextPageToken')
        if not page_token: break
    return items

def _download_file(service, file_id: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = service.files().get_media(fileId=file_id)
    with open(dest, 'wb') as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()

def _sync_folder_recursive(service, folder_id: str, out_root: Path, rel: Path = Path('.')) -> list:
    downloaded = []
    items = _list_children(service, folder_id)
    for it in items:
        name, mime, it_id = it['name'], it.get('mimeType', ''), it['id']
        if mime == FOLDER_MIME:
            if name.strip().lower() in EXCLUDED_DIR_NAMES: continue
            downloaded.extend(_sync_folder_recursive(service, it_id, out_root, rel / name))
        elif Path(name).suffix.lower() in ALLOWED_EXTENSIONS:
            dest = out_root / rel / name
            if not dest.exists():
                _download_file(service, it_id, dest)
            downloaded.append({'path': dest, 'gdrive_file_id': it_id, 'name': name})
    return downloaded

def _list_all_files(service, folder_id: str) -> list:
    result = []
    items = _list_children(service, folder_id)
    for it in items:
        if it.get('mimeType') == FOLDER_MIME:
            if it['name'].strip().lower() not in EXCLUDED_DIR_NAMES:
                result.extend(_list_all_files(service, it['id']))
        else:
            result.append(it)
    return result

def get_file_id_map(folder_id: str) -> dict:
    try:
        service = get_drive_service()
        items = _list_all_files(service, folder_id)
        return {it['name']: it['id'] for it in items if Path(it['name']).suffix.lower() in ALLOWED_EXTENSIONS}
    except Exception as e:
        print(f'Erro ao listar arquivos: {e}')
        return {}

def sync_folder(folder_id: str, out_dir: str, recursive: bool = True) -> list:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        service = get_drive_service()
        return _sync_folder_recursive(service, folder_id, out) if recursive else []
    except Exception as e:
        print(f'Erro na sincronizacao: {e}')
        return []


def upload_index_to_drive(db_dir: str, gdrive_folder_id: str) -> bool:
    import tempfile, zipfile
    if Path("/mount/src").exists(): db_dir = "/tmp/chroma_db"
    db_path = Path(db_dir)
    if not db_path.exists(): return False

    files_to_upload = [f for f in db_path.rglob('*') if f.is_file()]
    if not files_to_upload: return False

    try:
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
            zip_path = Path(tmp.name)
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file_path in files_to_upload:
                zf.write(file_path, file_path.relative_to(db_path))

        service = get_drive_service()
        q = f"'{gdrive_folder_id}' in parents and name = '{INDEX_ZIP_NAME}' and trashed = false"
        resp = service.files().list(q=q, fields='files(id)', supportsAllDrives=True, includeItemsFromAllDrives=True).execute()
        existing = resp.get('files', [])

        media = MediaFileUpload(str(zip_path), mimetype='application/zip', resumable=True)

        if existing:
            # CORREÇÃO: Removido addParents para usar a cota do dono do arquivo (você) [cite: 150]
            service.files().update(
                fileId=existing[0]['id'],
                media_body=media,
                supportsAllDrives=True
            ).execute()
            print(f"[Drive] Conteúdo atualizado no arquivo existente.")
        else:
            service.files().create(
                body={'name': INDEX_ZIP_NAME, 'parents': [gdrive_folder_id]},
                media_body=media,
                fields='id',
                supportsAllDrives=True,
            ).execute()
            print(f"[Drive] Novo arquivo criado.")

        zip_path.unlink(missing_ok=True)
        return True
    except Exception as e:
        print(f'[ERRO] Falha ao salvar índice: {e}')
        return False

def download_index_from_drive(db_dir: str, gdrive_folder_id: str) -> bool:
    import tempfile, zipfile
    try:
        service = get_drive_service()
        q = f"'{gdrive_folder_id}' in parents and name = '{INDEX_ZIP_NAME}' and trashed = false"
        resp = service.files().list(q=q, fields='files(id)', supportsAllDrives=True, includeItemsFromAllDrives=True).execute()
        files = resp.get('files', [])
        if not files: return False

        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
            zip_path = Path(tmp.name)
        _download_file(service, files[0]['id'], zip_path)
        
        db_path = Path(db_dir)
        db_path.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(db_path)
        zip_path.unlink(missing_ok=True)
        return True
    except Exception as e:
        print(f'[ERRO] Falha ao baixar índice: {e}')
        return False

def index_exists_on_drive(gdrive_folder_id: str) -> bool:
    try:
        service = get_drive_service()
        q = f"'{gdrive_folder_id}' in parents and name = '{INDEX_ZIP_NAME}' and trashed = false"
        resp = service.files().list(q=q, fields='files(id)', supportsAllDrives=True, includeItemsFromAllDrives=True).execute()
        return len(resp.get('files', [])) > 0
    except Exception: return False