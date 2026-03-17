import sys
print("inicio", flush=True)
try:
    from langchain_community.document_loaders import PyPDFLoader
    print("loader importado", flush=True)
    loader = PyPDFLoader('data/raw_docs/manuaisoutros/2021manualsaopaulo.pdf')
    print("loader criado", flush=True)
    docs = loader.load()
    print("docs:", len(docs), flush=True)
except Exception as e:
    print("ERRO:", e, flush=True)
    import traceback
    traceback.print_exc()
print("fim", flush=True)