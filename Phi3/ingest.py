# ingest_qa.py
import os
import re
from PyPDF2 import PdfReader
import chromadb
from chromadb.utils import embedding_functions
from config import DB_PATH, DATA_PATH

# ----------------------------
# Constants
# ----------------------------
MAX_CHUNK_WORDS = 100  # truncate extremely long answers
OVERLAP = 2

# ----------------------------
# Utility functions
# ----------------------------
def clean_text(text: str) -> str:
    """Normalize spaces and remove junk."""
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text

def extract_qa_chunks(text: str) -> list[str]:
    """
    Extract each Q/A pair as one chunk.
    - Q: ...  
    - A: ...  
    Handles multi-line answers.
    """
    chunks = []
    lines = text.split("\n")
    current_q, current_a = None, None

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("Q:"):
            if current_q and current_a:
                chunks.append(f"{current_q} {current_a}")
            current_q = line
            current_a = None
        elif line.startswith("A:"):
            current_a = line
        else:
            if current_a is not None:
                current_a += " " + line  # multiline answer

    if current_q and current_a:
        chunks.append(f"{current_q} {current_a}")

    # truncate very long answers if needed
    final_chunks = []
    for c in chunks:
        words = c.split()
        if len(words) > MAX_CHUNK_WORDS:
            final_chunks.append(" ".join(words[:MAX_CHUNK_WORDS]))
        else:
            final_chunks.append(c)
    return [clean_text(c) for c in final_chunks if c.strip()]

def load_documents(folder_path=DATA_PATH) -> dict[str, str]:
    docs = {}
    idx = 0

    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        if fname.endswith(".pdf"):
            reader = PdfReader(fpath)
            text = "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])
        elif fname.endswith(".txt"):
            with open(fpath, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            continue

        # Choose Q/A-aware chunking if dataset contains Q/A
        if "Q:" in text:
            chunks = extract_qa_chunks(text)
        else:
            # fallback for plain text
            units = [u.strip() for u in text.split("\n") if u.strip()]
            chunks = units

        for chunk in chunks:
            docs[f"{fname}_{idx}"] = chunk
            idx += 1

    return docs

def ingest(folder_path=DATA_PATH, db_path=DB_PATH, collection_name="admissions"):
    client = chromadb.PersistentClient(path=db_path)
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedder
    )

    docs = load_documents(folder_path)
    for doc_id, text in docs.items():
        collection.add(documents=[text], ids=[doc_id])

    print(f"✅ Ingested {len(docs)} Q/A chunks into {collection_name}.")


def inspect_all():
    client = chromadb.PersistentClient(path=DB_PATH)
    collections = client.list_collections()
    if not collections:
        print("⚠️ No collections found in DB.")
        return

    for coll_info in collections:
        print(f"\n--- Collection: {coll_info.name} ---")
        coll = client.get_collection(coll_info.name)

        # ✅ ids are always included, don’t add to include[]
        results = coll.get(include=["documents", "metadatas", "embeddings"])
        ids = results.get("ids", [])
        docs = results.get("documents", [])
        metas = results.get("metadatas", [])

        for i, (doc_id, doc, meta) in enumerate(zip(ids, docs, metas)):
            print(f"\n[{i}] ID: {doc_id}")
            if meta:
                print(f"   Meta: {meta}")
            print(f"   Doc: {doc[:200]}{'...' if len(doc) > 200 else ''}")

        print(f"\nTotal records: {len(ids)}")

if __name__ == "__main__":
    inspect_all()
