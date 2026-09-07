import os
import json
import glob
import faiss
import config

import numpy as np

from sentence_transformers import SentenceTransformer

now_next = []

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 75) -> list[str]:
    global now_next
    now_next.append('chunking text')
    print(f"{now_next[-1]} of {now_next[::-1]}")

    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - overlap
    now_next.pop()
    return chunks

def load_docs(docs_dir: str) -> list[dict]:
    global now_next
    now_next.append('loading docs')
    print(f"{now_next[-1]} of {now_next[::-1]}")

    docs = []
    for path in sorted(glob.glob(os.path.join(docs_dir, "**/*.txt"), recursive=True)):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            docs.append({"source": os.path.relpath(path, docs_dir), "text": f.read()})
    now_next.pop()
    return docs

def building_index(docs_dir: str = None, index_dir: str = None):
    global now_next
    now_next.append('building indices')
    print(f"{now_next[-1]} of {now_next[::-1]}")

    docs_dir = docs_dir or config.DOCS_DIR
    index_dir = index_dir or config.INDEX_DIR
    os.makedirs(index_dir, exist_ok=True)

    docs = load_docs(docs_dir)
    if not docs:
        raise SystemError(
            f"add some .txt file as the source document first\n where? here `{docs_dir}`"
        )

    chunk_recs = []
    for doc in docs:
        for i, chunk in enumerate(chunk_text(doc["text"])):
            chunk_recs.append(
                {"source": doc["source"], "chunk_id": i, "text": chunk}
            )

    now_next.append('encoding to embeddings')
    print(f"{now_next[-1]} of {now_next[::-1]}")
    
    model = SentenceTransformer(config.EMBED_MODEL)
    embs = model.encode(
        [c["text"] for c in chunk_recs],
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")
    now_next.pop()

    now_next.append('saving embeddings')
    print(f"{now_next[-1]} of {now_next[::-1]}")
    
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    faiss.write_index(index, os.path.join(index_dir, "faiss.index"))
    with open(os.path.join(index_dir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunk_recs, f, ensure_ascii=False, indent=2)
    now_next.pop()

    now_next.pop()
    print('done')

if __name__ == "__main__":
    building_index()