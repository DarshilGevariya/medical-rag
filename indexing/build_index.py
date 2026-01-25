import json
import numpy as np
from pathlib import Path
import faiss
from tqdm import tqdm

from indexing.chunking import chunk_text
from indexing.embeddings import embed


DATA_FILE = Path("data/processed/qa_records.jsonl")
OUT_INDEX = Path("index.faiss")
OUT_META = Path("index_metadata.json")


def load_records():
    records = []
    with DATA_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def build_index(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    print("Loading QA records...")
    records = load_records()
    print(f"Loaded {len(records)} records")

    texts = []
    metadata = []

    print("Chunking answers...")
    for r in tqdm(records):
        chunks = chunk_text(r["answer"], chunk_size=256, overlap=50)
        for c in chunks:
            texts.append(c)
            metadata.append({
                "question": r["question"],
                "source": r["source"],
                "text": c
            })

    print(f"Total chunks: {len(texts)}")

    print("Generating embeddings...")
    embeddings, dim, elapsed = embed(texts, model_name)
    embeddings = np.array(embeddings).astype("float32")

    print("Building FAISS index...")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    print("Saving index to disk...")
    faiss.write_index(index, str(OUT_INDEX))

    with OUT_META.open("w", encoding="utf-8") as f:
        json.dump(metadata, f)

    print("✅ index.faiss created successfully")
    print(f"Index size: {index.ntotal} vectors")


if __name__ == "__main__":
    build_index()
