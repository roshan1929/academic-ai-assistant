"""Diagnostic tool to inspect a per-subject FAISS index and metadata.

Usage:
    python app/offline/ingestion/diagnose_index.py --subject cn

This script prints metadata info, FAISS index stats, and attempts a
retrieval using SentenceTransformer if available (falls back to random
vector if not).
"""

from pathlib import Path
import argparse
import pickle
import sys


def main(subject: str, base: str = "data/vector_store"):
    idx_dir = Path(base) / f"{subject}_index"
    meta_file = idx_dir / "metadata.pkl"
    idx_file = idx_dir / "index.faiss"

    print(f"Inspecting subject '{subject}' in {idx_dir}")
    if not idx_dir.exists():
        print("Index directory not found.")
        return 1

    if not meta_file.exists():
        print("Metadata file missing:", meta_file)
        return 1

    with open(meta_file, "rb") as fh:
        metadata = pickle.load(fh)

    chunks = metadata.get("chunks", [])
    sources = metadata.get("sources", [])

    print(f"Metadata keys: {list(metadata.keys())}")
    print(f"Chunks: {len(chunks)}; Sources: {len(sources)}")
    if chunks:
        print("Sample chunk (first 300 chars):\n", chunks[0][:300])

    try:
        import faiss
    except Exception as exc:
        print("faiss not available:", exc)
        return 1

    try:
        index = faiss.read_index(str(idx_file))
    except Exception as exc:
        print("Failed to read FAISS index:", exc)
        return 1

    ntotal = getattr(index, "ntotal", None)
    dim = getattr(index, "d", None)
    print(f"FAISS index ntotal={ntotal}, dim={dim}")

    # Try to get an embedding for a test query
    query = "what is dns"
    try:
        from sentence_transformers import SentenceTransformer
        emb_model = SentenceTransformer(metadata.get("model_name", "all-MiniLM-L6-v2"))
        qv = emb_model.encode([query])
        print("Using SentenceTransformer for query embedding")
    except Exception as exc:
        print("Could not load SentenceTransformer, using random vector for test:", exc)
        import numpy as np
        if dim is None:
            print("Unknown index dim; cannot create random vector")
            return 1
        qv = np.random.rand(1, dim).astype("float32")

    import numpy as np
    qv = np.array(qv, dtype=np.float32)
    if qv.ndim == 1:
        qv = qv.reshape(1, -1)

    try:
        D, I = index.search(qv, 2)
        print("Search distances:", D)
        print("Search indices:", I)
    except Exception as exc:
        print("FAISS search failed:", exc)
        return 1

    # Map results to chunks
    for idx in I[0]:
        if idx < 0 or idx >= len(chunks):
            print(f"Result index {idx} out of range for chunks (len={len(chunks)})")
        else:
            print(f"--- Retrieved (idx={idx}) source={sources[idx] if idx < len(sources) else 'unknown'} ---")
            print(chunks[idx][:500])

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="cn")
    parser.add_argument("--base", default="data/vector_store")
    args = parser.parse_args()
    sys.exit(main(args.subject, args.base))
