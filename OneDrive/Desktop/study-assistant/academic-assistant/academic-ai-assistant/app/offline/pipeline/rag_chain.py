"""Simple RAG chain implementation.

This module loads a per-subject FAISS index, embeds a question using
SentenceTransformer (cached singleton), retrieves the top-k chunks,
constructs a strict context-based prompt, and asks the LLM for an
answer.

External heavy dependencies are imported inside functions so the
module can be imported without requiring them.
"""

from pathlib import Path
from typing import Optional
import threading

_EMB_MODEL = None
_EMB_LOCK = threading.Lock()


def _get_embedder(model_name: str = "all-MiniLM-L6-v2"):
    """Return a singleton SentenceTransformer embedder instance."""
    global _EMB_MODEL
    if _EMB_MODEL is not None:
        return _EMB_MODEL

    with _EMB_LOCK:
        if _EMB_MODEL is not None:
            return _EMB_MODEL
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            print("sentence-transformers is required for embeddings:", exc)
            return None

        model = SentenceTransformer(model_name)
        _EMB_MODEL = model
        print(f"Loaded embedding model: {model_name}")
        return _EMB_MODEL


def _load_faiss_index(subject: str, base_path: str = "data/vector_store"):
    """Load FAISS index and metadata for a subject.

    Returns (index, metadata) or (None, None) on failure.
    """
    try:
        import faiss
    except Exception as exc:
        print("faiss is required to load indices:", exc)
        return None, None

    import pickle

    idx_dir = Path(base_path) / f"{subject}_index"
    idx_file = idx_dir / "index.faiss"
    meta_file = idx_dir / "metadata.pkl"

    if not idx_file.exists() or not meta_file.exists():
        print(f"Index or metadata missing for subject '{subject}' in {idx_dir}")
        return None, None

    try:
        index = faiss.read_index(str(idx_file))
    except Exception as exc:
        print(f"Failed to read FAISS index: {exc}")
        return None, None

    try:
        with open(meta_file, "rb") as fh:
            metadata = pickle.load(fh)
    except Exception as exc:
        print(f"Failed to read metadata: {exc}")
        return None, None

    return index, metadata


def answer_question(subject: str, question: str, top_k: int = 2) -> Optional[str]:
    """Run a simple RAG flow: retrieve context and ask the LLM.

    Returns the LLM response string, or None on failure.
    """
    print(f"RAG: subject={subject}, question={question}")

    index, metadata = _load_faiss_index(subject)
    if index is None or metadata is None:
        print("Failed to load index/metadata; aborting RAG.")
        return None

    embedder = _get_embedder()
    if embedder is None:
        print("Embedding model not available; aborting RAG.")
        return None

    print("Encoding question...")
    try:
        q_emb = embedder.encode([question])
    except Exception as exc:
        print(f"Failed to encode question: {exc}")
        return None

    import numpy as np
    q_vec = np.asarray(q_emb, dtype=np.float32)

    # Ensure 2D
    if q_vec.ndim == 1:
        q_vec = q_vec.reshape(1, -1)

    try:
        D, I = index.search(q_vec, top_k)
    except Exception as exc:
        print(f"FAISS search failed: {exc}")
        return None

    indices = I[0].tolist()
    print(f"Retrieved indices: {indices}")

    chunks = metadata.get("chunks", [])
    sources = metadata.get("sources", [])

    retrieved = []
    for idx in indices:
        if idx is None or idx < 0 or idx >= len(chunks):
            continue
        retrieved.append((chunks[idx], sources[idx] if idx < len(sources) else "unknown"))

    if not retrieved:
        print("No relevant chunks retrieved.")
        return None

    # Build strict prompt
    context_parts = []
    for i, (txt, src) in enumerate(retrieved, start=1):
        context_parts.append(f"--- Context chunk {i} (source: {src}) ---\n{txt}\n")

    context = "\n".join(context_parts)

    prompt = (
        "Answer strictly from the provided context. Do not use external knowledge. "
        "If the answer is not contained in the context, respond with 'I don't know.'\n\n"
        f"Context:\n{context}\nQuestion: {question}\nAnswer:"
    )

    # Send to LLM
    try:
        from app.offline.llm.load_llm import get_llm
    except Exception as exc:
        print(f"Failed to import LLM loader: {exc}")
        return None

    llm = get_llm()
    if llm is None:
        print("LLM not available; cannot generate answer.")
        return None

    print("Sending prompt to LLM...")
    try:
        # Our wrapper exposes generate(prompt)
        resp = llm.generate(prompt)
        # If the underlying client returns complex object, try to extract text
        if isinstance(resp, dict) and "text" in resp:
            return resp["text"]
        return str(resp)
    except Exception as exc:
        print(f"LLM generation failed: {exc}")
        return None


def create_rag_chain(store=None, llm=None):
    """Compatibility placeholder: return a callable that runs `answer_question`.

    This keeps the original API shape while implementing the RAG logic in
    `answer_question()`.
    """
    return answer_question

