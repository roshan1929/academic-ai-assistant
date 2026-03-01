"""Index building utilities for document collections.

This module provides a per-subject FAISS index builder. It keeps
imports that require external packages inside functions so importing
this module does not fail when those packages are not installed.
"""

from pathlib import Path
from typing import List


def scan_subject_dirs(source_dir: str) -> List[Path]:
    """Return a list of subject directories inside `source_dir`.

    Each immediate subdirectory of `source_dir` is treated as a subject.
    """
    base = Path(source_dir)
    if not base.exists():
        print(f"Source directory does not exist: {base}")
        return []
    return [p for p in base.iterdir() if p.is_dir()]


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract and return text from a PDF using PyMuPDF (fitz).

    Returns an empty string on failure.
    """
    try:
        import fitz  # PyMuPDF
    except Exception as exc:
        print("PyMuPDF (fitz) is required to extract PDF text:", exc)
        return ""

    try:
        doc = fitz.open(str(pdf_path))
        text_chunks = []
        for page in doc:
            text_chunks.append(page.get_text())
        doc.close()
        return "\n".join(text_chunks)
    except Exception as exc:
        print(f"Failed to extract {pdf_path}: {exc}")
        return ""


def extract_text_from_docx(docx_path: Path) -> str:
    """Extract text from a .docx Word document using python-docx.

    Returns an empty string on failure.
    """
    try:
        from docx import Document
    except Exception as exc:
        print("python-docx is required to extract .docx text:", exc)
        return ""

    try:
        doc = Document(str(docx_path))
        paragraphs = [p.text for p in doc.paragraphs if p.text]
        return "\n".join(paragraphs)
    except Exception as exc:
        print(f"Failed to extract {docx_path}: {exc}")
        return ""


def clean_text(text: str) -> str:
    """Aggressively clean extracted text.

    Steps:
    - Unicode normalize
    - Replace control characters (except common whitespace) with spaces
    - Remove isolated short lines likely to be headers/footers
    - Collapse repeated punctuation
    - Collapse whitespace
    """
    import re
    import unicodedata

    if not text:
        return ""

    # Normalize unicode
    text = unicodedata.normalize("NFKC", text)

    # Replace obvious replacement chars
    text = text.replace("\ufffd", " ")

    # Replace control characters except commonly useful whitespace
    cleaned_chars = []
    for ch in text:
        # keep tab, newline, carriage return
        if ch in "\t\n\r":
            cleaned_chars.append(ch)
            continue
        # remove other control chars
        if unicodedata.category(ch)[0] == "C":
            cleaned_chars.append(" ")
        else:
            cleaned_chars.append(ch)
    text = "".join(cleaned_chars)

    # Split lines and drop noisy short lines (likely headers/footers)
    lines = [ln.strip() for ln in text.splitlines()]
    kept = []
    for ln in lines:
        if not ln:
            continue
        # drop lines that are short and mostly non-alphanumeric
        alnum = sum(1 for c in ln if c.isalnum())
        if len(ln) < 20 and alnum < 5:
            continue
        kept.append(ln)
    text = "\n".join(kept)

    # Reduce long runs of punctuation to a single char (any non-word, non-space)
    text = re.sub(r'([^\w\s])\1{2,}', r'\1', text)

    # Collapse multiple whitespace into single space
    text = re.sub(r"\s+", " ", text).strip()

    return text


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Chunk text into pieces of approximately `chunk_size` tokens (words).

    This implementation uses whitespace tokenization (words) as a
    lightweight proxy for model tokens. `overlap` is the number of
    tokens to overlap between consecutive chunks.
    """
    if not text:
        return []
    words = text.split()
    if chunk_size <= 0:
        return [text]
    chunks = []
    start = 0
    L = len(words)
    while start < L:
        end = min(start + chunk_size, L)
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == L:
            break
        start = max(end - overlap, end)
    return chunks


def build_subject_index(subject_dir: Path, out_base: Path, model_name: str = "all-MiniLM-L6-v2") -> None:
    """Build a FAISS index for a single subject directory.

    - Loads PDFs, extracts text, chunks content
    - Embeds chunks using SentenceTransformer
    - Builds and saves a FAISS index and chunk metadata under
      `out_base/{subject}_index/`
    """
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        print("sentence-transformers is required to compute embeddings:", exc)
        return

    try:
        import faiss
    except Exception as exc:
        print("faiss is required to build vector indices:", exc)
        return

    import numpy as np
    import pickle

    subject_name = subject_dir.name
    print(f"Building index for subject: {subject_name}")

    # gather texts from PDFs and DOCX files
    files = sorted([p for p in subject_dir.iterdir() if p.suffix.lower() in (".pdf", ".docx")])
    all_chunks = []
    chunk_sources = []
    for f in files:
        print(f"  - Extracting {f.name}")
        if f.suffix.lower() == ".pdf":
            text = extract_text_from_pdf(f)
        else:
            text = extract_text_from_docx(f)

        # Aggressive cleaning before chunking to remove OCR/noise artifacts
        cleaned = clean_text(text)
        if cleaned != text:
            print(f"    cleaned text: {len(text)} -> {len(cleaned)} chars")

        chunks = chunk_text(cleaned, chunk_size=500, overlap=100)
        for ch in chunks:
            all_chunks.append(ch)
            chunk_sources.append(str(f.name))

    if not all_chunks:
        print(f"  No content found for subject {subject_name}; skipping.")
        return

    print(f"  Encoding {len(all_chunks)} chunks with SentenceTransformer ({model_name})")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(all_chunks, show_progress_bar=True)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    dim = embeddings.shape[1]

    print(f"  Creating FAISS index (dim={dim})")
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # prepare output path
    out_dir = out_base / f"{subject_name}_index"
    out_dir.mkdir(parents=True, exist_ok=True)

    idx_file = out_dir / "index.faiss"
    meta_file = out_dir / "metadata.pkl"

    print(f"  Saving index to {idx_file}")
    faiss.write_index(index, str(idx_file))

    print(f"  Saving metadata to {meta_file}")
    metadata = {
        "chunks": all_chunks,
        "sources": chunk_sources,
        "subject": subject_name,
        "model_name": model_name,
    }
    with open(meta_file, "wb") as fh:
        pickle.dump(metadata, fh)

    print(f"  Completed index for subject: {subject_name}")


def build_index(source_dir: str, index_path: str) -> None:
    """Scan `source_dir` for subject folders and build one FAISS index per subject.

    Index files and metadata are written to `index_path/{subject}_index/`.
    """
    src = Path(source_dir)
    out_base = Path(index_path)
    subjects = scan_subject_dirs(source_dir)
    if not subjects:
        print(f"No subjects found in {src}")
        return

    print(f"Found {len(subjects)} subject(s): {[p.name for p in subjects]}")
    for subj in subjects:
        build_subject_index(subj, out_base)


def _main():
    """Command-line entrypoint for quick local testing.

    Defaults to scanning `data/raw_pdfs` and writing to `data/vector_store`.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Build per-subject FAISS indices from PDFs")
    parser.add_argument("--source", default="data/raw_pdfs", help="Source folder containing subject subfolders")
    parser.add_argument("--out", default="data/vector_store", help="Output base folder for per-subject indices")
    args = parser.parse_args()

    build_index(args.source, args.out)


if __name__ == "__main__":
    _main()

