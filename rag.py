"""Simple PDF RAG utilities for MarineWise AI."""

from __future__ import annotations

import re
from typing import Any

import faiss
import fitz
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 900
DEFAULT_CHUNK_OVERLAP = 120


def get_embedder() -> SentenceTransformer:
    """Load the small local embedding model."""
    return SentenceTransformer(EMBEDDING_MODEL)


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError("Chunk overlap must be smaller than chunk size.")

    text = _clean_text(text)
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def extract_pdf(pdf_bytes: bytes, source_name: str) -> list[dict[str, Any]]:
    """Extract page text while preserving the exact PDF page number."""
    records: list[dict[str, Any]] = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_no, page in enumerate(doc, start=1):
            text = _clean_text(page.get_text("text"))
            if not text:
                continue
            for chunk_no, chunk in enumerate(_chunk_text(text), start=1):
                records.append(
                    {
                        "text": chunk,
                        "source": source_name,
                        "page": page_no,
                        "chunk": chunk_no,
                    }
                )
    return records


def build_index(
    pdf_items: list[tuple[str, bytes]],
    embedder: SentenceTransformer,
) -> dict[str, Any]:
    """Build a normalized inner-product FAISS index."""
    records: list[dict[str, Any]] = []
    for name, data in pdf_items:
        records.extend(extract_pdf(data, name))

    if not records:
        raise ValueError(
            "No readable text was found. If the PDF is scanned, OCR it first."
        )

    embeddings = embedder.encode(
        [record["text"] for record in records],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return {
        "index": index,
        "records": records,
        "chunk_size": DEFAULT_CHUNK_SIZE,
        "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
        "embedding_model": EMBEDDING_MODEL,
        "manual_count": len(pdf_items),
    }


def search_index(
    rag_state: dict[str, Any],
    query: str,
    embedder: SentenceTransformer,
    k: int = 6,
) -> list[dict[str, Any]]:
    if not rag_state or rag_state.get("index") is None:
        return []

    records = rag_state["records"]
    if not records:
        return []

    vector = embedder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    k = min(k, len(records))
    scores, indices = rag_state["index"].search(vector, k)

    results: list[dict[str, Any]] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        item = dict(records[int(idx)])
        item["score"] = float(score)
        results.append(item)
    return results


def retrieve_context(
    results: list[dict[str, Any]],
    min_score: float = 0.28,
    max_chunks: int = 5,
    max_chars: int = 7000,
) -> tuple[str, list[dict[str, Any]]]:
    """Return a small, high-signal evidence set for the LLM.

    This prevents a large manual from turning a good FAISS search into an
    oversized Groq request. Results are already similarity-ranked, so the
    highest scoring chunks are retained first.
    """
    relevant = [item for item in results if item["score"] >= min_score][:max_chunks]
    parts = []
    used_chars = 0
    kept = []
    for number, item in enumerate(relevant, start=1):
        part = (
            f"SOURCE {number}: {item['source']} | PAGE {item['page']} | "
            f"similarity {item['score']:.3f}\n{item['text']}"
        )
        if used_chars + len(part) > max_chars:
            remaining = max_chars - used_chars
            if remaining > 300:
                part = part[:remaining] + "\n[Excerpt truncated.]"
                parts.append(part)
                kept.append(item)
            break
        parts.append(part)
        kept.append(item)
        used_chars += len(part)
    return "\n\n".join(parts), kept


def download_google_drive_pdf(url: str) -> tuple[str, bytes]:
    """Download a publicly accessible Google Drive file."""
    match = re.search(r"/file/d/([\w-]+)", url) or re.search(r"[?&]id=([\w-]+)", url)
    if not match:
        raise ValueError("Could not find a Google Drive file ID in that link.")

    file_id = match.group(1)
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(download_url, timeout=60)
    response.raise_for_status()

    content_type = response.headers.get("content-type", "").lower()
    if "text/html" in content_type:
        raise ValueError(
            "Google Drive returned an HTML sharing/confirmation page. "
            "Make the PDF accessible to anyone with the link."
        )

    content = response.content
    if not content.startswith(b"%PDF"):
        raise ValueError(
            "The Google Drive response was not a PDF. Check the sharing permission and link."
        )

    return f"Google Drive PDF {file_id}.pdf", content
