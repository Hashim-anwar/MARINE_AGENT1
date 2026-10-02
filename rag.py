"""MarineWise AI RAG utilities.

Handles:
- PDF text extraction
- Chunking
- SentenceTransformer embeddings
- FAISS indexing
- Manual retrieval
- Google Drive PDF download

The embedding model is cached so Streamlit does not reload it for every
question or index search.
"""

from __future__ import annotations

import re
from typing import Any

import faiss
import fitz
import numpy as np
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL = "all-MiniLM-L6-v2"

DEFAULT_CHUNK_SIZE = 900
DEFAULT_CHUNK_OVERLAP = 120


@st.cache_resource(show_spinner=False)
def cached_embedder() -> SentenceTransformer:
    """Load the embedding model once and reuse it."""
    return SentenceTransformer(EMBEDDING_MODEL)


def get_embedder() -> SentenceTransformer:
    """Return the cached embedding model."""
    return cached_embedder()


def _clean_text(text: str) -> str:
    """Clean PDF text."""
    return re.sub(r"\s+", " ", text or "").strip()


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping chunks."""

    if overlap >= chunk_size:
        raise ValueError(
            "Chunk overlap must be smaller than chunk size."
        )

    text = _clean_text(text)

    if not text:
        return []

    chunks: list[str] = []

    start = 0

    while start < len(text):

        end = min(
            len(text),
            start + chunk_size,
        )

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


def extract_pdf(
    pdf_bytes: bytes,
    source_name: str,
) -> list[dict[str, Any]]:
    """Extract PDF text page-by-page."""

    records: list[dict[str, Any]] = []

    try:
        document = fitz.open(
            stream=pdf_bytes,
            filetype="pdf",
        )
    except Exception as exc:
        raise ValueError(
            f"Could not open PDF '{source_name}': {exc}"
        ) from exc

    try:
        for page_number, page in enumerate(
            document,
            start=1,
        ):

            raw_text = page.get_text("text")

            text = _clean_text(raw_text)

            if not text:
                continue

            chunks = chunk_text(text)

            for chunk_number, chunk in enumerate(
                chunks,
                start=1,
            ):

                records.append(
                    {
                        "text": chunk,
                        "source": source_name,
                        "page": page_number,
                        "chunk": chunk_number,
                    }
                )

    finally:
        document.close()

    return records


def build_index(
    pdf_items: list[tuple[str, bytes]],
    embedder: SentenceTransformer | None = None,
) -> dict[str, Any]:
    """Build a FAISS index from PDF manuals."""

    if not pdf_items:
        raise ValueError(
            "No PDF manuals were supplied."
        )

    if embedder is None:
        embedder = cached_embedder()

    records: list[dict[str, Any]] = []

    for source_name, pdf_bytes in pdf_items:

        if not pdf_bytes:
            continue

        pdf_records = extract_pdf(
            pdf_bytes,
            source_name,
        )

        records.extend(pdf_records)

    if not records:
        raise ValueError(
            "No readable text was found in the uploaded manuals. "
            "If your PDF contains scanned pages, OCR may be required."
        )

    texts = [
        record["text"]
        for record in records
    ]

    embeddings = embedder.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32",
    )

    if embeddings.ndim != 2:
        raise ValueError(
            "The embedding model returned an invalid embedding array."
        )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    return {
        "index": index,
        "records": records,
        "chunk_size": DEFAULT_CHUNK_SIZE,
        "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
        "embedding_model": EMBEDDING_MODEL,
        "manual_count": len(pdf_items),
        "total_chunks": len(records),
    }


def search_index(
    rag_state: dict[str, Any],
    query: str,
    embedder: SentenceTransformer | None = None,
    k: int = 6,
) -> list[dict[str, Any]]:
    """Search the FAISS manual index."""

    if not rag_state:
        return []

    index = rag_state.get("index")

    records = rag_state.get(
        "records",
        [],
    )

    if index is None:
        return []

    if not records:
        return []

    query = _clean_text(query)

    if not query:
        return []

    if embedder is None:
        embedder = cached_embedder()

    query_embedding = embedder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32",
    )

    k = max(
        1,
        min(k, len(records)),
    )

    scores, indices = index.search(
        query_embedding,
        k,
    )

    results: list[dict[str, Any]] = []

    for score, index_number in zip(
        scores[0],
        indices[0],
    ):

        if index_number < 0:
            continue

        index_number = int(index_number)

        if index_number >= len(records):
            continue

        result = dict(
            records[index_number]
        )

        result["score"] = float(score)

        results.append(result)

    return results


def retrieve_context(
    results: list[dict[str, Any]],
    min_score: float = 0.28,
    max_chunks: int = 3,
    max_chars: int = 4000,
) -> tuple[str, list[dict[str, Any]]]:
    """Create a small evidence package for the AI model.

    This function is deliberately configurable so Troubleshooting can use a
    smaller context than Training.
    """

    if not results:
        return "", []

    relevant = [
        result
        for result in results
        if result.get("score", 0) >= min_score
    ]

    relevant = relevant[:max_chunks]

    if not relevant:
        return "", []

    parts: list[str] = []

    kept: list[dict[str, Any]] = []

    used_chars = 0

    for number, item in enumerate(
        relevant,
        start=1,
    ):

        source = item.get(
            "source",
            "Unknown source",
        )

        page = item.get(
            "page",
            "?",
        )

        score = item.get(
            "score",
            0,
        )

        text = item.get(
            "text",
            "",
        )

        part = (
            f"SOURCE {number}: "
            f"{source} | PAGE {page} | "
            f"similarity {score:.3f}\n"
            f"{text}"
        )

        remaining = (
            max_chars - used_chars
        )

        if remaining <= 0:
            break

        if len(part) > remaining:

            if remaining < 250:
                break

            part = (
                part[:remaining]
                + "\n[Excerpt truncated.]"
            )

        parts.append(part)

        kept.append(item)

        used_chars += len(part)

        if used_chars >= max_chars:
            break

    return (
        "\n\n".join(parts),
        kept,
    )


def download_google_drive_pdf(
    url: str,
) -> tuple[str, bytes]:
    """Download a publicly accessible Google Drive PDF."""

    url = url.strip()

    if not url:
        raise ValueError(
            "Google Drive link is empty."
        )

    match = (
        re.search(
            r"/file/d/([\w-]+)",
            url,
        )
        or re.search(
            r"[?&]id=([\w-]+)",
            url,
        )
    )

    if not match:
        raise ValueError(
            "Could not find a Google Drive file ID "
            "in that link."
        )

    file_id = match.group(1)

    download_url = (
        "https://drive.google.com/uc"
        f"?export=download&id={file_id}"
    )

    response = requests.get(
        download_url,
        timeout=60,
        allow_redirects=True,
    )

    response.raise_for_status()

    content_type = (
        response.headers
        .get("content-type", "")
        .lower()
    )

    if "text/html" in content_type:
        raise ValueError(
            "Google Drive returned an HTML page. "
            "Make the PDF accessible to anyone with "
            "the link."
        )

    pdf_bytes = response.content

    if not pdf_bytes.startswith(b"%PDF"):
        raise ValueError(
            "The Google Drive response was not a PDF. "
            "Check the sharing permission and link."
        )

    filename = (
        f"Google Drive PDF {file_id}.pdf"
    )

    return filename, pdf_bytes
