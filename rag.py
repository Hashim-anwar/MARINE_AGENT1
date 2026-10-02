"""
MarineWise AI - PDF RAG utilities.

This module handles:
- PDF text extraction
- Text chunking
- Sentence-transformer embeddings
- FAISS vector search
- Google Drive PDF downloading
- Low-resolution manual page images for presentations

The FAISS index is built once by the Streamlit app and kept in
st.session_state.

Manual text remains the primary source for troubleshooting and training.
"""

from __future__ import annotations

import io
import re
from typing import Any

import faiss
import fitz
import numpy as np
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

DEFAULT_CHUNK_SIZE = 900
DEFAULT_CHUNK_OVERLAP = 120

# Number of pixels / rendering scale is deliberately modest.
# We want useful presentation visuals without consuming huge
# amounts of Streamlit memory.
PAGE_IMAGE_SCALE = 1.0

# JPEG quality for stored manual page images.
PAGE_IMAGE_QUALITY = 72

# Safety limit for a single page image.
MAX_PAGE_IMAGE_BYTES = 900_000


# ============================================================
# EMBEDDING MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def get_embedder() -> SentenceTransformer:
    """
    Load the sentence-transformer model once.

    Streamlit caches the model so it is not downloaded and
    loaded again for every user interaction.
    """

    return SentenceTransformer(EMBEDDING_MODEL)


# ============================================================
# TEXT CLEANING
# ============================================================

def _clean_text(text: str) -> str:
    """Normalize whitespace in extracted PDF text."""

    if not text:
        return ""

    text = text.replace("\x00", " ")

    # Replace excessive whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# TEXT CHUNKING
# ============================================================

def _chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """
    Split text into overlapping chunks.

    Character-based chunking is intentionally simple for this
    beginner-friendly MVP.
    """

    text = _clean_text(text)

    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0:
        raise ValueError("overlap cannot be negative.")

    if overlap >= chunk_size:
        raise ValueError(
            "chunk overlap must be smaller than chunk size."
        )

    chunks: list[str] = []

    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(
            start + chunk_size,
            text_length,
        )

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = end - overlap

    return chunks


# ============================================================
# MANUAL PAGE IMAGE
# ============================================================

def _render_page_image(page: fitz.Page) -> bytes | None:
    """
    Render one PDF page as a compressed JPEG.

    These images are used by the Training section to show
    actual manual pages in PowerPoint.

    The image is deliberately lower resolution than the original
    PDF so that Streamlit memory usage remains reasonable.
    """

    try:
        matrix = fitz.Matrix(
            PAGE_IMAGE_SCALE,
            PAGE_IMAGE_SCALE,
        )

        pixmap = page.get_pixmap(
            matrix=matrix,
            alpha=False,
        )

        image = fitz.Pixmap(
            fitz.csRGB,
            pixmap,
        )

        jpeg_bytes = image.tobytes(
            "jpeg",
            jpg_quality=PAGE_IMAGE_QUALITY,
        )

        # If the page is unusually large, reduce it further.
        if len(jpeg_bytes) > MAX_PAGE_IMAGE_BYTES:
            return _compress_page_image(
                jpeg_bytes,
                max_bytes=MAX_PAGE_IMAGE_BYTES,
            )

        return jpeg_bytes

    except Exception:
        return None


def _compress_page_image(
    image_bytes: bytes,
    max_bytes: int = MAX_PAGE_IMAGE_BYTES,
) -> bytes | None:
    """
    Further compress a page image if it is too large.

    Pillow is used here because it is already required by the app.
    """

    try:
        from PIL import Image

        image = Image.open(
            io.BytesIO(image_bytes)
        )

        image = image.convert("RGB")

        # Start with a reasonable width.
        max_width = 1400

        if image.width > max_width:
            ratio = max_width / image.width

            new_size = (
                max_width,
                max(
                    1,
                    int(image.height * ratio),
                ),
            )

            image = image.resize(
                new_size,
                Image.LANCZOS,
            )

        quality = 65

        while quality >= 35:
            output = io.BytesIO()

            image.save(
                output,
                format="JPEG",
                quality=quality,
                optimize=True,
            )

            result = output.getvalue()

            if len(result) <= max_bytes:
                return result

            quality -= 5

        return result

    except Exception:
        return image_bytes


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf(
    pdf_bytes: bytes,
    source_name: str,
) -> list[dict[str, Any]]:
    """
    Extract text from a PDF.

    Returns one record per text chunk.

    Each record contains:
        text
        source
        page
        chunk

    Page numbers are 1-based for technicians.
    """

    if not pdf_bytes:
        return []

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
        for page_index in range(document.page_count):
            page = document.load_page(page_index)

            page_number = page_index + 1

            text = page.get_text("text")

            text = _clean_text(text)

            if not text:
                continue

            chunks = _chunk_text(
                text,
                DEFAULT_CHUNK_SIZE,
                DEFAULT_CHUNK_OVERLAP,
            )

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


# ============================================================
# PDF EXTRACTION + PAGE IMAGES
# ============================================================

def extract_pdf_with_images(
    pdf_bytes: bytes,
    source_name: str,
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], bytes]]:
    """
    Extract PDF text and render useful page images.

    Returns:
        (
            records,
            page_images
        )

    page_images is keyed by:
        (source_name, page_number)

    The image is stored only once per page even when that page
    contains several text chunks.
    """

    if not pdf_bytes:
        return [], {}

    records: list[dict[str, Any]] = []

    page_images: dict[tuple[str, int], bytes] = {}

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
        for page_index in range(document.page_count):
            page = document.load_page(page_index)

            page_number = page_index + 1

            # ------------------------------------------------
            # Text
            # ------------------------------------------------

            text = page.get_text("text")

            text = _clean_text(text)

            if text:
                chunks = _chunk_text(
                    text,
                    DEFAULT_CHUNK_SIZE,
                    DEFAULT_CHUNK_OVERLAP,
                )

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

            # ------------------------------------------------
            # Page image
            # ------------------------------------------------

            # We only need page visuals for pages that contain
            # meaningful text. This avoids storing blank pages.
            if text:
                image_bytes = _render_page_image(page)

                if image_bytes:
                    page_images[
                        (source_name, page_number)
                    ] = image_bytes

    finally:
        document.close()

    return records, page_images


# ============================================================
# BUILD FAISS INDEX
# ============================================================

def build_index(
    pdf_items: list[tuple[str, bytes]],
    embedder: SentenceTransformer,
) -> dict[str, Any]:
    """
    Build a FAISS vector index from uploaded PDFs.

    pdf_items format:
        [
            ("manual1.pdf", pdf_bytes),
            ("manual2.pdf", pdf_bytes),
        ]

    Returns a dictionary containing:
        index
        records
        page_images
        chunk_size
        chunk_overlap
        embedding_model
        manual_count
    """

    if not pdf_items:
        raise ValueError(
            "No PDF manuals were supplied."
        )

    if embedder is None:
        raise ValueError(
            "Embedding model is not available."
        )

    all_records: list[dict[str, Any]] = []

    all_page_images: dict[
        tuple[str, int],
        bytes,
    ] = {}

    successful_manuals = 0

    errors: list[str] = []

    # --------------------------------------------------------
    # Extract every manual.
    # --------------------------------------------------------

    for source_name, pdf_bytes in pdf_items:

        try:
            records, page_images = extract_pdf_with_images(
                pdf_bytes,
                source_name,
            )

            if records:
                successful_manuals += 1

            all_records.extend(records)

            all_page_images.update(page_images)

        except Exception as exc:
            errors.append(
                f"{source_name}: {exc}"
            )

    if not all_records:
        message = (
            "No readable text was found in the supplied PDFs."
        )

        if errors:
            message += " " + " | ".join(errors)

        raise ValueError(message)

    # --------------------------------------------------------
    # Create embeddings.
    # --------------------------------------------------------

    texts = [
        record["text"]
        for record in all_records
    ]

    embeddings = embedder.encode(
        texts,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32",
    )

    # --------------------------------------------------------
    # FAISS cosine-similarity index.
    #
    # Because embeddings are normalized,
    # inner product = cosine similarity.
    # --------------------------------------------------------

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(embeddings)

    return {
        "index": index,
        "records": all_records,
        "page_images": all_page_images,
        "chunk_size": DEFAULT_CHUNK_SIZE,
        "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
        "embedding_model": EMBEDDING_MODEL,
        "manual_count": successful_manuals,
        "page_image_count": len(all_page_images),
        "errors": errors,
    }


# ============================================================
# FAISS SEARCH
# ============================================================

def search_index(
    rag_state: dict[str, Any],
    query: str,
    embedder: SentenceTransformer,
    k: int = 6,
) -> list[dict[str, Any]]:
    """
    Search the FAISS index.

    Returns records with:
        score
        text
        source
        page
        chunk
    """

    if not rag_state:
        return []

    index = rag_state.get("index")

    records = rag_state.get(
        "records",
        [],
    )

    if index is None or not records:
        return []

    query = _clean_text(query)

    if not query:
        return []

    if embedder is None:
        return []

    # Do not request more records than exist.
    k = max(
        1,
        min(
            int(k),
            len(records),
        ),
    )

    query_embedding = embedder.encode(
        [query],
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32",
    )

    scores, indices = index.search(
        query_embedding,
        k,
    )

    results: list[dict[str, Any]] = []

    for score, index_position in zip(
        scores[0],
        indices[0],
    ):
        if index_position < 0:
            continue

        if index_position >= len(records):
            continue

        record = dict(
            records[index_position]
        )

        record["score"] = float(score)

        results.append(record)

    return results


# ============================================================
# RETRIEVE CONTEXT
# ============================================================

def retrieve_context(
    results: list[dict[str, Any]],
    min_score: float = 0.32,
    max_chunks: int = 3,
    max_chars: int = 4500,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Convert search results into compact LLM context.

    Returns:
        context_text
        relevant_results
    """

    if not results:
        return "", []

    selected: list[dict[str, Any]] = []

    current_chars = 0

    seen_keys: set[tuple[str, int, int]] = set()

    for result in results:

        score = float(
            result.get("score", 0.0)
        )

        if score < min_score:
            continue

        source = str(
            result.get("source", "")
        )

        page = int(
            result.get("page", 0)
        )

        chunk = int(
            result.get("chunk", 0)
        )

        unique_key = (
            source,
            page,
            chunk,
        )

        if unique_key in seen_keys:
            continue

        seen_keys.add(unique_key)

        text = _clean_text(
            str(
                result.get("text", "")
            )
        )

        if not text:
            continue

        remaining = max_chars - current_chars

        if remaining <= 0:
            break

        if len(text) > remaining:
            text = text[:remaining].rstrip()

        selected_result = dict(result)

        selected_result["text"] = text

        selected.append(
            selected_result
        )

        current_chars += len(text)

        if (
            len(selected) >= max_chunks
            or current_chars >= max_chars
        ):
            break

    context_parts: list[str] = []

    for result in selected:
        context_parts.append(
            "SOURCE: "
            f"{result.get('source', 'Unknown')}\n"
            "PAGE: "
            f"{result.get('page', 'Unknown')}\n"
            "CHUNK: "
            f"{result.get('chunk', 'Unknown')}\n"
            "RELEVANCE: "
            f"{float(result.get('score', 0.0)):.3f}\n"
            "TEXT:\n"
            f"{result.get('text', '')}"
        )

    context = "\n\n---\n\n".join(
        context_parts
    )

    return context, selected


# ============================================================
# PAGE IMAGE RETRIEVAL
# ============================================================

def get_page_image(
    rag_state: dict[str, Any],
    source: str,
    page: int,
) -> bytes | None:
    """
    Retrieve a stored manual page image.

    Useful for PowerPoint generation.
    """

    if not rag_state:
        return None

    page_images = rag_state.get(
        "page_images",
        {},
    )

    if not page_images:
        return None

    return page_images.get(
        (source, int(page))
    )


def get_relevant_page_images(
    rag_state: dict[str, Any],
    results: list[dict[str, Any]],
    max_images: int = 3,
) -> list[dict[str, Any]]:
    """
    Return manual page images corresponding to retrieved
    FAISS results.

    Each result contains:
        source
        page
        score
        image
    """

    if not rag_state:
        return []

    page_images = rag_state.get(
        "page_images",
        {},
    )

    if not page_images:
        return []

    output: list[dict[str, Any]] = []

    seen_pages: set[tuple[str, int]] = set()

    for result in results:

        source = str(
            result.get("source", "")
        )

        page = int(
            result.get("page", 0)
        )

        key = (
            source,
            page,
        )

        if key in seen_pages:
            continue

        seen_pages.add(key)

        image_bytes = page_images.get(key)

        if not image_bytes:
            continue

        output.append(
            {
                "source": source,
                "page": page,
                "score": float(
                    result.get("score", 0.0)
                ),
                "image": image_bytes,
            }
        )

        if len(output) >= max_images:
            break

    return output


# ============================================================
# GOOGLE DRIVE PDF DOWNLOAD
# ============================================================

def _google_drive_file_id(url: str) -> str | None:
    """
    Extract a Google Drive file ID from common public
    Google Drive URL formats.
    """

    if not url:
        return None

    patterns = [
        r"/file/d/([a-zA-Z0-9_-]+)",
        r"[?&]id=([a-zA-Z0-9_-]+)",
        r"/uc\?id=([a-zA-Z0-9_-]+)",
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            url,
        )

        if match:
            return match.group(1)

    return None


def download_google_drive_pdf(
    url: str,
) -> tuple[str, bytes]:
    """
    Download a publicly accessible Google Drive PDF.

    Returns:
        source_name, pdf_bytes

    The user must provide a public/shareable PDF link.
    """

    url = (url or "").strip()

    if not url:
        raise ValueError(
            "Google Drive URL is empty."
        )

    file_id = _google_drive_file_id(url)

    if not file_id:
        raise ValueError(
            "Could not identify a Google Drive file ID. "
            "Use a standard Google Drive sharing link."
        )

    download_url = (
        "https://drive.google.com/uc"
        f"?export=download&id={file_id}"
    )

    response = requests.get(
        download_url,
        timeout=45,
        allow_redirects=True,
    )

    response.raise_for_status()

    content = response.content

    if not content:
        raise ValueError(
            "Google Drive returned an empty file."
        )

    # --------------------------------------------------------
    # Verify that the response is actually a PDF.
    # --------------------------------------------------------

    is_pdf = (
        content[:4] == b"%PDF"
        or "application/pdf"
        in response.headers.get(
            "Content-Type",
            "",
        ).lower()
    )

    if not is_pdf:
        raise ValueError(
            "The Google Drive link did not return a PDF. "
            "Make sure the file is publicly accessible."
        )

    return (
        f"google_drive_{file_id}.pdf",
        content,
    )
