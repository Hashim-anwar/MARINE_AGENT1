"""
MarineWise AI
Professional marine-engine troubleshooting and technical training MVP.

Main features:
- Groq or Gemini as the AI provider
- Tavily for explicit web research
- PDF manual RAG with FAISS
- Manual-first troubleshooting
- Professional training PowerPoint generation
- Manual page visuals
- Online research and image retrieval
- Programmatically generated technical diagrams
- PDF / PPTX / Word training outputs
- Quiz generation
- Assessment scoring
- Remedial training presentation
"""

from __future__ import annotations

import io
import os
import re
import textwrap
from typing import Any

import requests
import streamlit as st
from PIL import Image, ImageOps

from agents import (
    run_marine_agent,
    run_web_search,
)

try:
    from agents import (
        run_training_web_research,
        generate_training_presentation_plan,
    )
except ImportError:
    run_training_web_research = None
    generate_training_presentation_plan = None

from rag import (
    DEFAULT_CHUNK_SIZE,
    build_index,
    download_google_drive_pdf,
    get_embedder,
    retrieve_context,
    search_index,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MarineWise AI",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROFESSIONAL MARINE THEME
# ============================================================

NAVY = "17324D"
DARK_NAVY = "10263D"
BLUE = "1976D2"
TEAL = "008C95"
LIGHT_TEAL = "E8F5F6"
LIGHT = "F4F7FA"
WHITE = "FFFFFF"
DARK = "17202A"
GRAY = "667085"
LIGHT_GRAY = "D9E2EC"
GREEN = "2E7D32"
ORANGE = "E67E22"
RED = "C62828"


st.markdown(
    f"""
    <style>
        .stApp {{
            background: #F7F9FC;
        }}

        [data-testid="stSidebar"] {{
            background: linear-gradient(
                180deg,
                #{DARK_NAVY} 0%,
                #{NAVY} 100%
            );
        }}

        [data-testid="stSidebar"] * {{
            color: white !important;
        }}

        .main-title {{
            font-size: 2.5rem;
            font-weight: 800;
            color: #{NAVY};
            margin-bottom: 0.1rem;
        }}

        .main-subtitle {{
            color: #{GRAY};
            font-size: 1rem;
            margin-bottom: 1.5rem;
        }}

        .section-card {{
            background: white;
            border-radius: 14px;
            padding: 1.2rem 1.4rem;
            border: 1px solid #E4E7EC;
            box-shadow: 0 2px 8px rgba(16, 38, 61, 0.05);
            margin-bottom: 1rem;
        }}

        .source-tag {{
            display: inline-block;
            padding: 4px 9px;
            margin: 3px;
            border-radius: 12px;
            background: #{LIGHT_TEAL};
            color: #{DARK_NAVY};
            font-size: 0.8rem;
            border: 1px solid #B8DFE2;
        }}

        .research-tag {{
            display: inline-block;
            padding: 4px 9px;
            margin: 3px;
            border-radius: 12px;
            background: #EAF2FF;
            color: #174A7E;
            font-size: 0.8rem;
            border: 1px solid #BDD4F2;
        }}

        .metric-card {{
            background: white;
            border-radius: 12px;
            padding: 1rem;
            border: 1px solid #E4E7EC;
            text-align: center;
        }}

        .metric-value {{
            font-size: 1.7rem;
            font-weight: 800;
            color: #{NAVY};
        }}

        .metric-label {{
            font-size: 0.8rem;
            color: #{GRAY};
        }}

        .slide-card {{
            background: white;
            border-radius: 12px;
            border: 1px solid #D9E2EC;
            padding: 1rem;
            margin: 0.5rem 0;
        }}

        .warning-box {{
            background: #FFF8E7;
            border-left: 5px solid #{ORANGE};
            padding: 1rem;
            border-radius: 8px;
        }}

        .safety-box {{
            background: #FFF1F1;
            border-left: 5px solid #{RED};
            padding: 1rem;
            border-radius: 8px;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# GENERAL HELPERS
# ============================================================


def get_secret(name: str) -> str | None:
    """Read a secret from Streamlit Secrets or environment variables."""
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None

    return value or os.getenv(name)


def selected_provider() -> str:
    return st.session_state.get("ai_provider", "Groq")


def require_provider_key(provider: str | None = None) -> str:
    provider = provider or selected_provider()

    key_name = (
        "GROQ_API_KEY"
        if provider == "Groq"
        else "GEMINI_API_KEY"
    )

    key = get_secret(key_name)

    if not key:
        raise RuntimeError(
            f"{key_name} is not configured. "
            "Add it to Streamlit Secrets or environment variables."
        )

    return key


def run_agent(
    system_prompt: str,
    user_prompt: str,
    agent_kind: str,
) -> str:
    """Run the selected AI provider."""
    provider = selected_provider()

    return run_marine_agent(
        provider,
        agent_kind,
        system_prompt,
        user_prompt,
        require_provider_key(provider),
    )


def ask_web(
    system_prompt: str,
    user_prompt: str,
) -> str:
    """Run Tavily web research followed by the selected AI provider."""
    provider = selected_provider()

    return run_web_search(
        provider,
        system_prompt,
        user_prompt,
        require_provider_key(provider),
    )


def safe_text(value: Any) -> str:
    return str(value or "").strip()


def split_text(
    text: str,
    max_chars: int = 1100,
) -> list[str]:
    """Split long generated text into readable sections."""
    paragraphs = [
        p.strip()
        for p in text.split("\n")
        if p.strip()
    ]

    parts: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(current) + len(paragraph) + 1 > max_chars:
            if current:
                parts.append(current)

            current = paragraph
        else:
            current = (
                f"{current}\n{paragraph}".strip()
            )

    if current:
        parts.append(current)

    return parts


def safe_paragraph(text: str) -> str:
    return (
        safe_text(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br/>")
    )


# ============================================================
# IMAGE HELPERS
# ============================================================


def download_image(
    url: str,
    timeout: int = 15,
    max_size: int = 1800,
) -> bytes | None:
    """
    Download and validate an image.

    Returns optimized JPEG bytes or None.
    """
    if not url:
        return None

    try:
        response = requests.get(
            url,
            timeout=timeout,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 "
                    "MarineWiseAI/1.0"
                )
            },
        )

        response.raise_for_status()

        content_type = (
            response.headers.get("content-type", "")
            .lower()
        )

        if "image" not in content_type:
            return None

        image = Image.open(
            io.BytesIO(response.content)
        )

        image = ImageOps.exif_transpose(image)

        if image.mode not in ("RGB", "RGBA"):
            image = image.convert("RGB")

        image.thumbnail(
            (max_size, max_size),
            Image.Resampling.LANCZOS,
        )

        output = io.BytesIO()

        if image.mode == "RGBA":
            background = Image.new(
                "RGB",
                image.size,
                "white",
            )
            background.paste(
                image,
                mask=image.getchannel("A"),
            )
            image = background

        image.save(
            output,
            format="JPEG",
            quality=88,
            optimize=True,
        )

        return output.getvalue()

    except Exception:
        return None


def image_dimensions(
    image_bytes: bytes,
) -> tuple[int, int]:
    try:
        image = Image.open(
            io.BytesIO(image_bytes)
        )
        return image.size
    except Exception:
        return 1, 1


def unique_image_urls(
    research: dict[str, Any],
    limit: int = 6,
) -> list[dict[str, str]]:
    """
    Extract unique images from the research bundle.
    """
    images = research.get("images", []) or []

    output: list[dict[str, str]] = []
    seen: set[str] = set()

    for item in images:
        if isinstance(item, str):
            url = item
            title = ""
            source = ""
        else:
            url = safe_text(item.get("url"))
            title = safe_text(item.get("title"))
            source = safe_text(item.get("source"))

        if not url or url in seen:
            continue

        seen.add(url)

        output.append(
            {
                "url": url,
                "title": title,
                "source": source,
            }
        )

        if len(output) >= limit:
            break

    return output


# ============================================================
# MANUAL SOURCE DISPLAY
# ============================================================


def render_sources(
    results: list[dict[str, Any]],
) -> None:
    if not results:
        return

    st.markdown("**Manual sources retrieved:**")

    unique: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()

    for item in results:
        source = safe_text(item.get("source"))
        page = int(item.get("page", 0))

        key = (source, page)

        if key not in seen:
            seen.add(key)
            unique.append(key)

    for source, page in unique:
        st.markdown(
            (
                f'<span class="source-tag">'
                f"{source} — page {page}"
                f"</span>"
            ),
            unsafe_allow_html=True,
        )


def get_manual_page_images(
    rag: dict[str, Any] | None,
    sources: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Retrieve stored page images from the RAG index.
    """
    if not rag:
        return []

    page_images = rag.get("page_images", {})

    output: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()

    for source in sources:
        name = safe_text(source.get("source"))
        page = int(source.get("page", 0))

        key = (name, page)

        if key in seen:
            continue

        seen.add(key)

        image_bytes = page_images.get(key)

        if image_bytes:
            output.append(
                {
                    "source": name,
                    "page": page,
                    "bytes": image_bytes,
                }
            )

    return output


# ============================================================
# SIDEBAR
# ============================================================


def sidebar_manuals() -> None:
    st.sidebar.markdown(
        """
        <div style="
            text-align:center;
            padding:12px 4px 18px 4px;
        ">
            <div style="font-size:40px;">⚓</div>
            <div style="
                font-size:25px;
                font-weight:800;
            ">
                MarineWise AI
            </div>
            <div style="
                font-size:12px;
                opacity:0.85;
            ">
                Marine Engine Intelligence
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("### AI Provider")

    provider = st.sidebar.radio(
        "Choose AI provider",
        ["Groq", "Gemini"],
        index=(
            0
            if st.session_state.get(
                "ai_provider",
                "Groq",
            )
            == "Groq"
            else 1
        ),
        key="ai_provider",
    )

    key_name = (
        "GROQ_API_KEY"
        if provider == "Groq"
        else "GEMINI_API_KEY"
    )

    if get_secret(key_name):
        st.sidebar.success(
            f"{provider} API key configured ✓"
        )
    else:
        st.sidebar.warning(
            f"Add {key_name} to Streamlit Secrets."
        )

    st.sidebar.markdown("### Manuals")

    uploads = st.sidebar.file_uploader(
        "Upload PDF manuals",
        type=["pdf"],
        accept_multiple_files=True,
        help=(
            "Manuals are indexed only when you "
            "click Build / Rebuild FAISS Index."
        ),
    )

    drive_url = st.sidebar.text_input(
        "Or paste a public Google Drive PDF link",
        placeholder="https://drive.google.com/file/d/...",
    )

    if st.sidebar.button(
        "Build / Rebuild FAISS Index",
        type="primary",
        use_container_width=True,
    ):
        items: list[tuple[str, bytes]] = []

        if uploads:
            items.extend(
                (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                )
                for uploaded_file in uploads
            )

        if drive_url.strip():
            try:
                with st.spinner(
                    "Downloading the Google Drive PDF..."
                ):
                    items.append(
                        download_google_drive_pdf(
                            drive_url.strip()
                        )
                    )

            except Exception as exc:
                st.sidebar.error(
                    f"Drive download failed: {exc}"
                )

        if not items:
            st.sidebar.warning(
                "Upload a PDF or provide a Google Drive "
                "PDF link first."
            )

        else:
            try:
                with st.spinner(
                    "Reading manuals, extracting page visuals, "
                    "and building the FAISS index..."
                ):
                    st.session_state.rag = build_index(
                        items,
                        get_embedder(),
                    )

                st.session_state.last_retrieved = []

                st.sidebar.success(
                    f"Indexed {len(items)} manual(s)."
                )

            except Exception as exc:
                st.sidebar.error(
                    f"Indexing failed: {exc}"
                )

    rag = st.session_state.get("rag")

    if rag:
        st.sidebar.success(
            f"Index ready: {len(rag['records'])} chunks"
        )

        st.sidebar.caption(
            f"Chunk size: {rag['chunk_size']} characters"
        )

        st.sidebar.caption(
            f"Manual page visuals: "
            f"{len(rag.get('page_images', {}))}"
        )

        st.sidebar.caption(
            f"AI provider: {selected_provider()}"
        )

    else:
        st.sidebar.info(
            "No manual index yet."
        )


# ============================================================
# TROUBLESHOOTING AGENT
# ============================================================


def make_context_query(
    manufacturer: str,
    model: str,
    alarm: str,
) -> str:
    return (
        f"Marine engine manufacturer {manufacturer}; "
        f"engine model {model}; "
        f"defect/alarm: {alarm}"
    )


def troubleshooting_page() -> None:
    st.markdown(
        '<div class="main-title">Troubleshooting Agent</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="main-subtitle">'
        "Manual-grounded marine engine troubleshooting"
        "</div>",
        unsafe_allow_html=True,
    )

    with st.form("troubleshoot_form"):
        manufacturer = st.selectbox(
            "Manufacturer",
            [
                "CAT",
                "MTU",
                "YANMAR",
                "YAMAHA",
                "HONDA",
                "MAN",
                "OTHER",
            ],
        )

        engine_model = st.text_input(
            "Engine Model",
            placeholder="e.g. MTU 16V 4000 M90",
        )

        serial = st.text_input(
            "Serial Number (optional)"
        )

        defect = st.text_area(
            "Defect or Alarm",
            placeholder=(
                "Describe the symptom, alarm code, "
                "and what happened."
            ),
        )

        submitted = st.form_submit_button(
            "Troubleshoot",
            type="primary",
        )

    if submitted:
        if (
            not engine_model.strip()
            or not defect.strip()
        ):
            st.warning(
                "Please enter the engine model "
                "and defect/alarm."
            )
            return

        rag = st.session_state.get("rag")

        if not rag:
            st.warning(
                "No manuals are indexed. "
                "Build the FAISS index first."
            )
            return

        st.session_state.troubleshooting_case = {
            "manufacturer": manufacturer,
            "engine_model": engine_model.strip(),
            "serial": serial.strip(),
            "defect": defect.strip(),
        }

        st.session_state.troubleshooting_answer = None
        st.session_state.troubleshooting_web_answer = None
        st.session_state.troubleshooting_web_choice = None

        query = make_context_query(
            manufacturer,
            engine_model,
            defect,
        )

        with st.spinner(
            "Searching the supplied manuals..."
        ):
            results = search_index(
                rag,
                query,
                get_embedder(),
                k=6,
            )

            context, relevant = retrieve_context(
                results,
                min_score=0.32,
                max_chunks=2,
                max_chars=3000,
            )

            st.session_state.troubleshooting_context = context
            st.session_state.troubleshooting_sources = relevant
            st.session_state.last_retrieved = relevant

        if context:
            with st.spinner(
                "Preparing a manual-grounded answer..."
            ):
                answer = run_agent(
                    """
You are a marine engine troubleshooting specialist.

Use ONLY the supplied manual excerpts.

Do not invent:
- specifications
- causes
- alarm limits
- procedures
- component locations
- maintenance intervals

If the excerpts do not actually answer the question, return exactly:

Not found in manuals. Do you want me to search online?

Cite the source file and exact page number
for claims that are supported by the excerpts.
""",
                    (
                        f"Manufacturer: {manufacturer}\n"
                        f"Engine model: {engine_model}\n"
                        f"Serial: {serial or 'not provided'}\n"
                        f"Defect/alarm: {defect}\n\n"
                        f"MANUAL EXCERPTS:\n{context}"
                    ),
                    "troubleshooting",
                )

            st.session_state.troubleshooting_answer = answer

        else:
            st.session_state.troubleshooting_answer = (
                "Not found in manuals. "
                "Do you want me to search online?"
            )

    case = st.session_state.get(
        "troubleshooting_case"
    )

    if not case:
        return

    relevant = st.session_state.get(
        "troubleshooting_sources",
        [],
    )

    context = st.session_state.get(
        "troubleshooting_context",
        "",
    )

    answer = st.session_state.get(
        "troubleshooting_answer"
    )

    if context or relevant:
        rag = st.session_state.get("rag")

        if rag:
            st.caption(
                f"Chunk size: {rag['chunk_size']} characters • "
                f"Retrieved pages: "
                f"{len({r['page'] for r in relevant})}"
            )

        render_sources(relevant)

    if answer:
        st.markdown("### Troubleshooting answer")
        st.write(answer)

    not_found_phrase = (
        "Not found in manuals. "
        "Do you want me to search online?"
    )

    manual_not_found = (
        not context
        or not answer
        or not_found_phrase.lower()
        in answer.lower()
    )

    if (
        manual_not_found
        and st.session_state.get(
            "troubleshooting_web_answer"
        )
        is None
    ):
        st.warning(not_found_phrase)

        choice = st.session_state.get(
            "troubleshooting_web_choice"
        )

        if choice is None:
            st.write(
                "Would you like MarineWise AI to search "
                "reliable online technical sources?"
            )

            yes_col, no_col = st.columns(2)

            if yes_col.button(
                "Yes — Search Online",
                key="web_troubleshoot_yes",
                type="primary",
            ):
                st.session_state.troubleshooting_web_choice = (
                    "yes"
                )
                st.rerun()

            if no_col.button(
                "No — Stay Manual-Only",
                key="web_troubleshoot_no",
            ):
                st.session_state.troubleshooting_web_choice = (
                    "no"
                )
                st.rerun()

            return

        if choice == "no":
            st.info(
                "Online search was not requested. "
                "No web information was used."
            )
            return

        if choice == "yes":
            with st.spinner(
                "Searching online technical sources..."
            ):
                web_answer = ask_web(
                    """
You are a marine engine troubleshooting assistant.

This answer is FROM THE WEB, not from the supplied manuals.

Search reliable manufacturer documentation and reputable
technical sources.

Do not invent specifications.

Clearly say that the answer is from the web and instruct
the technician to verify it against the current engine manual.
""",
                    (
                        f"Find reliable information for "
                        f"{case['manufacturer']} "
                        f"{case['engine_model']}, "
                        f"serial "
                        f"{case['serial'] or 'not provided'}, "
                        f"symptom/alarm: "
                        f"{case['defect']}.\n\n"
                        "Explain likely checks and safe next steps."
                    ),
                )

            st.session_state.troubleshooting_web_answer = (
                web_answer
            )

    web_answer = st.session_state.get(
        "troubleshooting_web_answer"
    )

    if web_answer:
        st.markdown("### Web-sourced answer")

        st.info(
            "This answer was obtained from online sources "
            "after you selected **Yes — Search Online**. "
            "Verify it against the current engine manual."
        )

        st.write(web_answer)


# ============================================================
# PROFESSIONAL DIAGRAM GENERATORS
# ============================================================


def make_system_diagram(
    engine: str,
    topic: str,
) -> bytes:
    """
    Generate a clean marine-system training diagram.
    """
    from matplotlib.patches import FancyBboxPatch
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(
        figsize=(13.33, 5.8)
    )

    ax.set_xlim(0, 13.33)
    ax.set_ylim(0, 5.8)
    ax.axis("off")

    boxes = [
        (
            0.5,
            2.0,
            2.3,
            1.25,
            "INPUT",
            "Fuel / Air /\nCooling Medium",
        ),
        (
            3.5,
            2.0,
            2.3,
            1.25,
            "CONTROL",
            "Sensors /\nECU / Controls",
        ),
        (
            6.5,
            2.0,
            2.3,
            1.25,
            "ENGINE",
            "Combustion /\nMechanical Output",
        ),
        (
            9.5,
            2.0,
            2.3,
            1.25,
            "MONITOR",
            "Sensors /\nAlarm / Feedback",
        ),
    ]

    for (
        x,
        y,
        width,
        height,
        title,
        subtitle,
    ) in boxes:
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.06",
            linewidth=1.8,
        )

        ax.add_patch(patch)

        ax.text(
            x + width / 2,
            y + 0.82,
            title,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
        )

        ax.text(
            x + width / 2,
            y + 0.38,
            subtitle,
            ha="center",
            va="center",
            fontsize=9,
        )

    for x in [2.8, 5.8, 8.8]:
        ax.annotate(
            "",
            xy=(x + 0.5, 2.62),
            xytext=(x, 2.62),
            arrowprops={
                "arrowstyle": "->",
                "lw": 2,
            },
        )

    ax.text(
        6.66,
        5.05,
        f"{engine} — {topic}",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
    )

    ax.text(
        6.66,
        4.45,
        "Simplified training architecture — verify details against the engine manual",
        ha="center",
        va="center",
        fontsize=9,
    )

    buffer = io.BytesIO()

    fig.savefig(
        buffer,
        format="png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    return buffer.getvalue()


def make_process_diagram(
    steps: list[str],
    title: str = "Technical Process",
) -> bytes:
    from matplotlib.patches import FancyBboxPatch
    import matplotlib.pyplot as plt

    steps = steps[:6]

    fig, ax = plt.subplots(
        figsize=(13.33, 5.5)
    )

    ax.set_xlim(0, 13.33)
    ax.set_ylim(0, 5.5)
    ax.axis("off")

    if not steps:
        steps = [
            "Prepare",
            "Inspect",
            "Measure",
            "Verify",
            "Correct",
            "Test",
        ]

    width = 1.7
    gap = 0.35
    start_x = 0.45
    y = 2.0

    for index, step in enumerate(steps):
        x = start_x + index * (width + gap)

        patch = FancyBboxPatch(
            (x, y),
            width,
            1.3,
            boxstyle="round,pad=0.05",
            linewidth=1.5,
        )

        ax.add_patch(patch)

        ax.text(
            x + width / 2,
            y + 0.82,
            f"{index + 1}",
            ha="center",
            va="center",
            fontsize=15,
            fontweight="bold",
        )

        wrapped = "\n".join(
            textwrap.wrap(
                step,
                width=17,
            )
        )

        ax.text(
            x + width / 2,
            y + 0.35,
            wrapped,
            ha="center",
            va="center",
            fontsize=8.5,
        )

        if index < len(steps) - 1:
            ax.annotate(
                "",
                xy=(
                    x + width + gap - 0.05,
                    y + 0.65,
                ),
                xytext=(
                    x + width + 0.05,
                    y + 0.65,
                ),
                arrowprops={
                    "arrowstyle": "->",
                    "lw": 1.7,
                },
            )

    ax.text(
        6.66,
        4.7,
        title,
        ha="center",
        va="center",
        fontsize=17,
        fontweight="bold",
    )

    buffer = io.BytesIO()

    fig.savefig(
        buffer,
        format="png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    return buffer.getvalue()


def make_training_diagram(
    engine: str,
    topic: str,
) -> bytes:
    return make_system_diagram(
        engine,
        topic,
    )


# ============================================================
# POWERPOINT HELPERS
# ============================================================


def add_slide_background(
    slide,
    color: str = LIGHT,
) -> None:
    from pptx.dml.color import RGBColor
    from pptx.enum.shapes import MSO_SHAPE
    from pptx.util import Inches

    shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        0,
        0,
        Inches(13.333),
        Inches(7.5),
    )

    shape.fill.solid()

    shape.fill.fore_color.rgb = RGBColor.from_string(
        color
    )

    shape.line.fill.background()

    slide.shapes._spTree.remove(shape._element)

    slide.shapes._spTree.insert(
        2,
        shape._element,
    )


def add_top_bar(
    slide,
    title: str,
    section: str = "",
) -> None:
    from pptx.dml.color import RGBColor
    from pptx.enum.shapes import MSO_SHAPE
    from pptx.util import Inches, Pt

    # Navy title bar
    bar = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        0,
        0,
        Inches(13.333),
        Inches(0.78),
    )

    bar.fill.solid()

    bar.fill.fore_color.rgb = RGBColor.from_string(
        NAVY
    )

    bar.line.fill.background()

    title_box = slide.shapes.add_textbox(
        Inches(0.55),
        Inches(0.15),
        Inches(9.8),
        Inches(0.42),
    )

    paragraph = title_box.text_frame.paragraphs[0]

    paragraph.text = title
    paragraph.font.size = Pt(25)
    paragraph.font.bold = True
    paragraph.font.color.rgb = RGBColor.from_string(
        WHITE
    )

    if section:
        section_box = slide.shapes.add_textbox(
            Inches(10.3),
            Inches(0.18),
            Inches(2.4),
            Inches(0.35),
        )

        paragraph = (
            section_box.text_frame.paragraphs[0]
        )

        paragraph.text = section.upper()
        paragraph.font.size = Pt(9)
        paragraph.font.bold = True
        paragraph.font.color.rgb = RGBColor.from_string(
            "B9D7EA"
        )

        paragraph.alignment = 2


def add_footer(
    slide,
    slide_number: int,
    source_text: str = "",
) -> None:
    from pptx.dml.color import RGBColor
    from pptx.util import Inches, Pt

    footer = slide.shapes.add_textbox(
        Inches(0.55),
        Inches(7.12),
        Inches(11.6),
        Inches(0.22),
    )

    paragraph = footer.text_frame.paragraphs[0]

    paragraph.text = source_text[:170]

    paragraph.font.size = Pt(7)
    paragraph.font.color.rgb = RGBColor.from_string(
        GRAY
    )

    number = slide.shapes.add_textbox(
        Inches(12.35),
        Inches(7.08),
        Inches(0.45),
        Inches(0.25),
    )

    p = number.text_frame.paragraphs[0]

    p.text = str(slide_number)
    p.font.size = Pt(8)
    p.font.bold = True
    p.font.color.rgb = RGBColor.from_string(
        NAVY
    )

    p.alignment = 2


def add_bullets(
    slide,
    bullets: list[str],
    left: float = 0.8,
    top: float = 1.5,
    width: float = 5.5,
    height: float = 4.8,
    font_size: int = 19,
) -> None:
    from pptx.dml.color import RGBColor
    from pptx.util import Inches, Pt

    box = slide.shapes.add_textbox(
        Inches(left),
        Inches(top),
        Inches(width),
        Inches(height),
    )

    tf = box.text_frame
    tf.clear()

    for index, bullet in enumerate(bullets[:7]):
        paragraph = (
            tf.paragraphs[0]
            if index == 0
            else tf.add_paragraph()
        )

        paragraph.text = safe_text(bullet)
        paragraph.font.size = Pt(font_size)
        paragraph.font.color.rgb = RGBColor.from_string(
            DARK
        )
        paragraph.space_after = Pt(12)

        paragraph.level = 0

        paragraph.text = (
            "• " + paragraph.text
        )


def add_callout(
    slide,
    title: str,
    body: str,
    left: float,
    top: float,
    width: float,
    height: float,
    fill_color: str = LIGHT_TEAL,
) -> None:
    from pptx.dml.color import RGBColor
    from pptx.enum.shapes import MSO_SHAPE
    from pptx.util import Inches, Pt

    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(left),
        Inches(top),
        Inches(width),
        Inches(height),
    )

    shape.fill.solid()

    shape.fill.fore_color.rgb = RGBColor.from_string(
        fill_color
    )

    shape.line.color.rgb = RGBColor.from_string(
        LIGHT_GRAY
    )

    tf = shape.text_frame
    tf.clear()

    p = tf.paragraphs[0]
    p.text = title
    p.font.bold = True
    p.font.size = Pt(15)
    p.font.color.rgb = RGBColor.from_string(
        NAVY
    )

    p2 = tf.add_paragraph()
    p2.text = body
    p2.font.size = Pt(11)
    p2.font.color.rgb = RGBColor.from_string(
        DARK
    )


def add_picture_contain(
    slide,
    image_bytes: bytes,
    left: float,
    top: float,
    width: float,
    height: float,
) -> None:
    """
    Add a picture while maintaining aspect ratio.
    """
    from pptx.util import Inches

    image = Image.open(
        io.BytesIO(image_bytes)
    )

    image_width, image_height = image.size

    target_width = width
    target_height = height

    source_ratio = (
        image_width / image_height
    )

    target_ratio = (
        target_width / target_height
    )

    if source_ratio > target_ratio:
        final_width = target_width
        final_height = (
            target_width / source_ratio
        )

        final_left = left
        final_top = (
            top
            + (target_height - final_height)
            / 2
        )

    else:
        final_height = target_height
        final_width = (
            target_height * source_ratio
        )

        final_top = top
        final_left = (
            left
            + (target_width - final_width)
            / 2
        )

    slide.shapes.add_picture(
        io.BytesIO(image_bytes),
        Inches(final_left),
        Inches(final_top),
        width=Inches(final_width),
        height=Inches(final_height),
    )


def add_picture_card(
    slide,
    image_bytes: bytes,
    title: str,
    caption: str,
    left: float,
    top: float,
    width: float,
    height: float,
) -> None:
    from pptx.dml.color import RGBColor
    from pptx.enum.shapes import MSO_SHAPE
    from pptx.util import Inches, Pt

    card = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(left),
        Inches(top),
        Inches(width),
        Inches(height),
    )

    card.fill.solid()

    card.fill.fore_color.rgb = RGBColor.from_string(
        WHITE
    )

    card.line.color.rgb = RGBColor.from_string(
        LIGHT_GRAY
    )

    add_picture_contain(
        slide,
        image_bytes,
        left + 0.15,
        top + 0.15,
        width - 0.3,
        height - 0.95,
    )

    title_box = slide.shapes.add_textbox(
        Inches(left + 0.18),
        Inches(top + height - 0.75),
        Inches(width - 0.36),
        Inches(0.25),
    )

    p = title_box.text_frame.paragraphs[0]

    p.text = title[:75]
    p.font.bold = True
    p.font.size = Pt(10)
    p.font.color.rgb = RGBColor.from_string(
        NAVY
    )

    caption_box = slide.shapes.add_textbox(
        Inches(left + 0.18),
        Inches(top + height - 0.48),
        Inches(width - 0.36),
        Inches(0.28),
    )

    cp = caption_box.text_frame.paragraphs[0]

    cp.text = caption[:110]
    cp.font.size = Pt(7)
    cp.font.color.rgb = RGBColor.from_string(
        GRAY
    )


def add_process_shapes(
    slide,
    steps: list[str],
    left: float = 0.65,
    top: float = 2.1,
) -> None:
    from pptx.dml.color import RGBColor
    from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE
    from pptx.enum.shapes import MSO_CONNECTOR
    from pptx.util import Inches, Pt

    steps = steps[:5]

    width = 2.25
    gap = 0.3

    for index, step in enumerate(steps):
        x = left + index * (
            width + gap
        )

        shape = slide.shapes.add_shape(
            MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE,
            Inches(x),
            Inches(top),
            Inches(width),
            Inches(1.35),
        )

        shape.fill.solid()

        shape.fill.fore_color.rgb = RGBColor.from_string(
            LIGHT_TEAL
        )

        shape.line.color.rgb = RGBColor.from_string(
            TEAL
        )

        tf = shape.text_frame
        tf.clear()

        p = tf.paragraphs[0]

        p.text = str(index + 1)
        p.font.bold = True
        p.font.size = Pt(18)
        p.font.color.rgb = RGBColor.from_string(
            NAVY
        )
        p.alignment = 1

        p2 = tf.add_paragraph()

        p2.text = safe_text(step)
        p2.font.size = Pt(11)
        p2.font.bold = True
        p2.font.color.rgb = RGBColor.from_string(
            DARK
        )
        p2.alignment = 1

        if index < len(steps) - 1:
            connector = slide.shapes.add_connector(
                MSO_CONNECTOR.STRAIGHT,
                Inches(
                    x + width
                ),
                Inches(
                    top + 0.68
                ),
                Inches(
                    x + width + gap
                ),
                Inches(
                    top + 0.68
                ),
            )

            connector.line.color.rgb = RGBColor.from_string(
                BLUE
            )

            connector.line.width = Pt(2)


# ============================================================
# TRAINING PLAN PARSER
# ============================================================


def parse_training_plan(
    raw: Any,
) -> list[dict[str, Any]]:
    """
    Try to parse the Training Agent JSON output.

    The agent is instructed to produce JSON, but this
    function also has a fallback for imperfect output.
    """
    import json

    # The selected AI provider may return a string, dict, or list.
    # Normalize it before calling string methods.
    if isinstance(raw, (dict, list)):
        if isinstance(raw, dict):
            raw_for_fallback = raw.get("slides", raw)
        else:
            raw_for_fallback = raw
        cleaned = json.dumps(raw)
    else:
        raw_for_fallback = safe_text(raw)
        cleaned = raw_for_fallback

    if "```json" in cleaned:
        cleaned = cleaned.split(
            "```json",
            1,
        )[1]

        cleaned = cleaned.split(
            "```",
            1,
        )[0]

    elif "```" in cleaned:
        cleaned = cleaned.replace(
            "```",
            "",
        )

    try:
        parsed = json.loads(cleaned)

        if isinstance(parsed, dict):
            slides = parsed.get(
                "slides",
                [],
            )
        else:
            slides = parsed

        if isinstance(slides, list):
            valid: list[dict[str, Any]] = []

            for item in slides:
                if isinstance(item, dict):
                    valid.append(item)

            if valid:
                return valid

    except Exception:
        pass

    # Fallback
    if isinstance(raw_for_fallback, list):
        sections = [
            safe_text(item)
            for item in raw_for_fallback
            if safe_text(item)
        ]
    elif isinstance(raw_for_fallback, dict):
        sections = [safe_text(raw_for_fallback)]
    else:
        sections = split_text(
            safe_text(raw_for_fallback),
            700,
        )

    slides: list[dict[str, Any]] = []

    for index, section in enumerate(
        sections[:8],
        start=1,
    ):
        slides.append(
            {
                "title": f"Training — Part {index}",
                "purpose": "Technical explanation",
                "bullets": [
                    section
                ],
                "visual_type": "none",
                "visual_query": "",
                "source_preference": "manual",
                "speaker_note": "",
            }
        )

    return slides


# ============================================================
# PROFESSIONAL TRAINING PPT
# ============================================================


def make_professional_training_ppt(
    engine: str,
    ship: str,
    topic: str,
    plan: list[dict[str, Any]],
    manual_images: list[dict[str, Any]],
    online_images: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    web_sources: list[dict[str, Any]],
) -> bytes:
    """
    Build a professional 16:9 training presentation.
    """
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN
    from pptx.util import Inches, Pt

    prs = Presentation()

    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    slide = prs.slides.add_slide(
        prs.slide_layouts[6]
    )

    add_slide_background(
        slide,
        NAVY,
    )

    # Accent line
    accent = slide.shapes.add_shape(
        1,
        Inches(0.7),
        Inches(1.0),
        Inches(1.5),
        Inches(0.12),
    )

    accent.fill.solid()
    accent.fill.fore_color.rgb = RGBColor.from_string(
        TEAL
    )
    accent.line.fill.background()

    title_box = slide.shapes.add_textbox(
        Inches(0.7),
        Inches(1.45),
        Inches(11.7),
        Inches(1.5),
    )

    p = title_box.text_frame.paragraphs[0]

    p.text = topic
    p.font.size = Pt(38)
    p.font.bold = True
    p.font.color.rgb = RGBColor.from_string(
        WHITE
    )

    subtitle = slide.shapes.add_textbox(
        Inches(0.75),
        Inches(3.15),
        Inches(10.8),
        Inches(1.2),
    )

    p = subtitle.text_frame.paragraphs[0]

    p.text = (
        f"{engine}"
        + (
            f"  |  {ship}"
            if ship
            else ""
        )
    )

    p.font.size = Pt(22)
    p.font.color.rgb = RGBColor.from_string(
        "C9D9E6"
    )

    tag = slide.shapes.add_textbox(
        Inches(0.75),
        Inches(5.5),
        Inches(7),
        Inches(0.5),
    )

    p = tag.text_frame.paragraphs[0]

    p.text = (
        "MarineWise AI • Technical Training"
    )

    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = RGBColor.from_string(
        "9CCFD2"
    )

    add_footer(
        slide,
        1,
        "Generated from supplied manuals and technical research.",
    )

    # --------------------------------------------------------
    # OBJECTIVES
    # --------------------------------------------------------

    slide = prs.slides.add_slide(
        prs.slide_layouts[6]
    )

    add_slide_background(
        slide
    )

    add_top_bar(
        slide,
        "Learning Objectives",
        "Training",
    )

    objectives = [
        "Understand the purpose of the system.",
        "Identify the major components and their roles.",
        "Trace the basic operating sequence.",
        "Recognize common technician checks and fault symptoms.",
        "Apply safe inspection and verification practices.",
    ]

    add_bullets(
        slide,
        objectives,
        left=0.85,
        top=1.55,
        width=11.4,
        height=4.9,
        font_size=20,
    )

    add_footer(
        slide,
        2,
        f"{engine} • {topic}",
    )

    # --------------------------------------------------------
    # SYSTEM OVERVIEW
    # --------------------------------------------------------

    slide = prs.slides.add_slide(
        prs.slide_layouts[6]
    )

    add_slide_background(
        slide
    )

    add_top_bar(
        slide,
        "System Overview",
        "Understand",
    )

    if manual_images:
        image = manual_images[0]

        add_picture_card(
            slide,
            image["bytes"],
            "Manual reference",
            (
                f"{image['source']} — "
                f"page {image['page']}"
            ),
            0.7,
            1.35,
            7.1,
            5.15,
        )

        add_callout(
            slide,
            "Technician focus",
            (
                "Use the manual illustration to identify "
                "components before beginning inspection "
                "or troubleshooting."
            ),
            8.15,
            1.55,
            4.3,
            1.6,
        )

        add_callout(
            slide,
            "Source",
            (
                f"{image['source']} — "
                f"page {image['page']}"
            ),
            8.15,
            3.55,
            4.3,
            1.2,
        )

    else:
        diagram = make_system_diagram(
            engine,
            topic,
        )

        add_picture_contain(
            slide,
            diagram,
            0.7,
            1.25,
            11.9,
            4.9,
        )

    add_footer(
        slide,
        3,
        (
            f"Manual visual: "
            f"{manual_images[0]['source']} "
            f"page {manual_images[0]['page']}"
            if manual_images
            else "Generated technical overview diagram."
        ),
    )

    # --------------------------------------------------------
    # PROCESS / HOW IT WORKS
    # --------------------------------------------------------

    slide = prs.slides.add_slide(
        prs.slide_layouts[6]
    )

    add_slide_background(
        slide
    )

    add_top_bar(
        slide,
        "How the System Works",
        "Principle",
    )

    process_steps = [
        "Input / supply",
        "Control / regulation",
        "Engine operation",
        "Sensor feedback",
        "Monitoring / alarm",
    ]

    # Use AI plan information if available
    for item in plan:
        visual_type = safe_text(
            item.get("visual_type")
        ).lower()

        if visual_type in {
            "process",
            "flow",
            "flowchart",
            "diagram",
        }:
            candidate = item.get(
                "bullets",
                [],
            )

            if isinstance(candidate, list):
                cleaned = [
                    safe_text(x)
                    for x in candidate
                    if safe_text(x)
                ]

                if len(cleaned) >= 3:
                    process_steps = cleaned[:5]

            break

    add_process_shapes(
        slide,
        process_steps,
        left=0.65,
        top=2.1,
    )

    add_callout(
        slide,
        "Key idea",
        (
            "A technician should be able to describe "
            "what enters the system, what controls it, "
            "what the engine does with it, and how the "
            "system reports abnormal conditions."
        ),
        1.1,
        4.25,
        11.1,
        1.25,
    )

    add_footer(
        slide,
        4,
        "Simplified instructional diagram — verify exact architecture against the engine manual.",
    )

    # --------------------------------------------------------
    # COMPONENT SPOTLIGHT
    # --------------------------------------------------------

    slide = prs.slides.add_slide(
        prs.slide_layouts[6]
    )

    add_slide_background(
        slide
    )

    add_top_bar(
        slide,
        "Component Spotlight",
        "Identify",
    )

    image_candidates = (
        manual_images[:2]
        + online_images[:2]
    )

    if image_candidates:
        card_width = 5.75

        for index, item in enumerate(
            image_candidates[:2]
        ):
            left = (
                0.7
                if index == 0
                else 6.85
            )

            if "bytes" in item:
                bytes_data = item["bytes"]

                title = (
                    item.get(
                        "title",
                        "Manual reference",
                    )
                    or "Technical reference"
                )

                source = item.get(
                    "source",
                    "",
                )

                if item.get("page"):
                    source = (
                        f"{source} — "
                        f"page {item['page']}"
                    )

            else:
                bytes_data = item.get(
                    "bytes"
                )

                title = item.get(
                    "title",
                    "Online technical image",
                )

                source = item.get(
                    "source",
                    "",
                )

            if bytes_data:
                add_picture_card(
                    slide,
                    bytes_data,
                    title,
                    source,
                    left,
                    1.35,
                    card_width,
                    4.9,
                )

    else:
        add_callout(
            slide,
            "No suitable image found",
            (
                "The presentation uses clean generated diagrams "
                "when a reliable manual or web image is unavailable."
            ),
            1.0,
            2.0,
            11.0,
            2.0,
        )

    add_footer(
        slide,
        5,
        "Visuals selected from supplied manuals and technical research where available.",
    )

    # --------------------------------------------------------
    # TECHNICIAN CHECKS
    # --------------------------------------------------------

    slide = prs.slides.add_slide(
        prs.slide_layouts[6]
    )

    add_slide_background(
        slide
    )

    add_top_bar(
        slide,
        "Technician Inspection Sequence",
        "Practice",
    )

    checks = [
        "Confirm the reported symptom and alarm.",
        "Check the relevant manual section.",
        "Inspect visible components and connections.",
        "Measure or verify only with approved procedures.",
        "Correct the cause and perform a controlled test.",
    ]

    add_bullets(
        slide,
        checks,
        left=0.8,
        top=1.35,
        width=7.1,
        height=4.9,
        font_size=17,
    )

    add_callout(
        slide,
        "Before testing",
        (
            "Follow the engine manufacturer's isolation, "
            "PPE, hot-surface, pressure and rotating-equipment "
            "requirements."
        ),
        8.2,
        1.7,
        4.0,
        2.0,
        "FFF3E6",
    )

    add_callout(
        slide,
        "Record findings",
        (
            "Capture measurements, alarm codes, observed "
            "conditions and the manual page used."
        ),
        8.2,
        4.05,
        4.0,
        1.7,
        LIGHT_TEAL,
    )

    add_footer(
        slide,
        6,
        "General technician workflow — use the current engine manual for exact procedures.",
    )

    # --------------------------------------------------------
    # FAULTS / TROUBLESHOOTING
    # --------------------------------------------------------

    slide = prs.slides.add_slide(
        prs.slide_layouts[6]
    )

    add_slide_background(
        slide
    )

    add_top_bar(
        slide,
        "Common Fault-Diagnosis Logic",
        "Troubleshoot",
    )

    fault_rows = [
        (
            "Symptom",
            "Where to look",
            "Verify",
        ),
        (
            "Abnormal temperature",
            "Cooling circuit",
            "Flow / level / sensor",
        ),
        (
            "Low pressure",
            "Pump / supply",
            "Pressure / restriction",
        ),
        (
            "Poor engine response",
            "Fuel / air / control",
            "Signals / filters / supply",
        ),
        (
            "Alarm indication",
            "Sensor / monitored system",
            "Alarm source and manual procedure",
        ),
    ]

    from pptx.dml.color import RGBColor
    from pptx.enum.shapes import MSO_SHAPE
    from pptx.util import Inches, Pt

    start_x = 0.7
    start_y = 1.35
    col_widths = [
        3.7,
        3.7,
        4.2,
    ]

    row_height = 0.85

    for row_index, row in enumerate(
        fault_rows
    ):
        x = start_x

        for col_index, value in enumerate(row):
            shape = slide.shapes.add_shape(
                MSO_SHAPE.RECTANGLE,
                Inches(x),
                Inches(
                    start_y
                    + row_index * row_height
                ),
                Inches(
                    col_widths[col_index]
                ),
                Inches(row_height),
            )

            shape.fill.solid()

            shape.fill.fore_color.rgb = (
                RGBColor.from_string(
                    NAVY
                    if row_index == 0
                    else WHITE
                )
            )

            shape.line.color.rgb = (
                RGBColor.from_string(
                    LIGHT_GRAY
                )
            )

            tf = shape.text_frame
            tf.clear()

            p = tf.paragraphs[0]
            p.text = value
            p.font.size = Pt(
                12
                if row_index
                else 13
            )
            p.font.bold = (
                row_index == 0
            )

            p.font.color.rgb = (
                RGBColor.from_string(
                    WHITE
                    if row_index == 0
                    else DARK
                )
            )

            p.alignment = 1

            x += col_widths[col_index]

    add_callout(
        slide,
        "Important",
        (
            "The table is a training framework, not a substitute "
            "for manufacturer-specific fault trees."
        ),
        1.0,
        6.0,
        11.0,
        0.75,
        "FFF3E6",
    )

    add_footer(
        slide,
        7,
        "Use manufacturer fault-finding procedures and specified measurements.",
    )

    # --------------------------------------------------------
    # SAFETY
    # --------------------------------------------------------

    slide = prs.slides.add_slide(
        prs.slide_layouts[6]
    )

    add_slide_background(
        slide
    )

    add_top_bar(
        slide,
        "Safety Before Maintenance",
        "Safety",
    )

    safety_points = [
        "Apply the required isolation / lockout procedure.",
        "Treat hot coolant, oil and engine surfaces as hazardous.",
        "Beware of pressurized systems and rotating machinery.",
        "Use appropriate PPE and approved test equipment.",
        "Follow the current manufacturer safety instructions.",
    ]

    add_bullets(
        slide,
        safety_points,
        left=0.8,
        top=1.4,
        width=7.5,
        height=4.8,
        font_size=17,
    )

    add_callout(
        slide,
        "STOP",
        (
            "Do not begin a maintenance procedure when the "
            "equipment state, isolation status or safe procedure "
            "is uncertain."
        ),
        8.65,
        2.0,
        3.6,
        2.0,
        "FFF1F1",
    )

    add_footer(
        slide,
        8,
        "Safety guidance must be checked against the current engine and vessel procedures.",
    )

    # --------------------------------------------------------
    # KNOWLEDGE CHECK
    # --------------------------------------------------------

    slide = prs.slides.add_slide(
        prs.slide_layouts[6]
    )

    add_slide_background(
        slide
    )

    add_top_bar(
        slide,
        "Technician Knowledge Check",
        "Assess",
    )

    questions = [
        "1. What is the primary purpose of the system?",
        "2. Which component should be checked first for the reported symptom?",
        "3. What measurement or observation confirms the suspected condition?",
        "4. What safety precaution must be completed before inspection?",
    ]

    add_bullets(
        slide,
        questions,
        left=0.8,
        top=1.45,
        width=11.5,
        height=4.6,
        font_size=17,
    )

    add_callout(
        slide,
        "Trainer prompt",
        (
            "Ask the technician to explain the reasoning, "
            "not only give the component name."
        ),
        1.1,
        5.8,
        11.0,
        0.8,
        LIGHT_TEAL,
    )

    add_footer(
        slide,
        9,
        f"Training topic: {topic}",
    )

    # --------------------------------------------------------
    # REFERENCES
    # --------------------------------------------------------

    slide = prs.slides.add_slide(
        prs.slide_layouts[6]
    )

    add_slide_background(
        slide
    )

    add_top_bar(
        slide,
        "References & Image Credits",
        "Sources",
    )

    reference_lines: list[str] = []

    seen_manual: set[str] = set()

    for source in sources:
        line = (
            f"Manual: {source.get('source')} "
            f"— page {source.get('page')}"
        )

        if line not in seen_manual:
            seen_manual.add(line)
            reference_lines.append(line)

    seen_web: set[str] = set()

    for source in web_sources:
        url = safe_text(
            source.get("url")
        )

        title = safe_text(
            source.get("title")
        )

        if url and url not in seen_web:
            seen_web.add(url)

            reference_lines.append(
                f"Web: {title or 'Technical source'} — {url}"
            )

    if online_images:
        for image in online_images:
            url = safe_text(
                image.get("url")
            )

            title = safe_text(
                image.get("title")
            )

            if url:
                reference_lines.append(
                    "Image: "
                    f"{title or 'Online technical image'} "
                    f"— {url}"
                )

    if not reference_lines:
        reference_lines = [
            "No external references were returned.",
            "Use the current engine manufacturer's manual.",
        ]

    # Keep reference slide readable.
    chunks = split_text(
        "\n".join(
            f"• {line}"
            for line in reference_lines
        ),
        1700,
    )

    box = slide.shapes.add_textbox(
        Inches(0.7),
        Inches(1.25),
        Inches(11.9),
        Inches(5.65),
    )

    tf = box.text_frame
    tf.clear()

    for index, chunk in enumerate(
        chunks[:4]
    ):
        p = (
            tf.paragraphs[0]
            if index == 0
            else tf.add_paragraph()
        )

        p.text = chunk
        p.font.size = Pt(9)
        p.font.color.rgb = RGBColor.from_string(
            DARK
        )

        p.space_after = Pt(8)

    add_footer(
        slide,
        10,
        "MarineWise AI — verify technical content against current manufacturer documentation.",
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output = io.BytesIO()

    prs.save(output)

    return output.getvalue()


# ============================================================
# TRAINING MATERIAL PAGE
# ============================================================


def training_material_page() -> None:
    st.markdown(
        '<div class="main-title">'
        "2A. Professional Training Material"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="main-subtitle">'
        "Manual-first RAG + Tavily research + professional PowerPoint generation"
        "</div>",
        unsafe_allow_html=True,
    )

    with st.form(
        "professional_training_material_form"
    ):
        engine = st.text_input(
            "Engine Model",
            placeholder="MTU 16V 4000 M90",
        )

        ship = st.text_input(
            "Ship",
            placeholder="MV Example",
        )

        topic = st.text_input(
            "Training Topic",
            placeholder="Cooling System",
        )

        c1, c2, c3 = st.columns(3)

        make_pdf = c1.checkbox(
            "PDF",
            True,
        )

        make_ppt = c2.checkbox(
            "Professional PPT",
            True,
        )

        make_word = c3.checkbox(
            "Word",
            False,
        )

        submitted = st.form_submit_button(
            "Generate Professional Training",
            type="primary",
        )

    if not submitted:
        return

    if not engine.strip() or not topic.strip():
        st.warning(
            "Enter an engine model and training topic."
        )
        return

    if not any(
        [
            make_pdf,
            make_ppt,
            make_word,
        ]
    ):
        st.warning(
            "Select at least one output format."
        )
        return

    rag = st.session_state.get("rag")

    context = ""
    relevant: list[dict[str, Any]] = []

    # --------------------------------------------------------
    # MANUAL RETRIEVAL
    # --------------------------------------------------------

    if rag:
        with st.spinner(
            "Step 1/4 — Searching uploaded manuals..."
        ):
            results = search_index(
                rag,
                f"{engine} {topic}",
                get_embedder(),
                k=8,
            )

            context, relevant = retrieve_context(
                results,
                min_score=0.28,
                max_chunks=6,
                max_chars=7000,
            )

            st.session_state.last_retrieved = relevant

    else:
        st.info(
            "No manual index is available. "
            "Training will use web research and general "
            "technical knowledge. Build the FAISS index "
            "first for manual-grounded training."
        )

    manual_images = get_manual_page_images(
        rag,
        relevant,
    )

    # --------------------------------------------------------
    # ONLINE RESEARCH
    # --------------------------------------------------------

    research: dict[str, Any] = {
        "results": [],
        "images": [],
    }

    tavily_key = get_secret(
        "TAVILY_API_KEY"
    )

    if tavily_key:
        with st.spinner(
            "Step 2/4 — Researching reliable technical sources online..."
        ):
            research = run_training_web_research(
                selected_provider(),
                engine,
                topic,
                ship,
                context,
                tavily_key,
            )

    else:
        st.warning(
            "TAVILY_API_KEY is not configured. "
            "Online training research and online images "
            "will be skipped."
        )

    web_sources = research.get(
        "results",
        [],
    ) or []

    image_candidates = unique_image_urls(
        research,
        limit=6,
    )

    # --------------------------------------------------------
    # DOWNLOAD ONLINE IMAGES
    # --------------------------------------------------------

    online_images: list[dict[str, Any]] = []

    if image_candidates:
        with st.spinner(
            "Preparing relevant technical images..."
        ):
            for image_info in image_candidates:
                image_bytes = download_image(
                    image_info["url"]
                )

                if image_bytes:
                    online_images.append(
                        {
                            **image_info,
                            "bytes": image_bytes,
                        }
                    )

                if len(online_images) >= 4:
                    break

    # --------------------------------------------------------
    # TRAINING PLAN
    # --------------------------------------------------------

    with st.spinner(
        "Step 3/4 — Designing the professional presentation..."
    ):
        plan_raw = generate_training_presentation_plan(
            selected_provider(),
            engine,
            ship,
            topic,
            context,
            research,
        )

        plan = parse_training_plan(
            plan_raw
        )

    # --------------------------------------------------------
    # CONTENT DISPLAY
    # --------------------------------------------------------

    st.markdown("### Research Summary")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Manual pages",
        len(
            {
                (
                    item.get("source"),
                    item.get("page"),
                )
                for item in relevant
            }
        ),
    )

    c2.metric(
        "Manual visuals",
        len(manual_images),
    )

    c3.metric(
        "Web sources",
        len(web_sources),
    )

    c4.metric(
        "Online images",
        len(online_images),
    )

    render_sources(relevant)

    if web_sources:
        st.markdown("**Web research sources:**")

        for source in web_sources[:8]:
            title = safe_text(
                source.get("title")
            )

            url = safe_text(
                source.get("url")
            )

            if title:
                st.markdown(
                    f'<span class="research-tag">'
                    f"{title[:100]}"
                    f"</span>",
                    unsafe_allow_html=True,
                )

    # --------------------------------------------------------
    # VISUAL PREVIEW
    # --------------------------------------------------------

    if manual_images or online_images:
        st.markdown("### Visual Research")

        preview_items = (
            manual_images[:2]
            + online_images[:2]
        )

        cols = st.columns(
            min(
                len(preview_items),
                4,
            )
        )

        for column, item in zip(
            cols,
            preview_items,
        ):
            with column:
                if "bytes" in item:
                    st.image(
                        item["bytes"],
                        use_container_width=True,
                    )

                if item.get("page"):
                    st.caption(
                        f"{item.get('source')} — "
                        f"page {item.get('page')}"
                    )
                else:
                    st.caption(
                        item.get(
                            "title",
                            "Online technical image",
                        )
                    )

    # --------------------------------------------------------
    # PRESENTATION PLAN PREVIEW
    # --------------------------------------------------------

    st.markdown(
        "### Professional Presentation Plan"
    )

    for index, slide_plan in enumerate(
        plan[:10],
        start=1,
    ):
        title = safe_text(
            slide_plan.get(
                "title",
                f"Slide {index}",
            )
        )

        purpose = safe_text(
            slide_plan.get(
                "purpose",
                "",
            )
        )

        visual_type = safe_text(
            slide_plan.get(
                "visual_type",
                "none",
            )
        )

        bullets = slide_plan.get(
            "bullets",
            [],
        )

        if not isinstance(
            bullets,
            list,
        ):
            bullets = [str(bullets)]

        bullet_text = "<br>".join(
            f"• {safe_text(b)}"
            for b in bullets[:5]
        )

        st.markdown(
            f"""
            <div class="slide-card">
                <b>{index}. {title}</b><br>
                <small>{purpose}</small><br><br>
                {bullet_text}<br>
                <small>
                    Visual: {visual_type}
                </small>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------
    # GENERATE PPT
    # --------------------------------------------------------

    with st.spinner(
        "Step 4/4 — Building professional PowerPoint..."
    ):
        ppt_bytes = make_professional_training_ppt(
            engine,
            ship,
            topic,
            plan,
            manual_images,
            online_images,
            relevant,
            web_sources,
        )

    # --------------------------------------------------------
    # TRAINING TEXT
    # --------------------------------------------------------

    with st.spinner(
        "Preparing the detailed training explanation..."
    ):
        content = run_agent(
            """
You are a senior marine technical training instructor.

Create concise but useful training material for technicians.

Use supplied manual excerpts as the primary source.

Use web evidence only as secondary supporting information.

Do not invent manufacturer-specific specifications.

Clearly distinguish:
1. Manual-supported information
2. General technical explanation
3. Web-supported information

Cover:
- learning objectives
- system overview
- operating principle
- component functions
- technician inspection
- common fault logic
- safety
- knowledge check

Keep the material practical and understandable.
""",
            (
                f"Engine: {engine}\n"
                f"Ship: {ship or 'Not specified'}\n"
                f"Topic: {topic}\n\n"
                f"MANUAL EXCERPTS:\n{context}\n\n"
                f"WEB RESEARCH:\n"
                f"{web_sources[:8]}"
            ),
            "training",
        )

    st.markdown(
        "### Training Explanation"
    )

    st.write(content)

    # --------------------------------------------------------
    # GENERIC DIAGRAM
    # --------------------------------------------------------

    diagram = make_training_diagram(
        engine,
        topic,
    )

    st.markdown(
        "### Generated Technical Diagram"
    )

    st.image(
        diagram,
        caption=(
            "Simplified training diagram. "
            "Verify engine-specific architecture "
            "against the current manual."
        ),
        use_container_width=True,
    )

    # --------------------------------------------------------
    # DOWNLOADS
    # --------------------------------------------------------

    st.markdown(
        "### Download Training Material"
    )

    download_columns = st.columns(3)

    if make_ppt:
        download_columns[0].download_button(
            "⬇ Download Professional PowerPoint",
            ppt_bytes,
            "marinewise_professional_training.pptx",
            (
                "application/vnd.openxmlformats-officedocument."
                "presentationml.presentation"
            ),
            use_container_width=True,
        )

    if make_pdf:
        download_columns[1].download_button(
            "⬇ Download PDF",
            make_training_pdf(
                engine,
                ship,
                topic,
                content,
                relevant,
                diagram,
            ),
            "marinewise_training.pdf",
            "application/pdf",
            use_container_width=True,
        )

    if make_word:
        download_columns[2].download_button(
            "⬇ Download Word",
            make_training_docx(
                engine,
                ship,
                topic,
                content,
                diagram,
            ),
            "marinewise_training.docx",
            (
                "application/vnd.openxmlformats-officedocument."
                "wordprocessingml.document"
            ),
            use_container_width=True,
        )


# ============================================================
# QUIZ GENERATOR
# ============================================================


def quiz_page() -> None:
    st.subheader(
        "2B. Quiz Generator"
    )

    st.caption(
        "Generate technically relevant marine technician questions "
        "using manual RAG + online technical research."
    )

    with st.form("quiz_form"):
        topic = st.text_input(
            "Quiz Topic",
            placeholder="Fuel Injection System",
        )

        qtype = st.selectbox(
            "Question Type",
            [
                "MCQ",
                "Short Question",
                "True-False",
            ],
        )

        count = st.number_input(
            "Number of Questions",
            min_value=1,
            max_value=30,
            value=10,
            step=1,
        )

        submitted = st.form_submit_button(
            "Generate Technical Quiz",
            type="primary",
        )

    if not submitted:
        return

    topic = topic.strip()

    if not topic:
        st.warning(
            "Enter a specific technical topic."
        )
        return

    rag = st.session_state.get(
        "rag"
    )

    context = ""
    relevant: list[dict[str, Any]] = []

    # --------------------------------------------------------
    # STEP 1 — MANUAL RAG RETRIEVAL
    # --------------------------------------------------------

    if rag:
        with st.spinner(
            "Step 1/4 — Searching technical manuals..."
        ):
            results = search_index(
                rag,
                topic,
                get_embedder(),
                k=8,
            )

            context, relevant = retrieve_context(
                results,
                min_score=0.28,
                max_chunks=5,
                max_chars=6000,
            )

            st.session_state.last_retrieved = relevant
    else:
        st.info(
            "No FAISS manual index is available. "
            "The quiz will rely on online technical research "
            "and the AI's technical knowledge."
        )

    # --------------------------------------------------------
    # STEP 2 — ONLINE TECHNICAL RESEARCH
    # --------------------------------------------------------

    research: dict[str, Any] = {
        "results": [],
        "images": [],
    }

    tavily_key = get_secret(
        "TAVILY_API_KEY"
    )

    if tavily_key and run_training_web_research:
        with st.spinner(
            "Step 2/4 — Researching reliable technical sources online..."
        ):
            try:
                research = run_training_web_research(
                    selected_provider(),
                    "quiz",
                    topic,
                    "",
                    context,
                    tavily_key,
                )
            except Exception as exc:
                st.warning(
                    f"Online research could not be completed: {exc}"
                )
                research = {
                    "results": [],
                    "images": [],
                }
    else:
        if not tavily_key:
            st.warning(
                "TAVILY_API_KEY is not configured. "
                "Online technical research will be skipped."
            )

    web_sources = (
        research.get(
            "results",
            [],
        )
        or []
    )

    # --------------------------------------------------------
    # PREPARE WEB RESEARCH TEXT
    # --------------------------------------------------------

    web_context_parts: list[str] = []

    for index, source in enumerate(
        web_sources[:10],
        start=1,
    ):
        title = safe_text(
            source.get(
                "title",
                "",
            )
        )

        url = safe_text(
            source.get(
                "url",
                "",
            )
        )

        content = safe_text(
            source.get(
                "content",
                source.get(
                    "snippet",
                    "",
                ),
            )
        )

        if not content:
            content = safe_text(
                source.get(
                    "description",
                    "",
                )
            )

        if title or content:
            web_context_parts.append(
                (
                    f"WEB SOURCE {index}\n"
                    f"Title: {title}\n"
                    f"URL: {url}\n"
                    f"Technical content:\n{content[:2500]}"
                )
            )

    web_context = "\n\n".join(
        web_context_parts
    )

    # --------------------------------------------------------
    # STEP 3 — GENERATE QUIZ
    # --------------------------------------------------------

    with st.spinner(
        "Step 3/4 — Generating technically focused questions..."
    ):
        quiz_system_prompt = """
You are the MarineWise Technical Assessment Agent.

Create a professional marine technician technical assessment.

The requested topic is the PRIMARY constraint.

Every question MUST directly test the requested topic.

IMPORTANT RELEVANCE RULES:

1. Do NOT create a question merely because the topic happens
   to be mentioned in a source.

2. Every question must test actual technical knowledge,
   operation, components, diagnosis, maintenance, safety,
   failure modes, inspection, or troubleshooting that is
   directly related to the requested topic.

3. Do NOT ask generic marine-engine questions unless they
   directly relate to the requested topic.

4. Do NOT ask questions about unrelated systems.

5. Do NOT ask questions about the source document itself.

6. Do NOT ask questions such as:
   - According to the manual...
   - On page X...
   - Which manual states...
   unless the question itself tests useful technical knowledge.

7. Do not invent manufacturer-specific specifications,
   torque values, pressures, clearances, temperatures,
   part numbers, or limits.

8. If a manufacturer-specific value is required but is not
   supported by the supplied evidence, avoid asking for
   that exact value.

9. Prefer technically meaningful questions over trivia.

10. Use the uploaded manual evidence as the primary source
    when relevant.

11. Use reliable online technical research as secondary
    supporting evidence.

12. If a web source is only loosely related to the topic,
    DO NOT use it to create a question.

13. The requested number of questions MUST be produced.

14. The requested question type MUST be followed exactly.

15. Each question must have one clearly defensible answer.

16. MCQ distractors must be technically plausible but wrong.

17. True-False statements must be technically precise.

18. Short questions must have a concise technically defensible answer.

OUTPUT FORMAT:

QUESTION 1:
<question>

For MCQ:
A. <option>
B. <option>
C. <option>
D. <option>

QUESTION 2:
<question>

...

ANSWER KEY:
1. <answer>
2. <answer>
...

EXPLANATIONS:
1. <brief technical explanation>
2. <brief technical explanation>
...

Do not add an introduction before QUESTION 1.
Do not put unrelated commentary between questions.
Do not use markdown tables.
"""

        quiz_user_prompt = (
            f"REQUESTED TOPIC:\n"
            f"{topic}\n\n"
            f"QUESTION TYPE:\n"
            f"{qtype}\n\n"
            f"NUMBER OF QUESTIONS:\n"
            f"{int(count)}\n\n"
            f"==================================================\n"
            f"MANUAL RAG EVIDENCE\n"
            f"==================================================\n"
            f"{context or 'No manual evidence available.'}\n\n"
            f"==================================================\n"
            f"ONLINE TECHNICAL RESEARCH\n"
            f"==================================================\n"
            f"{web_context or 'No online research available.'}\n\n"
            f"==================================================\n"
            f"FINAL REQUIREMENT\n"
            f"==================================================\n"
            f"Generate exactly {int(count)} {qtype} questions "
            f"that directly assess: {topic}\n\n"
            f"Before producing each question, internally verify "
            f"that the question is specifically about {topic} "
            f"and not merely associated with it."
        )

        quiz_text = run_agent(
            quiz_system_prompt,
            quiz_user_prompt,
            "training",
        )

    # --------------------------------------------------------
    # STEP 4 — TECHNICAL REVIEW
    # --------------------------------------------------------

    with st.spinner(
        "Step 4/4 — Verifying technical relevance..."
    ):
        review_system_prompt = """
You are the MarineWise Technical Quiz Reviewer.

Review the generated quiz against the requested topic and the
supplied technical evidence.

Your job is NOT to rewrite the quiz.

Check every question.

A question is VALID only if:

1. It directly tests the requested topic.
2. It is technically meaningful.
3. It is not merely based on a source mentioning the topic.
4. It is not unrelated marine-engine knowledge.
5. The answer is technically defensible.
6. It does not rely on an unsupported manufacturer-specific value.
7. MCQ questions have one clearly correct answer.
8. True/False questions are technically unambiguous.
9. Short questions have a defensible answer.
10. The requested question count and type are satisfied.

Return exactly this structure:

VALID: YES
or
VALID: NO

PROBLEMS:
- <problem 1>
- <problem 2>

REPAIR INSTRUCTIONS:
- <specific instructions for correcting invalid questions>

If all questions are technically relevant and valid,
return VALID: YES.
"""

        review_user_prompt = (
            f"REQUESTED TOPIC:\n"
            f"{topic}\n\n"
            f"QUESTION TYPE:\n"
            f"{qtype}\n\n"
            f"REQUESTED COUNT:\n"
            f"{int(count)}\n\n"
            f"MANUAL EVIDENCE:\n"
            f"{context or 'None'}\n\n"
            f"ONLINE TECHNICAL EVIDENCE:\n"
            f"{web_context or 'None'}\n\n"
            f"GENERATED QUIZ:\n"
            f"{quiz_text}"
        )

        review_text = run_agent(
            review_system_prompt,
            review_user_prompt,
            "training",
        )

    # --------------------------------------------------------
    # REGENERATE IF REVIEW FAILS
    # --------------------------------------------------------

    if "VALID: NO" in review_text.upper():
        with st.spinner(
            "Improving questions that failed the technical review..."
        ):
            repair_system_prompt = """
You are the MarineWise Technical Assessment Agent.

Regenerate the quiz using the reviewer's feedback.

The requested topic is the strict primary constraint.
Every question must directly test the requested topic.

Do not use questions merely because a source mentions the
requested topic.

Remove unrelated questions.

Keep the requested question type and exact question count.

Do not invent unsupported manufacturer-specific values.

Output ONLY:

QUESTION 1:
...

QUESTION 2:
...

ANSWER KEY:
1. ...
2. ...

EXPLANATIONS:
1. ...
2. ...

No introduction.
No conclusion.
No review commentary.
"""

            repair_user_prompt = (
                f"REQUESTED TOPIC:\n"
                f"{topic}\n\n"
                f"QUESTION TYPE:\n"
                f"{qtype}\n\n"
                f"QUESTION COUNT:\n"
                f"{int(count)}\n\n"
                f"MANUAL EVIDENCE:\n"
                f"{context or 'None'}\n\n"
                f"ONLINE TECHNICAL RESEARCH:\n"
                f"{web_context or 'None'}\n\n"
                f"ORIGINAL QUIZ:\n"
                f"{quiz_text}\n\n"
                f"TECHNICAL REVIEW:\n"
                f"{review_text}\n\n"
                f"Regenerate the complete quiz now."
            )

            quiz_text = run_agent(
                repair_system_prompt,
                repair_user_prompt,
                "training",
            )

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    st.markdown(
        "### Technical Quiz"
    )

    st.markdown(
        f"**Topic:** {topic}"
    )

    st.write(
        quiz_text
    )

    # --------------------------------------------------------
    # DOWNLOAD PDF
    # --------------------------------------------------------

    pdf_bytes = make_quiz_pdf(
        topic,
        qtype,
        quiz_text,
    )

    st.download_button(
        "Download Technical Assessment / Quiz PDF",
        pdf_bytes,
        "marinewise_technical_assessment_quiz.pdf",
        "application/pdf",
        use_container_width=True,
    )

    # --------------------------------------------------------
    # SOURCES
    # --------------------------------------------------------

    if relevant:
        st.markdown(
            "### Manual Sources Used"
        )
        render_sources(
            relevant
        )

    if web_sources:
        st.markdown(
            "### Online Technical Sources"
        )

        for source in web_sources[:10]:
            title = safe_text(
                source.get(
                    "title",
                    "Technical source",
                )
            )

            url = safe_text(
                source.get(
                    "url",
                    "",
                )
            )

            if url:
                st.markdown(
                    f"- [{title}]({url})"
                )
            else:
                st.markdown(
                    f"- {title}"
                )



def assessment_page() -> None:
    st.subheader(
        "2C. Score an Assessment"
    )

    st.caption(
        "Upload a clear assessment image and provide its answer key."
    )

    uploaded = st.file_uploader(
        "Assessment image",
        type=[
            "jpg",
            "jpeg",
            "png",
        ],
    )

    answer_key = st.text_area(
        "Answer key",
        placeholder=(
            "Example: 1=A, 2=C, 3=True, 4=B"
        ),
    )

    if uploaded and st.button(
        "Score Assessment",
        type="primary",
    ):
        with st.spinner(
            "Reading the assessment image..."
        ):
            try:
                import pytesseract

                image = Image.open(
                    uploaded
                ).convert("RGB")

                st.image(
                    image,
                    caption="Uploaded assessment",
                    width=500,
                )

                extracted = (
                    pytesseract.image_to_string(
                        image
                    )
                )

            except ImportError:
                st.error(
                    "OCR is not installed. "
                    "Install Tesseract OCR and pytesseract."
                )
                return

            except Exception as exc:
                st.error(
                    f"OCR could not run: {exc}"
                )
                return

        st.text_area(
            "OCR text — check this before scoring",
            extracted,
            height=250,
        )

        if not answer_key.strip():
            st.warning(
                "Add an answer key so the app can calculate a defensible score."
            )
            return

        with st.spinner(
            "Scoring assessment..."
        ):
            score = score_with_agent(
                extracted,
                answer_key,
            )

        st.markdown(
            "### Assessment Result"
        )

        st.write(
            score["feedback"]
        )

        st.metric(
            "Score",
            f"{score['score']:.0f}%",
        )

        if score["score"] < 50:
            st.warning(
                "Below 50% — remedial training is recommended."
            )

            with st.spinner(
                "Generating remedial presentation..."
            ):
                remedial = run_agent(
                    """
Create a short remedial marine technician training
presentation outline based only on the assessment mistakes.

Include:
- 5–7 slides
- explanations
- practice checks
- final retest

Do not invent technical values.
""",
                    (
                        f"Assessment OCR:\n{extracted}\n\n"
                        f"Answer key:\n{answer_key}\n\n"
                        f"Scoring feedback:\n"
                        f"{score['feedback']}"
                    ),
                    "training",
                )

            st.download_button(
                "Download Remedial Training PPT",
                make_remedial_ppt(
                    remedial
                ),
                "marinewise_remedial_training.pptx",
                (
                    "application/vnd.openxmlformats-officedocument."
                    "presentationml.presentation"
                ),
            )


def score_with_agent(
    extracted: str,
    answer_key: str,
) -> dict[str, Any]:
    raw = run_agent(
        """
Score a technician assessment.

Return exactly three lines:

SCORE_PERCENT: number
FEEDBACK: concise explanation
MISSED_TOPICS: comma-separated topics

Do not guess unreadable answers.
""",
        (
            f"OCR answers:\n{extracted}\n\n"
            f"Answer key:\n{answer_key}"
        ),
        "training",
    )

    match = re.search(
        r"SCORE_PERCENT:\s*"
        r"([0-9]+(?:\.[0-9]+)?)",
        raw,
        re.I,
    )

    score = (
        float(match.group(1))
        if match
        else 0.0
    )

    if "FEEDBACK:" in raw:
        feedback = raw.split(
            "FEEDBACK:",
            1,
        )[1]

        if "MISSED_TOPICS:" in feedback:
            feedback = feedback.split(
                "MISSED_TOPICS:",
                1,
            )[0]

        feedback = feedback.strip()

    else:
        feedback = raw

    return {
        "score": max(
            0.0,
            min(
                100.0,
                score,
            ),
        ),
        "feedback": feedback,
    }


# ============================================================
# LEARNING PAGE
# ============================================================


def learning_page() -> None:
    st.markdown(
        '<div class="main-title">'
        "Learning: RAG & MarineWise Architecture"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="main-subtitle">'
        "Understand how MarineWise turns manuals into useful technician guidance"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "### What is RAG?"
    )

    st.write(
        "Retrieval-Augmented Generation (RAG) first searches "
        "your documents and then supplies the most relevant "
        "passages to the AI model. This helps keep answers "
        "grounded in the uploaded manuals."
    )

    st.markdown(
        "### Advanced RAG"
    )

    st.write(
        "Advanced RAG can add query rewriting, metadata "
        "filters, reranking, multiple retrieval steps, "
        "better chunking and visual-document retrieval."
    )

    st.markdown(
        "### Chunks"
    )

    st.write(
        "A chunk is a small piece of a manual. "
        "MarineWise currently uses approximately "
        "900 characters with overlap so related text "
        "is not unnecessarily split apart."
    )

    st.markdown(
        "### FAISS"
    )

    st.write(
        "FAISS is a vector-search library. Each manual "
        "chunk becomes an embedding, and FAISS finds "
        "chunks that are mathematically close to the "
        "technician's question."
    )

    st.code(
        """
PDF Manuals
     ↓
Pages
     ↓
Text + Page Images
     ↓
Chunks
     ↓
Embeddings
     ↓
FAISS
     ↓
Technician Question
     ↓
Relevant Manual Pages
     ↓
MarineWise AI
     ↓
Answer / Training / Presentation
""",
        language="text",
    )

    rag = st.session_state.get(
        "rag"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Chunk size",
        rag["chunk_size"]
        if rag
        else DEFAULT_CHUNK_SIZE,
    )

    c2.metric(
        "Number of chunks",
        len(
            rag["records"]
        )
        if rag
        else 0,
    )

    c3.metric(
        "Manual page visuals",
        len(
            rag.get(
                "page_images",
                {},
            )
        )
        if rag
        else 0,
    )

    c4.metric(
        "Retrieved pages",
        len(
            {
                r["page"]
                for r in st.session_state.get(
                    "last_retrieved",
                    [],
                )
            }
        ),
    )

    if rag:
        st.caption(
            f"Embedding model: "
            f"{rag['embedding_model']} • "
            f"Overlap: "
            f"{rag['chunk_overlap']} characters"
        )


# ============================================================
# PDF / WORD OUTPUTS
# ============================================================


def make_training_pdf(
    engine: str,
    ship: str,
    topic: str,
    content: str,
    sources: list[dict[str, Any]],
    diagram: bytes,
) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Image as RLImage,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
    )

    output = io.BytesIO()

    document = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    story = [
        Paragraph(
            "MarineWise AI — Technical Training",
            styles["Title"],
        ),
        Paragraph(
            safe_paragraph(
                f"Engine: {engine} | "
                f"Ship: {ship or 'Not specified'} | "
                f"Topic: {topic}"
            ),
            styles["Heading2"],
        ),
        Spacer(1, 10),
        RLImage(
            io.BytesIO(diagram),
            width=6.8 * inch,
            height=3.06 * inch,
        ),
    ]

    for part in split_text(content):
        story.extend(
            [
                Paragraph(
                    safe_paragraph(part),
                    styles["BodyText"],
                ),
                Spacer(1, 8),
            ]
        )

    if sources:
        story.append(
            Paragraph(
                "Manual Sources",
                styles["Heading2"],
            )
        )

        for source in sources:
            story.append(
                Paragraph(
                    safe_paragraph(
                        f"{source['source']} — "
                        f"page {source['page']}"
                    ),
                    styles["BodyText"],
                )
            )

    document.build(
        story
    )

    return output.getvalue()


def make_training_docx(
    engine: str,
    ship: str,
    topic: str,
    content: str,
    diagram: bytes,
) -> bytes:
    from docx import Document
    from docx.shared import Inches

    document = Document()

    document.add_heading(
        "MarineWise AI — Technical Training",
        0,
    )

    document.add_paragraph(
        f"Engine: {engine} | "
        f"Ship: {ship or 'Not specified'} | "
        f"Topic: {topic}"
    )

    document.add_picture(
        io.BytesIO(diagram),
        width=Inches(6.5),
    )

    for part in split_text(content):
        document.add_paragraph(
            part
        )

    output = io.BytesIO()

    document.save(
        output
    )

    return output.getvalue()


def make_quiz_pdf(
    topic: str,
    qtype: str,
    text: str,
) -> bytes:
    import os
    import re
    import io

    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        PageBreak,
        KeepTogether,
    )

    output = io.BytesIO()

    document = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=22 * mm,
        leftMargin=22 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
        title="Technical Assessment / Quiz",
        author="MarineWise AI",
    )

    # --------------------------------------------------------
    # FONT
    # --------------------------------------------------------

    font_name = "Times-Roman"
    font_bold = "Times-Bold"

    times_regular_paths = [
        r"C:\Windows\Fonts\times.ttf",
        r"C:\Windows\Fonts\Times_New_Roman.ttf",
    ]

    times_bold_paths = [
        r"C:\Windows\Fonts\timesbd.ttf",
        r"C:\Windows\Fonts\Times_New_Roman_Bold.ttf",
    ]

    for path in times_regular_paths:
        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(
                    TTFont(
                        "MarineWiseTimes",
                        path,
                    )
                )
                font_name = "MarineWiseTimes"
                break
            except Exception:
                pass

    for path in times_bold_paths:
        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(
                    TTFont(
                        "MarineWiseTimesBold",
                        path,
                    )
                )
                font_bold = "MarineWiseTimesBold"
                break
            except Exception:
                pass

    # --------------------------------------------------------
    # CLEAN TEXT
    # --------------------------------------------------------

    def clean_pdf_text(value: Any) -> str:
        value = str(
            value or ""
        )

        replacements = {
            "\u25a0": "",
            "\u25aa": "",
            "\u25ab": "",
            "\u25cf": "",
            "\u2022": "",
            "\u00a0": " ",
            "\u200b": "",
            "\u200c": "",
            "\u200d": "",
            "\ufeff": "",
            "\u2013": "-",
            "\u2014": "-",
            "\u2018": "'",
            "\u2019": "'",
            "\u201c": '"',
            "\u201d": '"',
            "\u2026": "...",
            "\u2192": "->",
            "\u2190": "<-",
            "\u00b0": " degrees",
        }

        for old, new in replacements.items():
            value = value.replace(
                old,
                new,
            )

        value = "".join(
            char
            for char in value
            if char == "\n"
            or char == "\t"
            or ord(char) >= 32
        )

        value = re.sub(
            r"[ \t]+",
            " ",
            value,
        )

        value = re.sub(
            r"\n{3,}",
            "\n\n",
            value,
        )

        return value.strip()

    def pdf_paragraph_text(value: str) -> str:
        value = clean_pdf_text(
            value
        )

        return (
            value
            .replace(
                "&",
                "&amp;",
            )
            .replace(
                "<",
                "&lt;",
            )
            .replace(
                ">",
                "&gt;",
            )
            .replace(
                "\n",
                "<br/>",
            )
        )

    # --------------------------------------------------------
    # STYLES
    # --------------------------------------------------------

    title_style = ParagraphStyle(
        "QuizTitle",
        fontName=font_bold,
        fontSize=18,
        leading=22,
        alignment=TA_CENTER,
        textColor=colors.black,
        spaceAfter=8,
    )

    topic_style = ParagraphStyle(
        "QuizTopic",
        fontName=font_bold,
        fontSize=14,
        leading=18,
        alignment=TA_CENTER,
        textColor=colors.black,
        spaceAfter=12,
    )

    meta_style = ParagraphStyle(
        "QuizMeta",
        fontName=font_name,
        fontSize=11,
        leading=15,
        alignment=TA_CENTER,
        textColor=colors.black,
        spaceAfter=14,
    )

    question_style = ParagraphStyle(
        "QuizQuestion",
        fontName=font_bold,
        fontSize=14,
        leading=19,
        alignment=0,
        textColor=colors.black,
        spaceBefore=8,
        spaceAfter=7,
    )

    option_style = ParagraphStyle(
        "QuizOption",
        fontName=font_name,
        fontSize=14,
        leading=19,
        alignment=0,
        textColor=colors.black,
        leftIndent=14,
        spaceAfter=4,
    )

    answer_style = ParagraphStyle(
        "QuizAnswer",
        fontName=font_bold,
        fontSize=14,
        leading=19,
        alignment=0,
        textColor=colors.black,
        spaceBefore=5,
        spaceAfter=6,
    )

    explanation_style = ParagraphStyle(
        "QuizExplanation",
        fontName=font_name,
        fontSize=12,
        leading=17,
        alignment=0,
        textColor=colors.black,
        leftIndent=12,
        spaceAfter=8,
    )

    answer_heading_style = ParagraphStyle(
        "AnswerHeading",
        fontName=font_bold,
        fontSize=16,
        leading=20,
        alignment=TA_CENTER,
        textColor=colors.black,
        spaceAfter=14,
    )

    # --------------------------------------------------------
    # PARSE GENERATED QUIZ
    # --------------------------------------------------------

    cleaned = clean_pdf_text(
        text
    )

    cleaned = re.sub(
        r"(?im)^\s*#+\s*ANSWER\s+KEY\s*:?\s*$",
        "ANSWER KEY:",
        cleaned,
    )

    cleaned = re.sub(
        r"(?im)^\s*#+\s*EXPLANATIONS?\s*:?\s*$",
        "EXPLANATIONS:",
        cleaned,
    )

    answer_key_match = re.search(
        r"(?im)^\s*ANSWER\s+KEY\s*:?\s*$",
        cleaned,
    )

    explanations_match = re.search(
        r"(?im)^\s*EXPLANATIONS?\s*:?\s*$",
        cleaned,
    )

    if answer_key_match:
        questions_part = cleaned[
            :answer_key_match.start()
        ]
        remaining = cleaned[
            answer_key_match.end():
        ]
    else:
        questions_part = cleaned
        remaining = ""

    if explanations_match and (
        not answer_key_match
        or explanations_match.start()
        >= answer_key_match.end()
    ):
        relative_start = (
            explanations_match.start()
            - (
                answer_key_match.end()
                if answer_key_match
                else 0
            )
        )

        answer_part = remaining[
            :relative_start
        ]

        explanation_part = remaining[
            relative_start
            + len(
                explanations_match.group(0)
            ):]
    else:
        answer_part = remaining
        explanation_part = ""

    # --------------------------------------------------------
    # QUESTION PARSER
    # --------------------------------------------------------

    question_pattern = re.compile(
        r"(?im)"
        r"^\s*(?:QUESTION\s*)?"
        r"(\d+)"
        r"\s*[:.)-]\s*"
        r"(.*?)(?="
        r"^\s*(?:QUESTION\s*)?"
        r"\d+"
        r"\s*[:.)-]"
        r"|\Z)",
        re.MULTILINE
        | re.DOTALL,
    )

    question_matches = list(
        question_pattern.finditer(
            questions_part
        )
    )

    questions: list[dict[str, Any]] = []

    for match in question_matches:
        number = match.group(1)
        body = clean_pdf_text(
            match.group(2)
        )

        if not body:
            continue

        option_matches = list(
            re.finditer(
                r"(?im)(?:^|\n)\s*"
                r"([A-D])\s*[\.\):\-]\s*"
                r"(.*?)(?="
                r"\n\s*[A-D]\s*[\.\):\-]\s*"
                r"|\Z)",
                body,
                re.DOTALL,
            )
        )

        options: list[str] = []

        if option_matches:
            first_option_position = option_matches[0].start()
            question_text = clean_pdf_text(
                body[
                    :first_option_position
                ]
            )

            for option_match in option_matches:
                letter = option_match.group(1)
                option_text = clean_pdf_text(
                    option_match.group(2)
                )
                options.append(
                    f"{letter}. {option_text}"
                )
        else:
            question_text = body

        questions.append(
            {
                "number": number,
                "question": question_text,
                "options": options,
            }
        )

    if not questions:
        raw_blocks = [
            block.strip()
            for block in re.split(
                r"\n\s*\n",
                questions_part,
            )
            if block.strip()
        ]

        for index, block in enumerate(
            raw_blocks,
            start=1,
        ):
            questions.append(
                {
                    "number": str(index),
                    "question": clean_pdf_text(
                        block
                    ),
                    "options": [],
                }
            )

    # --------------------------------------------------------
    # PARSE ANSWERS
    # --------------------------------------------------------

    answers: dict[str, str] = {}

    for line in answer_part.splitlines():
        line = clean_pdf_text(
            line
        )

        match = re.match(
            r"^\s*(\d+)\s*[\.\):\-]\s*(.+)$",
            line,
        )

        if match:
            answers[
                match.group(1)
            ] = match.group(2).strip()

    # --------------------------------------------------------
    # PARSE EXPLANATIONS
    # --------------------------------------------------------

    explanations: dict[str, str] = {}
    current_number: str | None = None

    for line in explanation_part.splitlines():
        line = clean_pdf_text(
            line
        )

        if not line:
            continue

        match = re.match(
            r"^\s*(\d+)\s*[\.\):\-]\s*(.*)$",
            line,
        )

        if match:
            current_number = match.group(1)
            explanations[
                current_number
            ] = match.group(2).strip()
        elif current_number:
            explanations[
                current_number
            ] += " " + line

    # --------------------------------------------------------
    # BUILD PDF
    # --------------------------------------------------------

    story: list[Any] = []

    story.append(
        Paragraph(
            "Technical Assessment / Quiz",
            title_style,
        )
    )

    story.append(
        Paragraph(
            pdf_paragraph_text(
                topic
            ),
            topic_style,
        )
    )

    story.append(
        Paragraph(
            pdf_paragraph_text(
                f"Question Type: {qtype}    |    "
                f"Number of Questions: {len(questions)}"
            ),
            meta_style,
        )
    )

    story.append(
        Spacer(
            1,
            5,
        )
    )

    # --------------------------------------------------------
    # QUESTIONS
    # --------------------------------------------------------

    for question in questions:
        number = question[
            "number"
        ]
        question_text = question[
            "question"
        ]
        options = question[
            "options"
        ]

        question_block: list[Any] = []

        question_block.append(
            Paragraph(
                pdf_paragraph_text(
                    f"{number}. {question_text}"
                ),
                question_style,
            )
        )

        for option in options:
            question_block.append(
                Paragraph(
                    pdf_paragraph_text(
                        option
                    ),
                    option_style,
                )
            )

        story.append(
            KeepTogether(
                question_block
            )
        )

        story.append(
            Spacer(
                1,
                5,
            )
        )

    # --------------------------------------------------------
    # ANSWER KEY — ALWAYS NEW PAGE
    # --------------------------------------------------------

    story.append(
        PageBreak()
    )

    story.append(
        Paragraph(
            "ANSWER KEY",
            answer_heading_style,
        )
    )

    if answers:
        for number in sorted(
            answers.keys(),
            key=lambda value: int(value)
            if value.isdigit()
            else 9999,
        ):
            answer = answers[
                number
            ]

            story.append(
                Paragraph(
                    pdf_paragraph_text(
                        f"{number}. {answer}"
                    ),
                    answer_style,
                )
            )

            if number in explanations:
                story.append(
                    Paragraph(
                        pdf_paragraph_text(
                            "Explanation: "
                            + explanations[number]
                        ),
                        explanation_style,
                    )
                )
    else:
        fallback_answer_text = (
            answer_part.strip()
            or "No answer key was generated."
        )

        for paragraph in re.split(
            r"\n\s*\n",
            fallback_answer_text,
        ):
            paragraph = clean_pdf_text(
                paragraph
            )

            if paragraph:
                story.append(
                    Paragraph(
                        pdf_paragraph_text(
                            paragraph
                        ),
                        answer_style,
                    )
                )

    document.build(
        story
    )

    return output.getvalue()



def main() -> None:
    sidebar_manuals()

    page = st.sidebar.radio(
        "Navigate",
        [
            "Troubleshooting Agent",
            "Training Agent",
            "Learning",
        ],
    )

    try:
        if page == "Troubleshooting Agent":
            troubleshooting_page()

        elif page == "Training Agent":
            st.markdown(
                '<div class="main-title">'
                "Technical Training Agent"
                "</div>",
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="main-subtitle">'
                "Professional technician learning materials powered by manual RAG + AI research"
                "</div>",
                unsafe_allow_html=True,
            )

            st.success(
                "MarineWise Training Agent is ready."
            )

            st.write(
                "The Training Agent can create professional "
                "training presentations, quizzes, assessment "
                "scores and remedial learning material."
            )

            tabs = st.tabs(
                [
                    "2A Professional Training",
                    "2B Quiz Generator",
                    "2C Score Assessment",
                ]
            )

            with tabs[0]:
                training_material_page()

            with tabs[1]:
                quiz_page()

            with tabs[2]:
                assessment_page()

        else:
            learning_page()

    except Exception as exc:
        import traceback

        message = str(exc)

        st.error(
            f"MarineWise AI encountered an error: {exc}"
        )

        with st.expander("Show technical error details"):
            st.code(
                traceback.format_exc(),
                language="text",
            )

        if (
            "context_length_exceeded" in message
            or "reduce the length" in message.lower()
        ):
            st.error(
                f"The request was too large for the "
                f"current {selected_provider()} request limits."
            )

            st.info(
                "Troubleshooting uses a deliberately "
                "small context budget. Training uses a "
                "separate larger context budget."
            )
        else:
            st.caption(
                "Also check that the selected provider API key, "
                "TAVILY_API_KEY, and uploaded manual are configured "
                "correctly."
            )


if __name__ == "__main__":
    main()
