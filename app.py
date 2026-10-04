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

import numpy as np

import requests
import streamlit as st
from PIL import Image, ImageOps

from agents import (
    run_marine_agent,
    run_web_search,
    run_marine_command_center,
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


def clean_output_text(value: Any) -> str:
    """Normalize AI text before placing it in professional documents."""
    import unicodedata

    text = safe_text(value)
    replacements = {
        "\u00a0": " ", "\u200b": "", "\u200c": "", "\u200d": "", "\ufeff": "",
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "-", "\u2212": "-", "\u2022": "-",
        "\u2192": "->", "\u2190": "<-", "\u2713": "[OK]", "\u2714": "[OK]",
        "\u2717": "[X]", "\u2718": "[X]", "\u00d7": "x",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    cleaned=[]
    for char in text:
        category=unicodedata.category(char)
        if category == "Cf":
            continue
        if category == "Cc" and char not in {"\n", "\t", "\r"}:
            continue
        cleaned.append(char)
    text="".join(cleaned)
    text=re.sub(r"[ \t]+", " ", text)
    text=re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _find_document_font() -> tuple[str, str | None, str | None, str | None]:
    """Prefer Times New Roman; use a serif fallback only if unavailable."""
    candidates=[
        ("Times New Roman", r"C:\\Windows\\Fonts\\times.ttf", r"C:\\Windows\\Fonts\\timesbd.ttf", r"C:\\Windows\\Fonts\\timesi.ttf"),
        ("Times New Roman", "/Library/Fonts/Times New Roman.ttf", "/Library/Fonts/Times New Roman Bold.ttf", "/Library/Fonts/Times New Roman Italic.ttf"),
        ("Times New Roman", "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf", "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf", "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Italic.ttf"),
    ]
    for name, regular, bold, italic in candidates:
        if os.path.exists(regular):
            return name, regular, bold if os.path.exists(bold) else None, italic if os.path.exists(italic) else None
    try:
        from matplotlib import font_manager
        regular=font_manager.findfont(font_manager.FontProperties(family="DejaVu Serif"))
        bold=font_manager.findfont(font_manager.FontProperties(family="DejaVu Serif", weight="bold"))
        italic=font_manager.findfont(font_manager.FontProperties(family="DejaVu Serif", style="italic"))
        return "DejaVu Serif", regular, bold, italic
    except Exception:
        return "Times-Roman", None, None, None


def _register_reportlab_fonts() -> str:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    _family, regular, bold, italic = _find_document_font()
    if regular:
        family="MarineWiseSerif"
        try:
            if family not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont(family, regular))
                if bold:
                    pdfmetrics.registerFont(TTFont(f"{family}-Bold", bold))
                if italic:
                    pdfmetrics.registerFont(TTFont(f"{family}-Italic", italic))
            return family
        except Exception:
            pass
    return "Times-Roman"


def _source_lines(sources: list[dict[str, Any]] | None = None, web_sources: list[dict[str, Any]] | None = None) -> list[str]:
    lines=[]; seen=set()
    for source in sources or []:
        name=clean_output_text(source.get("source")); page=source.get("page", "")
        if name:
            line=f"Manual: {name} - page {page}"
            if line not in seen: seen.add(line); lines.append(line)
    for source in web_sources or []:
        title=clean_output_text(source.get("title")) or "Technical web source"
        url=clean_output_text(source.get("url"))
        if url:
            line=f"Web: {title} - {url}"
            if line not in seen: seen.add(line); lines.append(line)
    return lines


def make_troubleshooting_pdf(case: dict[str, Any], manual_answer: str | None, sources: list[dict[str, Any]] | None = None, web_answer: str | None = None, web_sources: list[dict[str, Any]] | None = None) -> bytes:
    """Create a clean A4 troubleshooting PDF using 14pt justified serif body text."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak

    font=_register_reportlab_fonts()
    out=io.BytesIO()
    doc=SimpleDocTemplate(out, pagesize=A4, rightMargin=18*mm, leftMargin=18*mm, topMargin=18*mm, bottomMargin=18*mm, title="MarineWise AI Troubleshooting", author="MarineWise AI")
    title=ParagraphStyle("t_title",fontName=font,fontSize=20,leading=24,alignment=TA_CENTER,textColor=colors.HexColor("#17324D"),spaceAfter=8)
    meta=ParagraphStyle("t_meta",fontName=font,fontSize=14,leading=20,alignment=TA_CENTER,textColor=colors.HexColor("#344054"),spaceAfter=12)
    body=ParagraphStyle("t_body",fontName=font,fontSize=12,leading=17,alignment=TA_JUSTIFY,textColor=colors.HexColor("#17202A"),spaceAfter=8,allowWidows=0,allowOrphans=0)
    step=ParagraphStyle("t_step",fontName=font,fontSize=12,leading=17,alignment=TA_JUSTIFY,textColor=colors.HexColor("#17202A"),leftIndent=7*mm,firstLineIndent=-7*mm,spaceAfter=9,allowWidows=0,allowOrphans=0)
    head=ParagraphStyle("t_head",fontName=font,fontSize=17,leading=21,alignment=TA_LEFT,textColor=colors.HexColor("#17324D"),spaceBefore=10,spaceAfter=7)
    small=ParagraphStyle("t_small",fontName=font,fontSize=11,leading=16,alignment=TA_LEFT,textColor=colors.HexColor("#475467"),spaceAfter=5)
    esc=lambda x: safe_paragraph(clean_output_text(x))
    c=case
    story=[Paragraph("MarineWise AI - Step-by-Step Troubleshooting",title), Paragraph(esc(f"Manufacturer: {c.get('manufacturer','Not specified')} | Engine Model: {c.get('engine_model','Not specified')} | Serial: {c.get('serial') or 'Not provided'}"),meta), Paragraph("Reported Defect / Alarm",head), Paragraph(esc(c.get("defect")),body)]

    def append_answer(label, text):
        if not text or not clean_output_text(text): return
        story.append(Paragraph(label,head))
        for raw in clean_output_text(text).splitlines():
            line=raw.strip()
            if not line: continue
            line=re.sub(r"^#{1,6}\s*", "", line)
            line=re.sub(r"^[-*•]+\s*", "", line)
            m=re.match(r"^(\d+)[.)]\s+(.*)$", line)
            if m: story.append(Paragraph(esc(f"{m.group(1)}. {m.group(2)}"),step))
            elif len(line)<=90 and line.endswith(":"): story.append(Paragraph(esc(line[:-1]),head))
            else: story.append(Paragraph(esc(line),body))

    if manual_answer:
        append_answer("OEM / Manual-Grounded Troubleshooting", manual_answer)
    if web_answer:
        story.append(PageBreak())
        append_answer("Approved Online Supplementary Troubleshooting", web_answer)
    refs=_source_lines(sources, web_sources)
    if refs:
        story.append(PageBreak()); story.append(Paragraph("Sources & References",head))
        for ref in refs: story.append(Paragraph(esc(ref),small))
    doc.build(story)
    return out.getvalue()


def make_command_center_pdf(result: dict[str, Any]) -> bytes:
    """Create a downloadable Command Center final-response PDF."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, PageBreak

    font=_register_reportlab_fonts(); out=io.BytesIO()
    doc=SimpleDocTemplate(out,pagesize=A4,rightMargin=18*mm,leftMargin=18*mm,topMargin=18*mm,bottomMargin=18*mm,title="MarineWise AI Command Center Result",author="MarineWise AI")
    title=ParagraphStyle("cc_title",fontName=font,fontSize=20,leading=24,alignment=TA_CENTER,textColor=colors.HexColor("#17324D"),spaceAfter=10)
    head=ParagraphStyle("cc_head",fontName=font,fontSize=17,leading=21,alignment=TA_LEFT,textColor=colors.HexColor("#17324D"),spaceBefore=10,spaceAfter=7)
    body=ParagraphStyle("cc_body",fontName=font,fontSize=12,leading=17,alignment=TA_JUSTIFY,textColor=colors.HexColor("#17202A"),spaceAfter=8)
    small=ParagraphStyle("cc_small",fontName=font,fontSize=11,leading=16,alignment=TA_LEFT,textColor=colors.HexColor("#475467"),spaceAfter=5)
    esc=lambda x:safe_paragraph(clean_output_text(x))
    topic=clean_output_text(result.get("topic")) or "Marine Technical Case"
    story=[Paragraph("MarineWise AI - Command Center Final Report",title),Paragraph(esc(f"Topic: {topic}"),body)]
    for label,key in [("Equipment / Engine Model","engine_model"),("Ship","ship"),("Technical Objective","objective")]:
        value=clean_output_text(result.get(key))
        if value: story += [Paragraph(label,head),Paragraph(esc(value),body)]
    story += [Paragraph("Final Marine AI Response",head)]
    final=clean_output_text(result.get("final"))
    for raw in final.splitlines():
        line=raw.strip()
        if not line: continue
        line=re.sub(r"^#{1,6}\s*", "", line); line=re.sub(r"^[-*•]+\s*", "", line)
        m=re.match(r"^(\d+)[.)]\s+(.*)$",line)
        story.append(Paragraph(esc(f"{m.group(1)}. {m.group(2)}") if m else esc(line),body))
    agents=result.get("agent_names",[]) or []
    if agents:
        story += [PageBreak(),Paragraph("CrewAI Agents Involved",head)]
        for a in agents: story.append(Paragraph(esc(a),body))
    if result.get("used_web_evidence"):
        story += [Paragraph("Evidence Note",head),Paragraph(esc("This final response includes supplementary online evidence that was explicitly requested by the user."),body)]
    else:
        story += [Paragraph("Evidence Note",head),Paragraph(esc("This final response was generated without supplementary online evidence."),body)]
    doc.build(story); return out.getvalue()


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
    st.markdown('<div class="main-title">Troubleshooting Agent</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Manual-first, human-in-the-loop marine engine troubleshooting</div>', unsafe_allow_html=True)

    with st.form("troubleshoot_form"):
        manufacturer=st.selectbox("Manufacturer",["CAT","MTU","YANMAR","YAMAHA","HONDA","MAN","OTHER"])
        engine_model=st.text_input("Engine Model",placeholder="e.g. MTU 16V 4000 M90")
        serial=st.text_input("Serial Number (optional)")
        defect=st.text_area("Defect or Alarm",placeholder="Describe the symptom, alarm code, and what happened.")
        submitted=st.form_submit_button("Troubleshoot",type="primary")

    if submitted:
        if not engine_model.strip() or not defect.strip():
            st.warning("Please enter the engine model and defect/alarm."); return
        rag=st.session_state.get("rag")
        if not rag:
            st.warning("No manuals are indexed. Build the FAISS index first."); return
        st.session_state.troubleshooting_case={"manufacturer":manufacturer,"engine_model":engine_model.strip(),"serial":serial.strip(),"defect":defect.strip()}
        st.session_state.troubleshooting_answer=None
        st.session_state.troubleshooting_web_answer=None
        st.session_state.troubleshooting_web_choice=None
        st.session_state.troubleshooting_web_error=None
        st.session_state.troubleshooting_combined_answer=None
        query=make_context_query(manufacturer,engine_model,defect)
        with st.spinner("Searching the supplied OEM/manual documents..."):
            results=search_index(rag,query,get_embedder(),k=8)
            context,relevant=retrieve_context(results,min_score=0.30,max_chunks=4,max_chars=5000)
            st.session_state.troubleshooting_context=context
            st.session_state.troubleshooting_sources=relevant
            st.session_state.last_retrieved=relevant
        if context:
            with st.spinner("Preparing a manual-grounded troubleshooting procedure..."):
                answer=run_agent(
                    """You are a marine engine troubleshooting specialist. Use ONLY the supplied OEM/manual excerpts as evidence. Produce a technician-ready step-by-step troubleshooting procedure. Organize it as: 1. symptom interpretation, 2. safety/isolation, 3. diagnostic checks in sequence, 4. expected result for each check, 5. decision point if abnormal, 6. corrective action when supported, 7. final verification. Do not invent specifications, alarm limits, component locations, measurements, or procedures. If the excerpts contain only partial information, provide only the supported steps and clearly identify the missing information. Cite manual file/page for supported claims.""",
                    f"Manufacturer: {manufacturer}\nEngine model: {engine_model}\nSerial: {serial or 'not provided'}\nDefect/alarm: {defect}\n\nOEM/MANUAL EXCERPTS:\n{context}",
                    "troubleshooting",
                )
            st.session_state.troubleshooting_answer=answer
        else:
            st.session_state.troubleshooting_answer=None

    case=st.session_state.get("troubleshooting_case")
    if not case: return
    relevant=st.session_state.get("troubleshooting_sources",[])
    context=st.session_state.get("troubleshooting_context","")
    answer=st.session_state.get("troubleshooting_answer")
    web_answer=st.session_state.get("troubleshooting_web_answer")
    choice=st.session_state.get("troubleshooting_web_choice")

    if context or relevant:
        rag=st.session_state.get("rag")
        if rag: st.caption(f"Chunk size: {rag['chunk_size']} characters • Retrieved pages: {len({r['page'] for r in relevant})}")
        render_sources(relevant)
    if answer:
        st.markdown("### OEM / Manual Troubleshooting")
        st.write(answer)

    # Human-in-the-loop: online research is offered whenever the manual did not
    # provide enough information for a reliable answer.
    manual_has_useful_answer=bool(context and answer and "not found in manuals" not in answer.lower())
    if not manual_has_useful_answer and web_answer is None:
        st.warning("The supplied OEM/manual evidence does not provide enough information for a complete troubleshooting procedure.")
        st.write("Would you like MarineWise AI to search reliable online technical sources?")
        if choice is None:
            yes_col,no_col=st.columns(2)
            if yes_col.button("Yes — Search Online",key="web_troubleshoot_yes",type="primary"):
                st.session_state.troubleshooting_web_choice="yes"; st.rerun()
            if no_col.button("No — Stay Manual-Only",key="web_troubleshoot_no"):
                st.session_state.troubleshooting_web_choice="no"; st.rerun()
            return
        if choice=="no":
            st.info("Online search was not requested. The troubleshooting result remains manual-only.")
        elif choice=="yes":
            try:
                with st.spinner("Searching approved online technical sources..."):
                    web_answer=ask_web(
                        """You are a marine engine troubleshooting research specialist. Search reliable OEM/manufacturer, classification, regulatory and reputable technical sources. Produce technician-ready step-by-step troubleshooting guidance: symptom interpretation, safety/isolation, diagnostic checks, expected results, abnormal-result decisions, corrective action only when supported, and final verification. Do not invent specifications, alarm limits, component locations or procedures. Clearly identify web-derived information and tell the technician to verify it against the current OEM manual.""",
                        f"Manufacturer: {case['manufacturer']}\nEngine model: {case['engine_model']}\nSerial: {case['serial'] or 'not provided'}\nSymptom/alarm: {case['defect']}\n\nManual evidence already reviewed:\n{context or '[No sufficiently relevant manual evidence]'}",
                    )
                st.session_state.troubleshooting_web_answer=web_answer
                st.session_state.troubleshooting_web_error=None
            except Exception as exc:
                st.session_state.troubleshooting_web_answer=None
                st.session_state.troubleshooting_web_error=safe_text(exc)
                st.error("Online technical research could not be completed. You can retry the online search.")

    web_answer=st.session_state.get("troubleshooting_web_answer")
    if web_answer:
        st.markdown("### Approved Online Supplement")
        st.info("Online research was performed only after your approval. Verify web-derived information against the current OEM manual.")
        st.write(web_answer)

    # When both evidence streams exist, synthesize them into ONE final procedure.
    combined=st.session_state.get("troubleshooting_combined_answer")
    if web_answer and not combined:
        with st.spinner("Combining OEM/manual evidence with the approved online research..."):
            combined=run_agent(
                """You are the final MarineWise troubleshooting editor. Combine the supplied OEM/manual evidence and approved online evidence into ONE technician-ready step-by-step troubleshooting procedure. The OEM/manual evidence has priority. Use web evidence only to supplement gaps. Do not invent facts, specifications, alarm limits, measurements, component locations or procedures. Clearly mark any web-derived step. If sources conflict, do not silently choose; state the conflict and instruct verification against the current OEM manual. Structure: 1. Problem/symptom interpretation, 2. Safety/isolation, 3. Diagnostic checks in sequence, 4. Expected result, 5. Action if abnormal, 6. Root cause only when supported, 7. Corrective action only when supported, 8. Final verification/return to service. Include source file/page or web source attribution where available.""",
                f"CASE:\nManufacturer: {case['manufacturer']}\nEngine model: {case['engine_model']}\nSerial: {case['serial'] or 'not provided'}\nDefect/alarm: {case['defect']}\n\nOEM/MANUAL EVIDENCE:\n{context or '[None]'}\n\nOEM/MANUAL ANSWER:\n{answer or '[None]'}\n\nAPPROVED ONLINE EVIDENCE:\n{web_answer}",
                "troubleshooting",
            )
            st.session_state.troubleshooting_combined_answer=combined
            st.session_state.troubleshooting_answer=combined

    combined=st.session_state.get("troubleshooting_combined_answer")
    if combined:
        st.markdown("### Final Combined Troubleshooting Procedure")
        st.write(combined)

    final_answer=combined or answer
    if final_answer:
        try:
            troubleshooting_web_sources = []
            if web_answer:
                for url in re.findall(r"https?://[^\\s)\\]}>]+", clean_output_text(web_answer)):
                    troubleshooting_web_sources.append(
                        {
                            "title": "Approved online technical source",
                            "url": url.rstrip(".,;"),
                        }
                    )

            pdf=make_troubleshooting_pdf(
                case,
                final_answer,
                relevant,
                None,
                troubleshooting_web_sources,
            )
            st.download_button(
                "Download Troubleshooting PDF",
                data=pdf,
                file_name="MarineWise_Troubleshooting.pdf",
                mime="application/pdf",
                type="primary",
                key="download_troubleshooting_pdf",
            )
        except Exception as exc:
            st.error(f"Could not create the troubleshooting PDF: {exc}")


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
                web_sources,
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



def clean_quiz_markdown(text: Any) -> str:
    """Remove Markdown emphasis/backticks that can accidentally reveal MCQ answers.

    This is intentionally used only for quiz content so normal technical text
    elsewhere in MarineWise AI is not changed.
    """
    value = clean_output_text(text)

    # Remove common Markdown emphasis markers.
    value = value.replace("**", "")
    value = value.replace("__", "")
    value = value.replace("```", "")
    value = value.replace("`", "")
    value = value.replace("~~", "")

    # Remove single-asterisk/underscore emphasis when they wrap text.
    value = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", value)
    value = re.sub(r"(?<!_)_([^_\n]+)_(?!_)", r"\1", value)

    return value.strip()


def quiz_page() -> None:
    st.subheader("2B. Technical Assessment / Quiz")
    st.caption(
        "Generate technically relevant technician questions using the supplied manuals first, "
        "then reliable online research where available, followed by a verification pass."
    )

    with st.form("quiz_form"):
        topic = st.text_input(
            "Topic",
            placeholder="Fuel Injection System",
        )

        qtype = st.selectbox(
            "Type",
            ["MCQ", "Short Question", "True-False"],
        )

        count = st.number_input(
            "Number of questions",
            min_value=1,
            max_value=30,
            value=10,
            step=1,
        )

        submitted = st.form_submit_button(
            "Generate Technical Assessment / Quiz",
            type="primary",
        )

    if not submitted:
        return

    topic = topic.strip()

    if not topic:
        st.warning("Enter a topic.")
        return

    rag = st.session_state.get("rag")
    context = ""
    relevant: list[dict[str, Any]] = []

    # --------------------------------------------------------
    # 1. MANUAL-FIRST RETRIEVAL
    # --------------------------------------------------------
    if rag:
        with st.spinner("Searching supplied manuals first..."):
            results = search_index(
                rag,
                topic,
                get_embedder(),
                k=8,
            )

            context, relevant = retrieve_context(
                results,
                min_score=0.30,
                max_chunks=5,
                max_chars=6000,
            )

    # --------------------------------------------------------
    # 2. SUPPLEMENTARY ONLINE TECHNICAL RESEARCH
    # --------------------------------------------------------
    research: dict[str, Any] = {
        "results": [],
        "images": [],
    }

    tavily_key = get_secret("TAVILY_API_KEY")

    if tavily_key and run_training_web_research is not None:
        with st.spinner("Researching the quiz topic from reliable technical sources..."):
            try:
                research = run_training_web_research(
                    selected_provider(),
                    "Marine engine / marine technical equipment",
                    topic,
                    "",
                    context,
                    tavily_key,
                ) or research
            except Exception as exc:
                st.warning(
                    "Online research could not be completed. "
                    "The quiz will continue using the supplied manuals."
                )
                st.caption(f"Online research note: {safe_text(exc)}")

    web_sources = research.get("results", []) or []

    # Keep only useful source information in the prompt.
    web_evidence = []
    for item in web_sources[:10]:
        if isinstance(item, dict):
            title = safe_text(item.get("title"))
            url = safe_text(item.get("url"))
            content = safe_text(item.get("content"))
            if title or content:
                web_evidence.append(
                    f"TITLE: {title}\nURL: {url}\nCONTENT: {content[:1800]}"
                )

    web_context = "\n\n---\n\n".join(web_evidence)

    # --------------------------------------------------------
    # 3. GENERATE THE FIRST QUIZ DRAFT
    # --------------------------------------------------------
    with st.spinner("Generating technically relevant questions and answer key..."):
        quiz_draft = run_agent(
            """
You are a senior marine technical training instructor.

Create a technically relevant assessment/quiz about the EXACT requested topic.

STRICT REQUIREMENTS:
- Follow the requested question type and question count exactly.
- Every question must test knowledge directly related to the requested topic.
- Do NOT ask questions about PDF page numbers, document titles, source locations,
  or incidental information merely because it appeared in a retrieved document.
- Use the supplied OEM/manual excerpts as the PRIMARY technical source.
- Use reliable online technical research only as SECONDARY supporting evidence.
- Never invent manufacturer-specific specifications, limits, clearances, pressures,
  temperatures, procedures, or component details.
- If a manufacturer-specific value is not supported, ask a conceptual question instead.
- For MCQ, provide exactly four options A-D and one unambiguous correct answer.
- For MCQ, NEVER use Markdown emphasis or special formatting in any question or option.
- NEVER use **bold**, ***bold/italic***, *italic*, __underline__, backticks, or any other marker to highlight an option.
- All four MCQ options must have identical plain-text formatting so the correct answer cannot be visually identified.
- For Short Question, provide a concise model answer.
- For True-False, provide an unambiguous statement and answer.
- Finish with an ANSWER KEY.
- The answer key must be easy to parse in this format:
  1=A, 2=C, 3=D
- Do not include unrelated questions.

The assessment will be independently checked before it is delivered.
""",
            (
                f"REQUESTED TOPIC:\n{topic}\n\n"
                f"QUESTION TYPE:\n{qtype}\n\n"
                f"QUESTION COUNT:\n{int(count)}\n\n"
                f"PRIMARY OEM / MANUAL EXCERPTS:\n"
                f"{context or '[No sufficiently relevant manual excerpts were retrieved.]'}\n\n"
                f"SECONDARY ONLINE TECHNICAL RESEARCH:\n"
                f"{web_context or '[No online research available.]'}"
            ),
            "training",
        )

    # --------------------------------------------------------
    # 4. VERIFICATION / QUALITY-CONTROL PASS
    # --------------------------------------------------------
    with st.spinner("Verifying question relevance, count, options, and answer key..."):
        quiz_text = run_agent(
            """
You are the final quality-control reviewer for a marine technician assessment.

Review the DRAFT assessment against the requested topic, type, and count.

Return a corrected final assessment, not a review report.

CHECK EVERY QUESTION:
1. Is it directly related to the requested topic?
2. Is it technically meaningful for a marine technician?
3. Does it avoid irrelevant document/page trivia?
4. Does it avoid unsupported manufacturer-specific facts?
5. Does the number of questions exactly match the requested count?
6. For MCQ, are there exactly four options A-D and only one clearly correct answer?
7. For MCQ, do all options use identical plain-text formatting with NO **, ***, *, _, backticks, or other answer-revealing markers? Remove any such markers before returning the final assessment.
8. For Short Question, is the model answer technically defensible?
8. For True-False, is the statement unambiguous?
9. Does the answer key match the final questions exactly?

SOURCE PRIORITY:
- OEM/manual evidence first.
- Reliable web evidence second.
- Do not invent unsupported details.

FINAL FORMAT:
- Keep the requested question type.
- Number every question sequentially.
- Put each question on its own clearly separated line/paragraph.
- End with:
  ANSWER KEY
  1=A, 2=C, 3=D

Do not include a QC explanation.
""",
            (
                f"REQUESTED TOPIC:\n{topic}\n\n"
                f"QUESTION TYPE:\n{qtype}\n\n"
                f"REQUIRED QUESTION COUNT:\n{int(count)}\n\n"
                f"MANUAL EVIDENCE:\n{context or '[None]'}\n\n"
                f"ONLINE EVIDENCE:\n{web_context or '[None]'}\n\n"
                f"DRAFT ASSESSMENT:\n{quiz_draft}"
            ),
            "training",
        )

    # FINAL SAFETY CLEANUP: prevent Markdown markers from revealing the correct MCQ.
    quiz_text = clean_quiz_markdown(quiz_text)

    st.markdown("### Technical Assessment / Quiz")
    st.write(quiz_text)

    st.download_button(
        "⬇ Download Technical Assessment / Quiz PDF",
        make_quiz_pdf(
            topic,
            qtype,
            quiz_text,
            relevant,
            web_sources,
        ),
        "marinewise_technical_assessment_quiz.pdf",
        "application/pdf",
        type="primary",
    )

    render_sources(relevant)

    if web_sources:
        st.markdown("**Web research sources used:**")
        for source in web_sources[:8]:
            if not isinstance(source, dict):
                continue
            title = safe_text(source.get("title")) or "Technical web source"
            url = safe_text(source.get("url"))
            if url:
                st.markdown(f"- [{title}]({url})")
            else:
                st.markdown(f"- {title}")



# ============================================================
# ASSESSMENT
# ============================================================


def configure_tesseract() -> str | None:
    """
    Locate the Tesseract executable on local Windows/Linux systems.

    Streamlit Cloud should provide Tesseract through packages.txt.
    """
    import shutil

    try:
        import pytesseract
    except ImportError:
        return None

    detected = shutil.which("tesseract")

    candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        "/usr/bin/tesseract",
        "/usr/local/bin/tesseract",
    ]

    executable = detected or next(
        (
            path
            for path in candidates
            if os.path.exists(path)
        ),
        None,
    )

    if executable:
        pytesseract.pytesseract.tesseract_cmd = executable

    return executable



def _assessment_ocr_lines(image: Image.Image) -> list[dict[str, Any]]:
    """Return Tesseract OCR lines with positions for answer-mark detection."""
    import pytesseract

    work = image.convert("RGB")
    scale = min(1.0, 1000 / max(work.size))

    if scale < 1.0:
        work = work.resize(
            (
                int(work.width * scale),
                int(work.height * scale),
            ),
            Image.Resampling.LANCZOS,
        )

    data = pytesseract.image_to_data(
        work,
        config="--psm 6",
        output_type=pytesseract.Output.DICT,
    )

    grouped: dict[tuple[int, int, int], list[dict[str, Any]]] = {}

    for index, raw_text in enumerate(data["text"]):
        text_value = raw_text.strip()

        if not text_value:
            continue

        key = (
            int(data["block_num"][index]),
            int(data["par_num"][index]),
            int(data["line_num"][index]),
        )

        grouped.setdefault(key, []).append(
            {
                "x": int(data["left"][index]),
                "y": int(data["top"][index]),
                "w": int(data["width"][index]),
                "h": int(data["height"][index]),
                "text": text_value,
            }
        )

    lines: list[dict[str, Any]] = []

    for words in grouped.values():
        words.sort(key=lambda item: item["x"])

        top = min(
            item["y"]
            for item in words
        )

        bottom = max(
            item["y"] + item["h"]
            for item in words
        )

        left = min(
            item["x"]
            for item in words
        )

        lines.append(
            {
                "y": top,
                "bottom": bottom,
                "x": left,
                "text": " ".join(
                    item["text"]
                    for item in words
                ),
            }
        )

    lines.sort(
        key=lambda item: item["y"]
    )

    return lines


def _assessment_colored_mark_groups(
    image: Image.Image,
) -> list[tuple[int, int]]:
    """
    Detect colored pen/highlighter marks near the answer-option column.

    The supplied assessments use purple/blue filled circles. This detector
    deliberately looks for saturated colored ink rather than ordinary
    black printed text.
    """
    work = image.convert("RGB")

    scale = min(
        1.0,
        1000 / max(work.size),
    )

    if scale < 1.0:
        work = work.resize(
            (
                int(work.width * scale),
                int(work.height * scale),
            ),
            Image.Resampling.LANCZOS,
        )

    pixels = np.asarray(
        work,
        dtype=np.uint8,
    )

    maximum = pixels.max(
        axis=2
    )
    minimum = pixels.min(
        axis=2
    )

    saturation = (
        maximum.astype(np.int16)
        - minimum.astype(np.int16)
    )

    red = pixels[:, :, 0].astype(
        np.int16
    )
    green = pixels[:, :, 1].astype(
        np.int16
    )
    blue = pixels[:, :, 2].astype(
        np.int16
    )

    # Purple/blue/colored writing:
    # - noticeably saturated
    # - darker than the white paper
    # - blue channel stronger than red
    mask = (
        (saturation > 45)
        & (maximum < 235)
        & ((blue - red) > 20)
    )

    x_start = int(
        work.width * 0.08
    )
    x_end = int(
        work.width * 0.28
    )

    row_counts = mask[
        :,
        x_start:x_end,
    ].sum(
        axis=1
    )

    active = row_counts >= 3

    groups: list[tuple[int, int]] = []

    index = 0

    while index < len(active):
        if not active[index]:
            index += 1
            continue

        end_index = index

        while (
            end_index + 1 < len(active)
            and (
                active[end_index + 1]
                or row_counts[end_index + 1] >= 1
            )
        ):
            end_index += 1

        total_pixels = int(
            row_counts[
                index:end_index + 1
            ].sum()
        )

        height = (
            end_index - index + 1
        )

        if (
            5 <= height <= 60
            and total_pixels >= 25
        ):
            groups.append(
                (
                    index,
                    end_index,
                )
            )

        index = end_index + 1

    return groups


def _infer_marked_option(
    mark_top: int,
    mark_bottom: int,
    lines: list[dict[str, Any]],
) -> str | None:
    """
    Infer A/B/C/D for one detected colored mark.

    The selected option's OCR line often contains '@' or another symbol
    instead of the printed option letter. We therefore treat the line
    containing the mark as an unlabeled option and infer its letter from
    the neighboring printed A/B/C/D lines.
    """
    window_top = mark_top - 50
    window_bottom = mark_bottom + 70

    nearby = [
        line
        for line in lines
        if line["bottom"] >= window_top
        and line["y"] <= window_bottom
    ]

    selected_line = None
    selected_distance = None

    for line in nearby:
        overlaps = (
            line["y"] <= mark_bottom
            and line["bottom"] >= mark_top
        )

        if overlaps:
            distance = abs(
                line["y"] - mark_top
            )

            if (
                selected_distance is None
                or distance < selected_distance
            ):
                selected_line = line
                selected_distance = distance

    if selected_line is not None:
        selected_match = re.search(
            r"(?<![A-Za-z])([ABCD])[\.:,\)]",
            selected_line["text"],
            re.I,
        )

        if selected_match:
            return selected_match.group(1).upper()

    labels: list[tuple[int, int, str]] = []

    for line in nearby:
        if (
            selected_line is not None
            and line is selected_line
        ):
            continue

        for match in re.finditer(
            r"(?<![A-Za-z])([ABCD])[\.:,\)]",
            line["text"],
            re.I,
        ):
            labels.append(
                (
                    line["y"],
                    line["bottom"],
                    match.group(1).upper(),
                )
            )

    # Keep the closest occurrence for each printed option label.
    label_positions: dict[str, int] = {}

    for top, bottom, label in labels:
        if label not in label_positions:
            label_positions[label] = top
        elif abs(top - mark_top) < abs(
            label_positions[label] - mark_top
        ):
            label_positions[label] = top

    option_index = {
        "A": 0,
        "B": 1,
        "C": 2,
        "D": 3,
    }

    known = sorted(
        (
            option_index[label],
            position,
            label,
        )
        for label, position
        in label_positions.items()
    )

    if not known:
        return None

    # If the selected line sits between two known labels and exactly one
    # option is missing, the missing middle option is the selected answer.
    for left, right in zip(
        known,
        known[1:],
    ):
        left_index, left_y, _ = left
        right_index, right_y, _ = right

        if (
            left_y <= mark_top <= right_y
            and right_index - left_index == 2
        ):
            return chr(
                ord("A")
                + left_index
                + 1
            )

    # Use the known labels as a vertical scale. This handles cases where
    # OCR misses A or B entirely.
    if len(known) >= 2:
        x_values = np.asarray(
            [
                item[0]
                for item in known
            ],
            dtype=float,
        )

        y_values = np.asarray(
            [
                item[1]
                for item in known
            ],
            dtype=float,
        )

        slope, intercept = np.polyfit(
            x_values,
            y_values,
            1,
        )

        predictions = {
            chr(ord("A") + index): (
                slope * index
                + intercept
            )
            for index in range(4)
        }

        return min(
            predictions,
            key=lambda label: abs(
                predictions[label]
                - mark_top
            ),
        )

    # Last-resort one-label estimate. Estimate the printed option
    # spacing from nearby OCR lines instead of assuming a fixed value.
    index, known_y, _ = known[0]

    nearby_tops = sorted(
        {
            line["y"]
            for line in nearby
            if line["y"] <= window_bottom
            and line["bottom"] >= window_top
        }
    )

    gaps = [
        right - left
        for left, right in zip(
            nearby_tops,
            nearby_tops[1:],
        )
        if 10 <= right - left <= 35
    ]

    spacing = (
        float(np.median(gaps))
        if gaps
        else 20.0
    )

    predictions = {
        chr(ord("A") + option): (
            known_y
            + (option - index) * spacing
        )
        for option in range(4)
    }

    return min(
        predictions,
        key=lambda label: abs(
            predictions[label]
            - mark_top
        ),
    )


def _assign_marked_question_numbers(
    mark_groups: list[tuple[int, int]],
    lines: list[dict[str, Any]],
) -> list[tuple[int, int, int]]:
    """
    Associate colored marks with question numbers.

    OCR occasionally misses a question number, so the detected marks are
    also used to fill missing sequential question numbers.
    """
    question_starts: list[tuple[int, int]] = []

    for line in lines:
        match = re.search(
            r"(?<!\d)(\d{1,2})\.",
            line["text"],
        )

        if match:
            question_starts.append(
                (
                    int(match.group(1)),
                    line["y"],
                )
            )

    question_starts.sort(
        key=lambda item: item[1]
    )

    assigned: list[tuple[int, int, int]] = []

    marks = sorted(
        mark_groups,
        key=lambda item: item[0],
    )

    for mark_index, (
        mark_top,
        mark_bottom,
    ) in enumerate(marks):
        prior_questions = [
            item
            for item in question_starts
            if item[1] <= mark_top
        ]

        if prior_questions:
            base_question, base_y = (
                prior_questions[-1]
            )

            next_question_y = next(
                (
                    y
                    for q, y
                    in question_starts
                    if y > base_y
                ),
                None,
            )

            earlier_marks = [
                item
                for item in marks[:mark_index]
                if item[0] >= base_y
                and (
                    next_question_y is None
                    or item[0] < next_question_y
                )
            ]

            question_number = (
                base_question
                + len(earlier_marks)
            )

        elif question_starts:
            first_question = min(
                question_starts,
                key=lambda item: item[1],
            )[0]

            earlier_before_first = [
                item
                for item in marks[:mark_index]
                if item[0] < question_starts[0][1]
            ]

            question_number = (
                first_question
                - len(earlier_before_first)
                + len(earlier_before_first)
                - 1
                + 1
            )

            # Equivalent to first_question minus the number of marks
            # still to be assigned before the first explicit question.
            total_before_first = sum(
                1
                for item in marks
                if item[0] < question_starts[0][1]
            )

            position_before_first = len(
                earlier_before_first
            )

            question_number = (
                first_question
                - total_before_first
                + position_before_first
            )

        else:
            question_number = (
                mark_index + 1
            )

        assigned.append(
            (
                question_number,
                mark_top,
                mark_bottom,
            )
        )

    return assigned


def detect_marked_assessment_answers(
    uploaded_files: list[Any],
) -> dict[int, str]:
    """
    Read colored technician answer marks from all assessment pages.

    The detector works from the original page images rather than relying
    on Tesseract to understand a filled circle.
    """
    detected: dict[int, str] = {}

    for uploaded_file in uploaded_files:
        image = Image.open(
            io.BytesIO(
                uploaded_file.getvalue()
            )
        )

        image = ImageOps.exif_transpose(
            image
        ).convert("RGB")

        lines = _assessment_ocr_lines(
            image
        )

        mark_groups = (
            _assessment_colored_mark_groups(
                image
            )
        )

        question_marks = (
            _assign_marked_question_numbers(
                mark_groups,
                lines,
            )
        )

        for (
            question_number,
            mark_top,
            mark_bottom,
        ) in question_marks:
            option = _infer_marked_option(
                mark_top,
                mark_bottom,
                lines,
            )

            if option in {
                "A",
                "B",
                "C",
                "D",
            }:
                detected[
                    question_number
                ] = option

    return dict(
        sorted(
            detected.items()
        )
    )


def format_detected_answers(
    answers: dict[int, str],
) -> str:
    """Format detected answers for display and model context."""
    if not answers:
        return "No marked answers were detected."

    return ", ".join(
        f"{question}={answer}"
        for question, answer
        in sorted(
            answers.items()
        )
    )


def parse_answer_key(
    answer_key: str,
) -> dict[int, str]:
    """Parse entries such as 1=B, 2=C, 3=True into MCQ answers."""
    parsed: dict[int, str] = {}

    for match in re.finditer(
        r"(?<!\d)(\d{1,3})\s*"
        r"(?:=|:|-)\s*"
        r"([ABCD])\b",
        answer_key,
        re.I,
    ):
        parsed[
            int(match.group(1))
        ] = match.group(2).upper()

    return dict(
        sorted(
            parsed.items()
        )
    )


def extract_assessment_images(
    uploaded_files: list[Any],
) -> str:
    """
    OCR all uploaded assessment pages in upload order.

    The assessment can contain 1–7 JPG/PNG pages. Each page is
    normalized before OCR and clearly labelled in the combined text.
    """
    import pytesseract

    extracted_pages: list[str] = []

    for page_number, uploaded_file in enumerate(
        uploaded_files,
        start=1,
    ):
        image = Image.open(
            io.BytesIO(uploaded_file.getvalue())
        )

        image = ImageOps.exif_transpose(
            image
        ).convert("RGB")

        image.thumbnail(
            (2600, 2600),
            Image.Resampling.LANCZOS,
        )

        gray = ImageOps.grayscale(
            image
        )

        gray = ImageOps.autocontrast(
            gray
        )

        text = pytesseract.image_to_string(
            gray,
            config="--psm 6",
        ).strip()

        extracted_pages.append(
            f"===== ASSESSMENT PAGE {page_number}: "
            f"{uploaded_file.name} =====\n"
            f"{text or '[No OCR text detected on this page.]'}"
        )

    return "\n\n".join(
        extracted_pages
    )



def assessment_page() -> None:
    st.subheader(
        "2C. Score an Assessment"
    )

    st.caption(
        "Upload 1–7 assessment pages/images together. "
        "The pages are processed and scored as one assessment."
    )

    engine = st.text_input(
        "Engine / Manufacturer (optional)",
        placeholder=(
            "Example: MTU 16V 4000 M90 / "
            "Caterpillar C32"
        ),
        key="assessment_engine",
    )

    uploaded = st.file_uploader(
        "Assessment pages",
        type=[
            "jpg",
            "jpeg",
            "png",
        ],
        accept_multiple_files=True,
        key="assessment_images",
        help=(
            "Select between 1 and 7 assessment "
            "pages/images."
        ),
    )

    answer_key = st.text_area(
        "Answer key",
        placeholder=(
            "Example: 1=B, 2=B, 3=B, 4=B, "
            "5=A, 6=B, 7=D, 8=B, 9=C, 10=D"
        ),
        key="assessment_answer_key",
    )

    if uploaded and len(uploaded) > 7:
        st.error(
            "Please upload no more than 7 assessment pages."
        )
        return

    if not uploaded:
        return

    st.markdown(
        "### Assessment Pages"
    )

    preview_columns = st.columns(
        min(
            len(uploaded),
            4,
        )
    )

    for index, uploaded_file in enumerate(
        uploaded,
        start=1,
    ):
        with preview_columns[
            (index - 1)
            % len(preview_columns)
        ]:
            image = Image.open(
                io.BytesIO(
                    uploaded_file.getvalue()
                )
            )

            image = ImageOps.exif_transpose(
                image
            )

            st.image(
                image,
                caption=(
                    f"Page {index}: "
                    f"{uploaded_file.name}"
                ),
                use_container_width=True,
            )

    executable = configure_tesseract()

    if not executable:
        st.error(
            "Tesseract OCR could not be found. "
            "Install Tesseract OCR on your computer, or add "
            "tesseract-ocr to packages.txt for Streamlit Cloud."
        )
        return

    # Detect the actual colored answer marks before the scoring button.
    # This prevents Tesseract from interpreting a filled circle as text.
    with st.spinner(
        "Detecting technician answer marks..."
    ):
        try:
            detected_answers = (
                detect_marked_assessment_answers(
                    uploaded
                )
            )
        except Exception as exc:
            detected_answers = {}
            st.warning(
                "Automatic answer-mark detection could not "
                f"complete: {exc}"
            )

    st.text_input(
        "Detected technician answers",
        value=format_detected_answers(
            detected_answers
        ),
        disabled=True,
        help=(
            "The app detects the colored marks next to "
            "the A/B/C/D choices. Review the result before "
            "scoring."
        ),
    )

    if not detected_answers:
        st.warning(
            "No colored answer marks were detected. "
            "Use clearly visible filled/circled answer marks."
        )

    if st.button(
        "Score Assessment",
        type="primary",
        key="score_assessment_button",
    ):
        if not answer_key.strip():
            st.warning(
                "Add an answer key so the app can calculate "
                "the assessment score."
            )
            return

        with st.spinner(
            f"Reading all {len(uploaded)} assessment page(s)..."
        ):
            try:
                extracted = extract_assessment_images(
                    uploaded
                )
            except Exception as exc:
                st.error(
                    f"OCR could not run: {exc}"
                )
                return

        st.text_area(
            "OCR text — check this before scoring",
            extracted,
            height=350,
            key="assessment_ocr_text",
        )

        with st.spinner(
            "Scoring the complete assessment..."
        ):
            score = score_with_agent(
                extracted,
                answer_key,
                detected_answers,
            )

        st.markdown(
            "### Assessment Result"
        )

        score_col, topic_col = st.columns(
            [1, 2]
        )

        with score_col:
            st.metric(
                "Technician Score",
                f"{score['score']:.0f}%",
            )

        with topic_col:
            if score.get(
                "missed_topics"
            ):
                st.markdown(
                    "**Improvement areas identified:** "
                    + ", ".join(
                        score["missed_topics"]
                    )
                )

        st.write(
            score["feedback"]
        )

        st.markdown(
            "**Answer comparison:**"
        )

        st.code(
            score.get(
                "answer_comparison",
                "No comparison available.",
            ),
            language="text",
        )

        if score["score"] >= 50:
            st.success(
                "Assessment is at or above 50%. "
                "No remedial package was automatically generated."
            )
            return

        st.warning(
            "Below 50% — a targeted technician retraining "
            "package is recommended."
        )

        missed_topics = score.get(
            "missed_topics",
            [],
        )

        if not missed_topics:
            missed_topics = [
                "Topics associated with incorrect "
                "or unanswered assessment questions"
            ]

        remedial_topic = ", ".join(
            missed_topics[:8]
        )

        # ----------------------------------------------------
        # TARGETED MANUAL RETRIEVAL
        # ----------------------------------------------------

        rag = st.session_state.get(
            "rag"
        )

        context = ""
        relevant: list[dict[str, Any]] = []

        if rag:
            with st.spinner(
                "Finding manual sections specifically related "
                "to the technician's missed topics..."
            ):
                rag_query = (
                    f"{engine.strip()}\n"
                    f"{remedial_topic}\n"
                    "technician assessment mistakes "
                    "troubleshooting training"
                )

                results = search_index(
                    rag,
                    rag_query,
                    get_embedder(),
                    k=10,
                )

                context, relevant = retrieve_context(
                    results,
                    min_score=0.25,
                    max_chunks=8,
                    max_chars=8000,
                )

                st.session_state.last_retrieved = (
                    relevant
                )

        # ----------------------------------------------------
        # TARGETED WEB RESEARCH
        # ----------------------------------------------------

        research: dict[str, Any] = {
            "results": [],
            "images": [],
        }

        tavily_key = get_secret(
            "TAVILY_API_KEY"
        )

        if (
            tavily_key
            and run_training_web_research is not None
        ):
            with st.spinner(
                "Researching the missed technical areas "
                "using reliable technical sources..."
            ):
                research = run_training_web_research(
                    selected_provider(),
                    engine.strip()
                    or "Marine engine",
                    remedial_topic,
                    "",
                    context,
                    tavily_key,
                )

        web_sources = research.get(
            "results",
            [],
        ) or []

        # ----------------------------------------------------
        # TARGETED RETRAINING CONTENT
        # ----------------------------------------------------

        with st.spinner(
            "Building a targeted retraining package "
            "from the technician's actual mistakes..."
        ):
            remedial_content = run_agent(
                """
You are a senior marine technical training instructor preparing a
REMEDIAL TRAINING PACKAGE for a technician who scored below 50%.

The technician's MISSED TOPICS are the primary constraint.

Do NOT create generic marine training. Teach only the technical areas
that the assessment shows the technician needs to improve.

SOURCE PRIORITY:
1. Supplied manufacturer/manual excerpts are the primary source.
2. Web research is secondary supporting evidence only.
3. If a detail is not supported by the supplied manual or reliable web
   evidence, do not invent a manufacturer-specific value or procedure.

For every missed area:
- explain the concept clearly
- explain why it matters to a marine technician
- identify the relevant components/functions
- explain the correct diagnostic or inspection logic where supported
- identify common technician mistakes
- provide corrective learning points
- provide a short practice check

Finish with:
- a practical knowledge check
- a final retest section directly related to the missed topics

Do not include unrelated systems or topics merely to make the package longer.
Clearly distinguish manual-supported information from secondary web-supported
information when they differ or when the manual does not cover a point.
""",
                (
                    f"ENGINE / MANUFACTURER:\n"
                    f"{engine.strip() or 'Not specified'}\n\n"
                    f"TECHNICIAN SCORE:\n"
                    f"{score['score']:.0f}%\n\n"
                    f"MISSED TOPICS:\n"
                    f"{remedial_topic}\n\n"
                    f"SCORING FEEDBACK:\n"
                    f"{score['feedback']}\n\n"
                    f"ANSWER COMPARISON:\n"
                    f"{score.get('answer_comparison', '')}\n\n"
                    f"OCR FROM ALL ASSESSMENT PAGES:\n"
                    f"{extracted}\n\n"
                    f"ANSWER KEY:\n"
                    f"{answer_key}\n\n"
                    f"PRIMARY MANUAL EXCERPTS:\n"
                    f"{context}\n\n"
                    f"SECONDARY WEB RESEARCH:\n"
                    f"{web_sources[:8]}"
                ),
                "training",
            )

        st.markdown(
            "### Targeted Retraining Package"
        )

        st.write(
            remedial_content
        )

        diagram = make_training_diagram(
            engine.strip()
            or "Marine engine",
            remedial_topic,
        )

        st.markdown(
            "### Targeted Training Diagram"
        )

        st.image(
            diagram,
            caption=(
                "Training diagram for the identified "
                "improvement areas. Verify engine-specific "
                "architecture against the current manual."
            ),
            use_container_width=True,
        )

        remedial_plan: list[dict[str, Any]] = []

        if generate_training_presentation_plan is not None:
            with st.spinner(
                "Creating the targeted professional "
                "PowerPoint plan..."
            ):
                plan_raw = (
                    generate_training_presentation_plan(
                        selected_provider(),
                        engine.strip()
                        or "Marine engine",
                        "",
                        remedial_topic,
                        context,
                        research,
                    )
                )

                remedial_plan = parse_training_plan(
                    plan_raw
                )

        if not remedial_plan:
            remedial_plan = [
                {
                    "title": "Assessment Findings",
                    "purpose": (
                        "Identify improvement areas"
                    ),
                    "bullets": [
                        (
                            f"Technician score: "
                            f"{score['score']:.0f}%"
                        ),
                        (
                            f"Missed areas: "
                            f"{remedial_topic}"
                        ),
                    ],
                    "visual_type": "none",
                },
                {
                    "title": (
                        "Targeted Technical Retraining"
                    ),
                    "purpose": (
                        "Correct knowledge gaps"
                    ),
                    "bullets": split_text(
                        remedial_content,
                        180,
                    )[:5],
                    "visual_type": "none",
                },
                {
                    "title": (
                        "Practice & Final Retest"
                    ),
                    "purpose": (
                        "Confirm improvement"
                    ),
                    "bullets": [
                        (
                            "Review each missed topic."
                        ),
                        (
                            "Explain the correct "
                            "diagnostic reasoning."
                        ),
                        (
                            "Complete the final retest."
                        ),
                    ],
                    "visual_type": "process",
                },
            ]

        ppt_bytes = (
            make_professional_training_ppt(
                engine.strip()
                or "Marine engine",
                "",
                (
                    "Remedial Training — "
                    f"{remedial_topic}"
                ),
                remedial_plan,
                get_manual_page_images(
                    rag,
                    relevant,
                ),
                [],
                relevant,
                web_sources,
            )
        )

        st.markdown(
            "### Download Targeted Retraining Package"
        )

        download_columns = st.columns(3)

        download_columns[0].download_button(
            "⬇ Download Remedial PDF",
            make_training_pdf(
                engine.strip()
                or "Marine engine",
                "",
                (
                    "Remedial Training — "
                    f"{remedial_topic}"
                ),
                remedial_content,
                relevant,
                diagram,
                web_sources,
            ),
            "marinewise_targeted_remedial_training.pdf",
            "application/pdf",
            use_container_width=True,
        )

        download_columns[1].download_button(
            "⬇ Download Remedial Word",
            make_training_docx(
                engine.strip()
                or "Marine engine",
                "",
                (
                    "Remedial Training — "
                    f"{remedial_topic}"
                ),
                remedial_content,
                diagram,
            ),
            "marinewise_targeted_remedial_training.docx",
            (
                "application/vnd.openxmlformats-officedocument."
                "wordprocessingml.document"
            ),
            use_container_width=True,
        )

        download_columns[2].download_button(
            "⬇ Download Remedial PowerPoint",
            ppt_bytes,
            "marinewise_targeted_remedial_training.pptx",
            (
                "application/vnd.openxmlformats-officedocument."
                "presentationml.presentation"
            ),
            use_container_width=True,
        )

        render_sources(
            relevant
        )

        if web_sources:
            st.markdown(
                "**Secondary web research used:**"
            )

            for source in web_sources[:8]:
                title = safe_text(
                    source.get("title")
                )

                url = safe_text(
                    source.get("url")
                )

                if title:
                    st.markdown(
                        f"- {title}"
                        + (
                            f" — {url}"
                            if url
                            else ""
                        )
                    )


def score_with_agent(
    extracted: str,
    answer_key: str,
    detected_answers: dict[int, str] | None = None,
) -> dict[str, Any]:
    """
    Score from the detected technician marks and supplied answer key.

    The numeric score is deliberately calculated in Python rather than
    asking the language model to invent/parse the final percentage.
    This prevents an OCR/model formatting failure from becoming 0%.
    """
    expected = parse_answer_key(
        answer_key
    )

    if detected_answers:
        comparisons: list[str] = []
        correct_count = 0

        for question_number, correct_answer in (
            expected.items()
        ):
            technician_answer = (
                detected_answers.get(
                    question_number
                )
            )

            if technician_answer is None:
                result = "UNANSWERED"
            elif technician_answer == correct_answer:
                result = "CORRECT"
                correct_count += 1
            else:
                result = "WRONG"

            comparisons.append(
                (
                    f"Q{question_number}: "
                    f"Technician="
                    f"{technician_answer or 'UNANSWERED'} | "
                    f"Correct={correct_answer} | "
                    f"{result}"
                )
            )

        score = (
            correct_count
            / len(expected)
            * 100
            if expected
            else 0.0
        )

        answer_comparison = (
            "\n".join(comparisons)
        )

        wrong_questions = [
            line
            for line in comparisons
            if (
                "WRONG" in line
                or "UNANSWERED" in line
            )
        ]

        if wrong_questions:
            analysis_prompt = """
You are reviewing a marine technician assessment.

The numeric score has already been calculated deterministically.
Do NOT change the score.

Identify the specific technical topics represented by the incorrect
or unanswered questions and explain the technician's knowledge gaps.

Return exactly:

FEEDBACK: concise explanation
MISSED_TOPICS: comma-separated specific technical topics

Use only topics supported by the supplied assessment text.
Do not invent unrelated topics.
"""

            try:
                raw = run_agent(
                    analysis_prompt,
                    (
                        f"ANSWER COMPARISON:\n"
                        f"{answer_comparison}\n\n"
                        f"INCORRECT / UNANSWERED:\n"
                        f"{wrong_questions}\n\n"
                        f"ASSESSMENT OCR:\n"
                        f"{extracted}"
                    ),
                    "training",
                )
            except Exception:
                raw = ""

            feedback = ""
            missed_topics: list[str] = []

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

            if "MISSED_TOPICS:" in raw:
                topic_text = raw.split(
                    "MISSED_TOPICS:",
                    1,
                )[1].strip()

                missed_topics = [
                    item.strip(
                        " -•"
                    )
                    for item in re.split(
                        r",|\n",
                        topic_text,
                    )
                    if item.strip()
                ]

            if not feedback:
                feedback = (
                    f"The technician answered "
                    f"{correct_count} of "
                    f"{len(expected)} questions correctly."
                )

        else:
            feedback = (
                f"The technician answered all "
                f"{len(expected)} questions correctly."
            )
            missed_topics = []

        return {
            "score": max(
                0.0,
                min(
                    100.0,
                    score,
                ),
            ),
            "feedback": feedback,
            "missed_topics": missed_topics[:8],
            "answer_comparison": answer_comparison,
        }

    # Fallback for assessments where no answer marks could be detected.
    # Keep the old AI path available, but explicitly report the limitation.
    raw = run_agent(
        """
Score a marine technician assessment from the supplied OCR.

Return exactly:

SCORE_PERCENT: number
FEEDBACK: concise explanation
MISSED_TOPICS: comma-separated topics

Do not guess unreadable or unmarked answers.
If technician answer marks are not visible in the OCR, say so in FEEDBACK.
""",
        (
            f"OCR ANSWERS FROM ALL ASSESSMENT PAGES:\n"
            f"{extracted}\n\n"
            f"ANSWER KEY:\n"
            f"{answer_key}"
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

    feedback = raw

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

    missed_topics: list[str] = []

    if "MISSED_TOPICS:" in raw:
        topic_text = raw.split(
            "MISSED_TOPICS:",
            1,
        )[1].strip()

        missed_topics = [
            item.strip(
                " -•"
            )
            for item in re.split(
                r",|\n",
                topic_text,
            )
            if item.strip()
        ]

    return {
        "score": max(
            0.0,
            min(
                100.0,
                score,
            ),
        ),
        "feedback": feedback,
        "missed_topics": missed_topics[:8],
        "answer_comparison": (
            "Automatic answer-mark detection "
            "did not produce a complete answer set."
        ),
    }


# ============================================================
# PAGE 3 — MARINE AI COMMAND CENTER
# ============================================================


def marine_ai_command_center_page() -> None:
    st.markdown(
        '<div class="main-title">'
        "Marine AI Command Center"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="main-subtitle">'
        "CrewAI-powered collaboration between marine technical agents"
        "</div>",
        unsafe_allow_html=True,
    )

    st.info(
        "The Command Center coordinates specialist agents. "
        "Manual evidence is reviewed first. Online research is "
        "performed only after you explicitly request it."
    )

    with st.form("marine_ai_command_center_form"):
        topic = st.text_input(
            "Topic",
            placeholder="e.g. Bow thruster hydraulic system",
        )

        c1, c2 = st.columns(2)

        with c1:
            engine_model = st.text_input(
                "Engine / Equipment Model (optional)",
                placeholder="e.g. MTU 16V 4000 M90",
            )

        with c2:
            ship = st.text_input(
                "Ship (optional)",
                placeholder="e.g. Vessel name / class",
            )

        objective = st.text_area(
            "Technical Objective",
            placeholder=(
                "What should the Marine AI team investigate, "
                "explain or prepare?"
            ),
            height=100,
        )

        submitted = st.form_submit_button(
            "Start Marine AI Command Center",
            type="primary",
        )

    if not submitted:
        previous = st.session_state.get(
            "command_center_result"
        )
        if previous:
            _render_command_center_result(previous)
        return

    topic = topic.strip()
    engine_model = engine_model.strip()
    ship = ship.strip()
    objective = objective.strip()

    if not topic:
        st.warning("Please enter a topic.")
        return

    rag = st.session_state.get("rag")

    manual_context = ""
    manual_results: list[dict[str, Any]] = []

    if rag:
        query_parts = [topic]
        if engine_model:
            query_parts.append(engine_model)
        if ship:
            query_parts.append(ship)
        if objective:
            query_parts.append(objective)

        query = " ".join(query_parts)

        with st.spinner(
            "Command Center Step 1/5 — searching uploaded manuals..."
        ):
            results = search_index(
                rag,
                query,
                get_embedder(),
                k=8,
            )

            manual_context, manual_results = retrieve_context(
                results,
                min_score=0.28,
                max_chunks=6,
                max_chars=7000,
            )

            st.session_state.last_retrieved = manual_results

    else:
        st.warning(
            "No FAISS manual index is available. The Command Center "
            "can still run, but manual grounding will be unavailable."
        )

    st.session_state.command_center_case = {
        "topic": topic,
        "engine_model": engine_model,
        "ship": ship,
        "objective": objective,
        "manual_context": manual_context,
        "manual_results": manual_results,
    }

    if manual_results:
        render_sources(manual_results)

    if manual_context:
        st.success(
            "Manual evidence retrieved. The CrewAI team will use "
            "this evidence as its primary technical source."
        )
    else:
        st.warning(
            "No sufficiently relevant manual evidence was retrieved. "
            "You can explicitly request online research below."
        )

    # --------------------------------------------------------
    # HUMAN-IN-THE-LOOP WEB RESEARCH
    # --------------------------------------------------------

    st.markdown("### Optional Online Research")

    st.write(
        "Online research is not automatic. Click the button below "
        "only if you want the Command Center to use Tavily."
    )

    web_search_requested = st.button(
        "Search Online with Tavily",
        key="command_center_web_search",
    )

    web_context = ""

    if web_search_requested:
        tavily_key = get_secret("TAVILY_API_KEY")

        if not tavily_key:
            st.error(
                "TAVILY_API_KEY is not configured. Add it to "
                "Streamlit Secrets or environment variables."
            )
        else:
            search_query = topic

            if engine_model:
                search_query += f" {engine_model}"

            if ship:
                search_query += f" {ship}"

            if objective:
                search_query += f" {objective}"

            with st.spinner(
                "Command Center Step 2/5 — searching approved online sources..."
            ):
                web_answer = ask_web(
                    """
You are the MarineWise approved web research specialist.

Research the supplied marine technical topic using reliable,
relevant technical sources.

Rules:
- Do not invent facts.
- Clearly distinguish web information from manufacturer manual evidence.
- Prefer manufacturer, classification, regulatory, OEM and authoritative
  technical sources where available.
- Include source titles and URLs when available.
- Do not present general web information as a vessel-specific procedure.
""",
                    (
                        f"TOPIC: {topic}\n"
                        f"ENGINE / EQUIPMENT MODEL: {engine_model or 'Not provided'}\n"
                        f"SHIP: {ship or 'Not provided'}\n"
                        f"OBJECTIVE: {objective or 'Technical investigation'}\n\n"
                        f"MANUAL EVIDENCE ALREADY RETRIEVED:\n"
                        f"{manual_context or '[None]'}\n\n"
                        f"ONLINE RESEARCH QUERY:\n{search_query}"
                    ),
                )

                web_context = web_answer
                st.session_state.command_center_web_context = web_context

            st.success(
                "Approved web research has been added as supplementary evidence."
            )

    else:
        web_context = st.session_state.get(
            "command_center_web_context",
            "",
        )

    # --------------------------------------------------------
    # CREWAI EXECUTION
    # --------------------------------------------------------

    with st.spinner(
        "Command Center Step 3/5 — CrewAI agents are communicating..."
    ):
        result = run_marine_command_center(
            provider=selected_provider(),
            topic=topic,
            engine_model=engine_model,
            ship=ship,
            objective=objective,
            manual_context=manual_context,
            web_context=web_context,
            api_key=require_provider_key(selected_provider()),
        )

    st.session_state.command_center_result = result
    _render_command_center_result(result)


def _render_command_center_result(
    result: dict[str, Any],
) -> None:
    st.markdown("### CrewAI Collaboration")

    agents = result.get("agent_names", []) or []

    if agents:
        cols = st.columns(
            min(len(agents), 5)
        )

        for index, agent_name in enumerate(agents):
            cols[index % len(cols)].success(
                agent_name
            )

    st.markdown("### Final Marine AI Result")

    final_text = safe_text(
        result.get("final")
    )

    if final_text:
        st.markdown(final_text)

        try:
            command_center_pdf = make_command_center_pdf(result)
            st.download_button(
                "Download Final AI Response as PDF",
                data=command_center_pdf,
                file_name="MarineWise_Command_Center_Final_Report.pdf",
                mime="application/pdf",
                type="primary",
                key="download_command_center_pdf",
            )
        except Exception as exc:
            st.error(f"Could not create the Command Center PDF: {exc}")

    task_outputs = result.get(
        "task_outputs",
        [],
    ) or []

    if task_outputs:
        with st.expander(
            "View Agent-to-Agent Communication"
        ):
            for item in task_outputs:
                st.markdown(
                    f"**{safe_text(item.get('task'))}**"
                )
                st.write(
                    safe_text(item.get("output"))
                )

    if result.get("used_web_evidence"):
        st.caption(
            "This result includes supplementary web evidence requested by the user."
        )
    else:
        st.caption(
            "This result was generated without supplementary web evidence."
        )


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
    web_sources: list[dict[str, Any]] | None = None,
) -> bytes:
    """Create a clean, professional A4 technical-training PDF."""

    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        Image as RLImage,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
    )

    font = _register_reportlab_fonts()
    output = io.BytesIO()

    document = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="MarineWise AI - Technical Training",
        author="MarineWise AI",
    )

    registered_fonts = __import__(
        "reportlab.pdfbase.pdfmetrics",
        fromlist=["pdfmetrics"],
    ).getRegisteredFontNames()

    bold_font = (
        f"{font}-Bold"
        if f"{font}-Bold" in registered_fonts
        else font
    )

    title_style = ParagraphStyle(
        "training_title",
        fontName=bold_font,
        fontSize=19,
        leading=23,
        alignment=TA_CENTER,
        textColor=colors.black,
        spaceAfter=10,
    )

    meta_style = ParagraphStyle(
        "training_meta",
        fontName=font,
        fontSize=12,
        leading=17,
        alignment=TA_CENTER,
        textColor=colors.black,
        spaceAfter=12,
    )

    body_style = ParagraphStyle(
        "training_body",
        fontName=font,
        fontSize=12,
        leading=17,
        alignment=TA_JUSTIFY,
        textColor=colors.black,
        spaceAfter=8,
        allowWidows=0,
        allowOrphans=0,
    )

    heading_style = ParagraphStyle(
        "training_heading",
        fontName=bold_font,
        fontSize=15,
        leading=19,
        alignment=TA_LEFT,
        textColor=colors.black,
        spaceBefore=10,
        spaceAfter=7,
    )

    source_style = ParagraphStyle(
        "training_source",
        fontName=font,
        fontSize=10,
        leading=14,
        alignment=TA_LEFT,
        textColor=colors.black,
        spaceAfter=5,
    )

    story = [
        Paragraph(
            "MarineWise AI - Technical Training",
            title_style,
        ),
        Paragraph(
            safe_paragraph(
                f"Engine: {engine or 'Not specified'} | "
                f"Ship: {ship or 'Not specified'} | "
                f"Topic: {topic}"
            ),
            meta_style,
        ),
    ]

    if diagram:
        try:
            story.extend(
                [
                    RLImage(
                        io.BytesIO(diagram),
                        width=174 * mm,
                        height=78 * mm,
                    ),
                    Spacer(1, 10),
                ]
            )
        except Exception:
            # A broken optional diagram must never prevent PDF generation.
            pass

    # Preserve readable sections rather than creating huge dense paragraphs.
    for raw in clean_output_text(content).splitlines():
        line = raw.strip()

        if not line:
            story.append(Spacer(1, 3))
            continue

        line = re.sub(r"^#{1,6}\s*", "", line)

        numbered = re.match(
            r"^(\d+)[.)]\s+(.*)$",
            line,
        )

        if numbered:
            story.append(
                Paragraph(
                    safe_paragraph(
                        f"{numbered.group(1)}. {numbered.group(2)}"
                    ),
                    body_style,
                )
            )
        elif (
            len(line) <= 100
            and line.endswith(":")
        ):
            story.append(
                Paragraph(
                    safe_paragraph(line[:-1]),
                    heading_style,
                )
            )
        else:
            story.append(
                Paragraph(
                    safe_paragraph(line),
                    body_style,
                )
            )

    refs = _source_lines(
        sources or [],
        web_sources or [],
    )

    if refs:
        story.append(PageBreak())
        story.append(
            Paragraph(
                "Sources & References",
                heading_style,
            )
        )

        for ref in refs:
            story.append(
                Paragraph(
                    safe_paragraph(ref),
                    source_style,
                )
            )

    document.build(story)
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
    sources: list[dict[str, Any]] | None = None,
    web_sources: list[dict[str, Any]] | None = None,
) -> bytes:
    """Create a clean, professional Technical Assessment / Quiz PDF."""

    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
    )

    font = _register_reportlab_fonts()
    output = io.BytesIO()

    document = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="MarineWise AI - Technical Assessment / Quiz",
        author="MarineWise AI",
    )

    title_style = ParagraphStyle(
        "quiz_title",
        fontName=font,
        fontSize=19,
        leading=23,
        alignment=TA_CENTER,
        textColor=colors.black,
        spaceAfter=10,
    )

    topic_style = ParagraphStyle(
        "quiz_topic",
        fontName=f"{font}-Bold"
        if f"{font}-Bold" in __import__("reportlab.pdfbase.pdfmetrics", fromlist=["pdfmetrics"]).getRegisteredFontNames()
        else font,
        fontSize=12,
        leading=16,
        alignment=TA_CENTER,
        textColor=colors.black,
        spaceAfter=12,
    )

    question_style = ParagraphStyle(
        "quiz_question",
        fontName=font,
        fontSize=12,
        leading=17,
        alignment=TA_JUSTIFY,
        textColor=colors.black,
        spaceAfter=7,
        allowWidows=0,
        allowOrphans=0,
    )

    option_style = ParagraphStyle(
        "quiz_option",
        fontName=font,
        fontSize=12,
        leading=16,
        alignment=TA_JUSTIFY,
        textColor=colors.black,
        leftIndent=7 * mm,
        firstLineIndent=-5 * mm,
        spaceAfter=5,
        allowWidows=0,
        allowOrphans=0,
    )

    answer_style = ParagraphStyle(
        "quiz_answer",
        fontName=font,
        fontSize=12,
        leading=17,
        alignment=TA_LEFT,
        textColor=colors.black,
        spaceAfter=7,
    )

    source_style = ParagraphStyle(
        "quiz_source",
        fontName=font,
        fontSize=10,
        leading=14,
        alignment=TA_LEFT,
        textColor=colors.black,
        spaceAfter=5,
    )

    bold_font = (
        f"{font}-Bold"
        if f"{font}-Bold"
        in __import__(
            "reportlab.pdfbase.pdfmetrics",
            fromlist=["pdfmetrics"],
        ).getRegisteredFontNames()
        else font
    )

    story = [
        Paragraph(
            "MarineWise AI - Technical Assessment / Quiz",
            title_style,
        ),
        Paragraph(
            safe_paragraph(topic),
            ParagraphStyle(
                "quiz_topic_final",
                parent=topic_style,
                fontName=bold_font,
            ),
        ),
        Paragraph(
            safe_paragraph(f"Question Type: {qtype}"),
            question_style,
        ),
        Spacer(1, 5),
    ]

    # Normalize the AI output and split it into logical lines.
    cleaned = clean_quiz_markdown(text)

    answer_key_match = re.search(
        r"(?is)\bANSWER\s*KEY\b\s*:?\s*(.*)$",
        cleaned,
    )

    if answer_key_match:
        question_text = cleaned[:answer_key_match.start()].strip()
        answer_key_text = answer_key_match.group(1).strip()
    else:
        question_text = cleaned
        answer_key_text = ""

    # Render questions/options separately so every question begins cleanly.
    for raw_line in question_text.splitlines():
        line = raw_line.strip()

        if not line:
            story.append(Spacer(1, 3))
            continue

        line = re.sub(r"^#{1,6}\s*", "", line)

        question_match = re.match(
            r"^(\d+)[.)]\s*(.*)$",
            line,
        )

        option_match = re.match(
            r"^([A-Da-d])[.)]\s*(.*)$",
            line,
        )

        if question_match:
            number = question_match.group(1)
            question = question_match.group(2).strip()
            story.append(
                Paragraph(
                    safe_paragraph(f"{number}. {question}"),
                    question_style,
                )
            )
        elif option_match:
            letter = option_match.group(1).upper()
            option = option_match.group(2).strip()
            story.append(
                Paragraph(
                    safe_paragraph(f"{letter}. {option}"),
                    option_style,
                )
            )
        else:
            story.append(
                Paragraph(
                    safe_paragraph(line),
                    question_style,
                )
            )

    # GUARANTEED: answer key starts on a completely new page.
    story.append(PageBreak())

    story.append(
        Paragraph(
            "Answer Key",
            ParagraphStyle(
                "answer_heading",
                fontName=bold_font,
                fontSize=16,
                leading=20,
                alignment=TA_LEFT,
                textColor=colors.black,
                spaceAfter=10,
            ),
        )
    )

    # Normalize answer key into the requested compact format.
    normalized_pairs: list[str] = []

    for number, letter in re.findall(
        r"(\d+)\s*[\.\):\-]?\s*([A-Da-d])",
        answer_key_text,
    ):
        normalized_pairs.append(
            f"{number}={letter.upper()}"
        )

    if normalized_pairs:
        answer_key_final = ", ".join(normalized_pairs)
    else:
        answer_key_final = clean_output_text(answer_key_text)

    if not answer_key_final:
        answer_key_final = "Answer key was not returned by the assessment generator."

    story.append(
        Paragraph(
            safe_paragraph(answer_key_final),
            answer_style,
        )
    )

    # Sources are kept after the answer key so the answer-key page remains clean.
    refs = _source_lines(
        sources or [],
        web_sources or [],
    )

    if refs:
        story.append(PageBreak())
        story.append(
            Paragraph(
                "Sources & References",
                ParagraphStyle(
                    "source_heading",
                    fontName=bold_font,
                    fontSize=15,
                    leading=19,
                    alignment=TA_LEFT,
                    textColor=colors.black,
                    spaceAfter=9,
                ),
            )
        )

        for ref in refs:
            story.append(
                Paragraph(
                    safe_paragraph(ref),
                    source_style,
                )
            )

    document.build(story)
    return output.getvalue()



def make_remedial_ppt(
    text: str,
) -> bytes:
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.util import Inches, Pt

    prs = Presentation()

    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    parts = split_text(
        text,
        950,
    )

    for index, part in enumerate(
        parts[:8],
        start=1,
    ):
        slide = prs.slides.add_slide(
            prs.slide_layouts[6]
        )

        add_slide_background(
            slide
        )

        add_top_bar(
            slide,
            f"Remedial Training — Part {index}",
            "Remedial",
        )

        box = slide.shapes.add_textbox(
            Inches(0.75),
            Inches(1.4),
            Inches(11.8),
            Inches(5.2),
        )

        tf = box.text_frame

        tf.text = part

        for paragraph in tf.paragraphs:
            paragraph.font.size = Pt(18)
            paragraph.font.color.rgb = (
                RGBColor.from_string(
                    DARK
                )
            )

        add_footer(
            slide,
            index,
            "MarineWise AI — Remedial Training",
        )

    output = io.BytesIO()

    prs.save(
        output
    )

    return output.getvalue()


# ============================================================
# MAIN APPLICATION
# ============================================================


def main() -> None:
    sidebar_manuals()

    page = st.sidebar.radio(
        "Navigate",
        [
            "Troubleshooting Agent",
            "Training Agent",
            "Marine AI Command Center",
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

        elif page == "Marine AI Command Center":
            marine_ai_command_center_page()

        elif page == "Learning":
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
