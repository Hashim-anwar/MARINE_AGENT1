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

# ReportLab Paragraph is used by the troubleshooting PDF generator.
# Import it at module level so the PDF function cannot fail with NameError.
try:
    from reportlab.platypus import Paragraph
except ImportError:
    Paragraph = None

import numpy as np

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


# ============================================================
# OUTPUT QUALITY / DOCUMENT HELPERS
# ============================================================


def clean_output_text(value: Any) -> str:
    """Normalize generated text before sending it to PDF/Word/PPT."""
    import unicodedata

    text = safe_text(value)
    replacements = {
        "\u00a0": " ",
        "\u200b": "",
        "\u200c": "",
        "\u200d": "",
        "\ufeff": "",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u2022": "-",
        "\u2192": "->",
        "\u2190": "<-",
        "\u2713": "[OK]",
        "\u2714": "[OK]",
        "\u2717": "[X]",
        "\u2718": "[X]",
        "\u00d7": "x",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Remove invisible Unicode formatting/control characters that can render
    # as black dots/squares in PDF viewers while preserving tabs/newlines.
    cleaned = []
    for char in text:
        category = unicodedata.category(char)
        if category == "Cf":
            continue
        if category == "Cc" and char not in {"\n", "\t", "\r"}:
            continue
        cleaned.append(char)

    text = "".join(cleaned)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _find_document_font() -> tuple[str, str | None, str | None, str | None]:
    """Return (font_name, regular_path, bold_path, italic_path).

    Times New Roman is preferred. A Unicode serif fallback is used only when
    Times New Roman is not installed on the host running Streamlit.
    """
    candidates = [
        (
            "Times New Roman",
            r"C:\\Windows\\Fonts\\times.ttf",
            r"C:\\Windows\\Fonts\\timesbd.ttf",
            r"C:\\Windows\\Fonts\\timesi.ttf",
        ),
        (
            "Times New Roman",
            r"/Library/Fonts/Times New Roman.ttf",
            r"/Library/Fonts/Times New Roman Bold.ttf",
            r"/Library/Fonts/Times New Roman Italic.ttf",
        ),
        (
            "Times New Roman",
            r"/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
            r"/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf",
            r"/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Italic.ttf",
        ),
    ]

    for name, regular, bold, italic in candidates:
        if os.path.exists(regular):
            return name, regular, bold if os.path.exists(bold) else None, italic if os.path.exists(italic) else None

    try:
        from matplotlib import font_manager
        regular = font_manager.findfont(font_manager.FontProperties(family="DejaVu Serif"))
        bold = font_manager.findfont(font_manager.FontProperties(family="DejaVu Serif", weight="bold"))
        italic = font_manager.findfont(font_manager.FontProperties(family="DejaVu Serif", style="italic"))
        return "DejaVu Serif", regular, bold, italic
    except Exception:
        return "Times-Roman", None, None, None


def _register_reportlab_fonts() -> str:
    """Register a Unicode serif font for ReportLab and return its family name."""
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    family, regular, bold, italic = _find_document_font()
    if regular:
        safe_family = "MarineWiseSerif"
        try:
            if safe_family not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont(safe_family, regular))
                if bold:
                    pdfmetrics.registerFont(TTFont(f"{safe_family}-Bold", bold))
                if italic:
                    pdfmetrics.registerFont(TTFont(f"{safe_family}-Italic", italic))
                if bold and italic:
                    try:
                        pdfmetrics.registerFont(TTFont(f"{safe_family}-BoldItalic", bold))
                    except Exception:
                        pass
        except Exception:
            return "Times-Roman"
        return safe_family
    return "Times-Roman"


def _set_docx_run_font(run, font_name: str = "Times New Roman", size: int = 14, bold: bool | None = None) -> None:
    from docx.shared import Pt

    run.font.name = font_name
    run.font.size = Pt(size)
    # Make the font name explicit for East Asia/Word fallback as well.
    try:
        from docx.oxml.ns import qn
        rpr = run._element.get_or_add_rPr()
        rfonts = rpr.get_or_add_rFonts()
        rfonts.set(qn("w:ascii"), font_name)
        rfonts.set(qn("w:hAnsi"), font_name)
        rfonts.set(qn("w:eastAsia"), font_name)
    except Exception:
        pass
    if bold is not None:
        run.bold = bold


def _add_docx_paragraph(document, text: str, size: int = 14, bold: bool = False, align: int = 3):
    """Add a clean Times New Roman paragraph; align=3 is justified."""
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    paragraph = document.add_paragraph()
    paragraph.alignment = {
        0: WD_ALIGN_PARAGRAPH.LEFT,
        1: WD_ALIGN_PARAGRAPH.CENTER,
        2: WD_ALIGN_PARAGRAPH.RIGHT,
        3: WD_ALIGN_PARAGRAPH.JUSTIFY,
    }.get(align, WD_ALIGN_PARAGRAPH.JUSTIFY)
    from docx.shared import Pt
    paragraph.paragraph_format.space_after = Pt(6)
    run = paragraph.add_run(clean_output_text(text))
    _set_docx_run_font(run, "Times New Roman", size, bold)
    return paragraph


def _add_ppt_text_box(slide, text: str, left: float, top: float, width: float, height: float,
                      font_size: int = 14, bold: bool = False, color: str = DARK,
                      align: int = 0, font_name: str = "Times New Roman"):
    """Add a consistently formatted PPT text box with safe wrapping."""
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
    from pptx.util import Inches, Pt

    box = slide.shapes.add_textbox(
        Inches(left), Inches(top), Inches(width), Inches(height)
    )
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = Inches(0.04)
    tf.margin_right = Inches(0.04)
    tf.margin_top = Inches(0.03)
    tf.margin_bottom = Inches(0.03)
    tf.vertical_anchor = MSO_ANCHOR.TOP
    p = tf.paragraphs[0]
    p.text = clean_output_text(text)
    p.alignment = {
        0: PP_ALIGN.LEFT,
        1: PP_ALIGN.CENTER,
        2: PP_ALIGN.RIGHT,
        3: PP_ALIGN.JUSTIFY,
    }.get(align, PP_ALIGN.LEFT)
    p.font.name = font_name
    p.font.size = Pt(font_size)
    p.font.bold = bold
    p.font.color.rgb = RGBColor.from_string(color)
    return box


def _add_ppt_bullets(slide, bullets: list[str], left: float, top: float, width: float, height: float,
                     font_size: int = 14) -> None:
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
    from pptx.util import Inches, Pt

    box = slide.shapes.add_textbox(
        Inches(left), Inches(top), Inches(width), Inches(height)
    )
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = Inches(0.08)
    tf.margin_right = Inches(0.06)
    tf.margin_top = Inches(0.04)
    tf.margin_bottom = Inches(0.04)
    tf.vertical_anchor = MSO_ANCHOR.TOP

    cleaned = [clean_output_text(b) for b in bullets if clean_output_text(b)]
    for index, bullet in enumerate(cleaned[:6]):
        p = tf.paragraphs[0] if index == 0 else tf.add_paragraph()
        p.text = f"• {bullet}"
        p.font.name = "Times New Roman"
        p.font.size = Pt(font_size)
        p.font.color.rgb = RGBColor.from_string(DARK)
        p.space_after = Pt(8)
        p.alignment = PP_ALIGN.LEFT
        p.level = 0


def _source_lines(sources: list[dict[str, Any]] | None = None,
                  web_sources: list[dict[str, Any]] | None = None) -> list[str]:
    lines: list[str] = []
    seen: set[str] = set()
    for source in sources or []:
        name = clean_output_text(source.get("source"))
        page = source.get("page", "")
        if name:
            line = f"Manual: {name} — page {page}"
            if line not in seen:
                seen.add(line)
                lines.append(line)
    for source in web_sources or []:
        title = clean_output_text(source.get("title")) or "Technical web source"
        url = clean_output_text(source.get("url"))
        if url:
            line = f"Web: {title} — {url}"
            if line not in seen:
                seen.add(line)
                lines.append(line)
    return lines


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


def make_troubleshooting_pdf(
    case: dict[str, Any],
    manual_answer: str | None,
    sources: list[dict[str, Any]] | None = None,
    web_answer: str | None = None,
    web_sources: list[dict[str, Any]] | None = None,
) -> bytes:
    """Create a structured troubleshooting PDF with 14pt Times New Roman body text.

    The troubleshooting workflow is deliberately kept separate from the existing
    training-output architecture. This function only formats the already-approved
    troubleshooting answer(s) for download and does not change the human-in-loop
    web-search decision.
    """
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

    font_name = _register_reportlab_fonts()
    output = io.BytesIO()
    document = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=clean_output_text(
            f"MarineWise AI - Troubleshooting - {case.get('defect', '')}"
        ),
        author="MarineWise AI",
    )

    title_style = ParagraphStyle(
        "TroubleTitle", fontName=font_name, fontSize=20, leading=24,
        alignment=TA_CENTER, textColor=colors.HexColor("#17324D"), spaceAfter=8,
    )
    meta_style = ParagraphStyle(
        "TroubleMeta", fontName=font_name, fontSize=14, leading=20,
        alignment=TA_CENTER, textColor=colors.HexColor("#344054"), spaceAfter=12,
    )
    body_style = ParagraphStyle(
        "TroubleBody", fontName=font_name, fontSize=14, leading=20,
        alignment=TA_JUSTIFY, textColor=colors.HexColor("#17202A"),
        spaceAfter=8, allowWidows=0, allowOrphans=0,
    )
    step_style = ParagraphStyle(
        "TroubleStep", fontName=font_name, fontSize=14, leading=20,
        alignment=TA_JUSTIFY, textColor=colors.HexColor("#17202A"),
        leftIndent=7 * mm, firstLineIndent=-7 * mm, spaceAfter=10,
        allowWidows=0, allowOrphans=0,
    )
    heading_style = ParagraphStyle(
        "TroubleHeading", fontName=font_name, fontSize=17, leading=21,
        alignment=TA_LEFT, textColor=colors.HexColor("#17324D"),
        spaceBefore=10, spaceAfter=7,
    )
    source_style = ParagraphStyle(
        "TroubleSource", fontName=font_name, fontSize=11, leading=16,
        alignment=TA_LEFT, textColor=colors.HexColor("#475467"), spaceAfter=5,
    )

    manufacturer = clean_output_text(case.get("manufacturer")) or "Not specified"
    engine_model = clean_output_text(case.get("engine_model")) or "Not specified"
    serial = clean_output_text(case.get("serial")) or "Not provided"
    defect = clean_output_text(case.get("defect")) or "Not specified"

    story = [
        Paragraph("MarineWise AI - Step-by-Step Troubleshooting", title_style),
        Paragraph(
            safe_paragraph(
                f"Manufacturer: {manufacturer} | Engine Model: {engine_model} | "
                f"Serial: {serial}"
            ),
            meta_style,
        ),
        Paragraph("Reported Defect / Alarm", heading_style),
        Paragraph(safe_paragraph(defect), body_style),
    ]

    if manual_answer and manual_answer.strip():
        story.append(Paragraph("Manual-Grounded Troubleshooting Procedure", heading_style))
        story.append(
            Paragraph(
                safe_paragraph(
                    "The following procedure is grounded in the supplied manual excerpts. "
                    "Follow the sequence, verify each expected result before proceeding, "
                    "and use the current manufacturer procedure for any final adjustment or repair."
                ),
                body_style,
            )
        )
        _append_troubleshooting_answer_flow(story, manual_answer, body_style, step_style, heading_style)

    if web_answer and web_answer.strip():
        story.append(PageBreak())
        story.append(Paragraph("Web-Sourced Supplementary Troubleshooting", heading_style))
        story.append(
            Paragraph(
                safe_paragraph(
                    "This section was added only after the human-in-loop online-search approval. "
                    "Verify all web-derived information against the current manufacturer manual "
                    "before carrying out work on the equipment."
                ),
                body_style,
            )
        )
        _append_troubleshooting_answer_flow(story, web_answer, body_style, step_style, heading_style)

    source_lines = _source_lines(sources or [], web_sources or [])
    if source_lines:
        story.append(PageBreak())
        story.append(Paragraph("Sources & References", heading_style))
        story.append(
            Paragraph(
                safe_paragraph(
                    "Manual sources are the primary evidence. Web sources are included only "
                    "when online research was explicitly approved by the user."
                ),
                body_style,
            )
        )
        for line in source_lines:
            story.append(Paragraph(safe_paragraph(line), source_style))

    document.build(story)
    return output.getvalue()


def _append_troubleshooting_answer_flow(
    story: list,
    answer: str,
    body_style: Any,
    step_style: Any,
    heading_style: Any,
) -> None:
    """Render troubleshooting text as readable numbered steps without changing its content."""
    text = clean_output_text(answer)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return

    for line in lines:
        normalized = re.sub(r"^[-*•]+\s*", "", line).strip()
        normalized = re.sub(r"^#{1,6}\s*", "", normalized).strip()
        numbered = re.match(r"^(\d+)[.)]\s+(.*)$", normalized)
        heading = (
            len(normalized) <= 90
            and normalized.endswith(":")
            and not numbered
        )
        if heading:
            story.append(Paragraph(safe_paragraph(normalized.rstrip(":")), heading_style))
        elif numbered:
            story.append(Paragraph(safe_paragraph(f"{numbered.group(1)}. {numbered.group(2)}"), step_style))
        else:
            story.append(Paragraph(safe_paragraph(normalized), body_style))



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
        st.session_state.troubleshooting_web_error = None
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

Your response must be a properly refined, technician-ready, STEP-BY-STEP troubleshooting procedure.
Do not give a loose paragraph or generic explanation. Organize the response in this order:
1. Problem / symptom interpretation
2. Safety and isolation precautions that are explicitly supported by the manual
3. Step-by-step checks in the correct diagnostic sequence
4. Expected result for each check
5. What to do if the expected result is NOT obtained
6. Root-cause conclusion only when supported by the manual
7. Corrective action only when supported by the manual
8. Final verification / return-to-service checks when supported

For every diagnostic step, make the action clear and practical. Keep the sequence logically aligned: start with safe, simple observations/checks before more involved checks, unless the manual specifies another order. Do not invent a sequence when the manual does not support it.

Do not invent:
- specifications
- causes
- alarm limits
- procedures
- component locations
- maintenance intervals
- measurements or acceptance criteria

If a required detail is missing from the excerpts, explicitly say that it is not stated in the supplied manual instead of guessing.

If the excerpts do not actually answer the question, return exactly:

Not found in manuals. Do you want me to search online?

Cite the source file and exact page number for claims that are supported by the excerpts.
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

            # The model may repeat the human-in-loop question even though the UI
            # already renders that question. Store it as a clean not-found state
            # so the message is shown only once.
            manual_not_found_markers = (
                "Not found in manuals. Do you want me to search online?",
                "Not found in manuals",
            )
            if any(
                marker.lower() in safe_text(answer).lower()
                for marker in manual_not_found_markers
            ):
                st.session_state.troubleshooting_answer = ""
            else:
                st.session_state.troubleshooting_answer = answer

        else:
            st.session_state.troubleshooting_answer = ""

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

    # Human-in-the-loop gate: show this prompt exactly once. The answer itself
    # is kept free of the UI question so it is not duplicated.
    manual_not_found = not context or not answer

    if (
        manual_not_found
        and st.session_state.get(
            "troubleshooting_web_answer"
        )
        is None
    ):
        st.warning(
            "Not found in the supplied manuals. "
            "Would you like MarineWise AI to search reliable online technical sources?"
        )

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
            # Keep the existing human-in-the-loop architecture, but do not let a
            # temporary provider outage crash the entire Streamlit page.
            try:
                with st.spinner(
                    "Searching online technical sources..."
                ):
                    web_answer = ask_web(
                        """
You are a marine engine troubleshooting assistant.

This answer is FROM THE WEB, not from the supplied manuals.
Search reliable manufacturer documentation and reputable technical sources.

Return a properly refined, technician-ready STEP-BY-STEP troubleshooting procedure. Organize it as:
1. Problem / symptom interpretation
2. Safety and isolation
3. Diagnostic checks in sequence
4. Expected result for each check
5. Decision point if the result is abnormal
6. Corrective action when supported by the source
7. Final verification

Do not invent specifications, alarm limits, procedures, component locations, measurements, or maintenance intervals. Clearly identify information that is web-derived and instruct the technician to verify it against the current engine manual.
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

                st.session_state.troubleshooting_web_answer = web_answer
                st.session_state.troubleshooting_web_error = None

            except Exception as exc:
                error_text = safe_text(exc)
                is_temporary_provider_error = (
                    "503" in error_text
                    or "UNAVAILABLE" in error_text.upper()
                    or "high demand" in error_text.lower()
                    or "temporarily unavailable" in error_text.lower()
                )

                st.session_state.troubleshooting_web_answer = None
                if is_temporary_provider_error:
                    st.session_state.troubleshooting_web_error = (
                        "The online technical-search service is temporarily unavailable "
                        "(provider returned HTTP 503). Your manual-only troubleshooting "
                        "workflow is still available. Please try the online search again shortly."
                    )
                else:
                    st.session_state.troubleshooting_web_error = (
                        "The online technical search could not be completed. "
                        "Please verify the selected provider API key and TAVILY_API_KEY, "
                        "then try again."
                    )

    web_error = st.session_state.get(
        "troubleshooting_web_error"
    )

    if web_error:
        st.warning(web_error)
        if st.session_state.get("troubleshooting_web_choice") == "yes":
            if st.button(
                "Retry Online Search",
                key="retry_web_troubleshoot",
            ):
                st.session_state.troubleshooting_web_answer = None
                st.session_state.troubleshooting_web_error = None
                st.rerun()

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

    # --------------------------------------------------------
    # DOWNLOADABLE STEP-BY-STEP TROUBLESHOOTING PDF
    # --------------------------------------------------------
    final_manual_answer = st.session_state.get("troubleshooting_answer")
    final_web_answer = st.session_state.get("troubleshooting_web_answer")
    if final_manual_answer or final_web_answer:
        try:
            troubleshooting_pdf = make_troubleshooting_pdf(
                case=case,
                manual_answer=final_manual_answer,
                sources=relevant,
                web_answer=final_web_answer,
                web_sources=st.session_state.get("troubleshooting_web_sources", []),
            )
            filename_model = re.sub(
                r"[^A-Za-z0-9_-]+",
                "_",
                case.get("engine_model", "engine"),
            ).strip("_") or "engine"
            st.download_button(
                "Download Step-by-Step Troubleshooting PDF",
                data=troubleshooting_pdf,
                file_name=f"MarineWise_Troubleshooting_{filename_model}.pdf",
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


def make_topic_plan_visual(
    engine: str,
    topic: str,
    plan: list[dict[str, Any]] | None = None,
) -> bytes:
    """Create a topic-aware technical schematic, not a generic architecture card grid.

    Priority:
      1) classify the requested topic into a marine-system diagram family;
      2) use source-grounded plan text to enrich labels where possible;
      3) fall back to a plan-driven process diagram for unfamiliar topics.

    The diagram is deliberately conceptual unless an actual manual figure is supplied
    elsewhere by the app. It must not invent manufacturer-specific component details.
    """
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    import matplotlib.pyplot as plt

    topic_clean = clean_output_text(topic) or "Technical Training Topic"
    engine_clean = clean_output_text(engine) or "Marine Engine"
    topic_l = topic_clean.lower()

    # Collect source-grounded plan terms. These are used only as supporting labels;
    # the system-flow itself is selected from the topic family so the visual is meaningful.
    plan_items: list[dict[str, Any]] = []
    for raw in plan or []:
        if not isinstance(raw, dict):
            continue
        title = clean_output_text(raw.get("title", ""))
        bullets = raw.get("bullets", [])
        if not isinstance(bullets, list):
            bullets = [bullets]
        bullets_clean = [
            clean_output_text(b) for b in bullets if clean_output_text(b)
        ]
        if title:
            plan_items.append({"title": title, "bullets": bullets_clean})

    def has(*words: str) -> bool:
        return any(w in topic_l for w in words)

    # Topic-specific marine system families.
    if has("exhaust", "aftertreatment", "after-treatment", "silencer", "emission"):
        family = "Exhaust Gas Path"
        nodes = [
            ("Engine Cylinders", "Exhaust gas generated"),
            ("Exhaust Manifold", "Collects exhaust flow"),
            ("Turbocharger Turbine", "Uses exhaust energy"),
            ("Exhaust Treatment / Silencer", "Noise / emissions control where fitted"),
            ("Exhaust Outlet / Stack", "Discharge to atmosphere"),
        ]
    elif has("fuel", "injection", "injector", "fuel oil", "diesel supply"):
        family = "Fuel Supply & Injection Path"
        nodes = [
            ("Fuel Source / Tank", "Stored fuel"),
            ("Strainer / Filter", "Removes contaminants"),
            ("Fuel Supply", "Delivers fuel to engine"),
            ("Injection Equipment", "Meters / injects fuel"),
            ("Cylinders", "Fuel combustion"),
            ("Return / Recirculation", "Returns excess fuel where applicable"),
        ]
    elif has("cooling", "jacket water", "fresh water", "sea water", "seawater", "heat exchanger"):
        family = "Cooling Circuit"
        nodes = [
            ("Cooling Source", "Cooling medium enters circuit"),
            ("Pump", "Circulates cooling medium"),
            ("Heat Exchanger / Cooler", "Transfers heat"),
            ("Engine Cooling Circuit", "Removes engine heat"),
            ("Temperature Control", "Controls operating temperature"),
            ("Return", "Cooling medium returns to circuit"),
        ]
    elif has("lubric", "lube oil", "lubrication oil", "oil system", "oil filter"):
        family = "Lubricating Oil Circuit"
        nodes = [
            ("Oil Sump / Tank", "Lubricant reservoir"),
            ("Lube Oil Pump", "Builds circulation flow"),
            ("Filter", "Removes contaminants"),
            ("Oil Cooler", "Controls oil temperature where fitted"),
            ("Engine Lubrication Points", "Lubricates moving components"),
            ("Return", "Oil drains back to reservoir"),
        ]
    elif has("turbocharger", "turbo charger", "boost", "charge air", "charge-air"):
        family = "Turbocharging & Charge-Air Path"
        nodes = [
            ("Exhaust Gas", "Energy source"),
            ("Turbine", "Converts exhaust energy"),
            ("Common Shaft", "Transfers rotational energy"),
            ("Compressor", "Compresses intake air"),
            ("Charge-Air Cooler", "Cools compressed air where fitted"),
            ("Engine Intake", "Supplies combustion air"),
        ]
    elif has("starting", "starter", "start system", "cranking"):
        family = "Engine Starting Sequence"
        nodes = [
            ("Starting Source", "Electrical / pneumatic source as documented"),
            ("Start Control", "Initiates starting sequence"),
            ("Starter", "Applies cranking torque"),
            ("Flywheel / Crankshaft", "Turns engine"),
            ("Fuel + Air Enable", "Supports engine firing"),
            ("Engine Running", "Starting sequence completes"),
        ]
    elif has("electrical", "alternator", "generator", "charging", "battery", "switchboard"):
        family = "Electrical / Charging Path"
        nodes = [
            ("Prime Mover", "Mechanical input"),
            ("Alternator / Generator", "Produces electrical power"),
            ("Regulation / Protection", "Controls and protects output"),
            ("Battery / Switchboard", "Stores or distributes power"),
            ("Electrical Loads", "Consume generated power"),
            ("Feedback / Monitoring", "Voltage / current / alarms"),
        ]
    elif has("governor", "speed control", "engine control", "ecu", "control system", "automation"):
        family = "Engine Control & Feedback"
        nodes = [
            ("Speed / Load Demand", "Operating demand"),
            ("Controller / Governor", "Processes demand"),
            ("Actuator", "Commands engine response"),
            ("Fuel / Engine Response", "Changes engine output"),
            ("Speed / Load Feedback", "Measures actual response"),
            ("Control Correction", "Closes the control loop"),
        ]
    elif has("air intake", "intake air", "air filter", "induction"):
        family = "Engine Air Intake Path"
        nodes = [
            ("Ambient Air", "Intake air source"),
            ("Air Filter", "Removes contaminants"),
            ("Compressor / Turbocharger", "Raises intake pressure where fitted"),
            ("Charge-Air Cooler", "Controls charge-air temperature where fitted"),
            ("Intake Manifold", "Distributes air"),
            ("Engine Cylinders", "Combustion air enters cylinders"),
        ]
    else:
        # Unknown topic: derive a genuine flow from the plan instead of using
        # a universal INPUT -> CONTROL -> ENGINE -> MONITOR architecture.
        family = f"{topic_clean} — Technical Process"
        candidates = []
        for item in plan_items:
            title = item["title"]
            if title not in [x[0] for x in candidates]:
                candidates.append((title, "Source-grounded training stage"))
        nodes = candidates[:6]
        if len(nodes) < 4:
            nodes = [
                (f"{topic_clean} — Function", "Purpose and operating role"),
                (f"{topic_clean} — Components", "Main components documented in source"),
                (f"{topic_clean} — Operation", "Normal operating sequence"),
                (f"{topic_clean} — Verification", "Technician checks and indications"),
                (f"{topic_clean} — Faults", "Documented fault / diagnosis path"),
            ]

    # Compact long labels while preserving the actual technical wording.
    nodes = nodes[:6]

    fig, ax = plt.subplots(figsize=(13.33, 6.2))
    ax.set_xlim(0, 13.33)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    ax.text(
        6.665, 5.72, f"{engine_clean} — {topic_clean}",
        ha="center", va="center", fontsize=18, fontweight="bold",
    )
    ax.text(
        6.665, 5.30, family,
        ha="center", va="center", fontsize=10, fontweight="bold",
    )
    ax.text(
        6.665, 5.00,
        "Conceptual training schematic — verify component arrangement against the supplied manual",
        ha="center", va="center", fontsize=8.5,
    )

    n = len(nodes)
    card_w = 1.85 if n >= 6 else 2.05
    card_h = 1.45
    gap = 0.28
    total_w = n * card_w + (n - 1) * gap
    start_x = (13.33 - total_w) / 2
    y = 2.65

    # Use the plan to provide a small evidence note, without replacing the system flow.
    plan_text = ""
    if plan_items:
        relevant = plan_items[0]["title"]
        plan_text = f"Training plan: {relevant}"

    for i, (title, desc) in enumerate(nodes):
        x = start_x + i * (card_w + gap)
        patch = FancyBboxPatch(
            (x, y), card_w, card_h,
            boxstyle="round,pad=0.06,rounding_size=0.08",
            linewidth=1.5,
        )
        ax.add_patch(patch)
        ax.text(
            x + card_w / 2, y + 0.98,
            textwrap.fill(clean_output_text(title), width=20),
            ha="center", va="center", fontsize=9.8, fontweight="bold",
        )
        ax.text(
            x + card_w / 2, y + 0.38,
            textwrap.fill(clean_output_text(desc), width=24),
            ha="center", va="center", fontsize=7.6,
        )

        if i < n - 1:
            x1 = x + card_w + 0.02
            x2 = x + card_w + gap - 0.02
            arrow = FancyArrowPatch(
                (x1, y + card_h / 2),
                (x2, y + card_h / 2),
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=1.4,
            )
            ax.add_patch(arrow)

    # A restrained evidence footer makes it clear this is a training schematic,
    # while the actual manual figure (when available) remains the preferred visual.
    ax.text(
        0.55, 0.55,
        "Source priority: relevant manual figure/page first; this schematic is used only when a suitable figure is unavailable.",
        ha="left", va="center", fontsize=7.6,
    )
    if plan_text:
        ax.text(
            12.78, 0.55,
            textwrap.fill(plan_text, width=34),
            ha="right", va="center", fontsize=7.2,
        )

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()

def make_training_diagram(
    engine: str,
    topic: str,
    plan: list[dict[str, Any]] | None = None,
) -> bytes:
    # Use the real topic/presentation plan. The legacy fixed
    # architecture is retained above only for backward compatibility
    # with any other internal caller, but is no longer used here.
    return make_topic_plan_visual(
        engine,
        topic,
        plan,
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
    p.font.name = "Times New Roman"
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
    cp.font.name = "Times New Roman"
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
    """Build a topic-grounded, professional 16:9 PowerPoint.

    The renderer deliberately uses the AI presentation plan instead of a
    hard-coded architecture. Manual visuals are consumed first and different
    manual pages are distributed across the deck before online images are used.
    """
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.enum.shapes import MSO_SHAPE
    from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
    from pptx.util import Inches, Pt

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    normalized_plan = [item for item in (plan or []) if isinstance(item, dict)]
    if not normalized_plan:
        normalized_plan = [
            {
                "title": "System Overview",
                "purpose": "Explain the requested technical topic.",
                "bullets": [topic],
                "visual_type": "process_diagram",
                "visual_query": topic,
                "source_preference": "manual_first",
            }
        ]

    # Keep the requested 8–10 slide structure when the planner returns enough
    # information; never pad a topic with unrelated generic systems.
    deck_plan = normalized_plan[:9]
    fallback_titles = [
        ("Topic Summary", "Summarize the requested topic.", "summary"),
        ("Technician Checks", "Apply the source-supported inspection logic.", "checklist"),
        ("Knowledge Check", "Reinforce the topic-specific learning points.", "questions"),
        ("References", "Show the evidence used for this presentation.", "references"),
    ]
    fallback_index = 0
    while len(deck_plan) < 8:
        title, purpose, visual = fallback_titles[min(fallback_index, len(fallback_titles) - 1)]
        deck_plan.append({
            "title": title,
            "purpose": purpose,
            "bullets": [
                clean_output_text(topic),
                "Use the cited source material for the detailed technical explanation.",
                "Do not substitute generic information for manufacturer-specific instructions.",
            ],
            "visual_type": visual,
            "source_preference": "all_sources" if visual == "references" else "manual_first",
        })
        fallback_index += 1
    deck_plan = deck_plan[:9]

    manual_pool = [x for x in (manual_images or []) if x.get("bytes")]
    online_pool = [x for x in (online_images or []) if x.get("bytes")]
    image_pool = manual_pool + online_pool
    image_index = 0

    def next_image(prefer_manual: bool = True):
        nonlocal image_index
        if image_index < len(image_pool):
            item = image_pool[image_index]
            image_index += 1
            return item
        if prefer_manual and manual_pool:
            # Reuse only after every distinct manual visual has been used.
            return manual_pool[0]
        return None

    def add_header(slide, title: str, section: str, number: int):
        add_slide_background(slide, LIGHT)
        add_top_bar(slide, clean_output_text(title)[:100], clean_output_text(section)[:28])
        add_footer(
            slide,
            number,
            f"{engine} | {topic} | Manual-first technical training",
        )

    def add_source_badge(slide, item, left=9.0, top=5.85, width=3.6):
        if not item:
            return
        source = clean_output_text(item.get("source"))
        page = item.get("page")
        if source:
            caption = f"Manual: {source} — page {page}"
        else:
            caption = clean_output_text(item.get("title")) or "Web technical image"
        _add_ppt_text_box(
            slide,
            caption,
            left,
            top,
            width,
            0.55,
            font_size=8,
            color=GRAY,
            align=2,
        )

    # --------------------------------------------------------
    # TITLE SLIDE
    # --------------------------------------------------------
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_background(slide, NAVY)

    accent = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Inches(0.7), Inches(1.0), Inches(1.6), Inches(0.12)
    )
    accent.fill.solid()
    accent.fill.fore_color.rgb = RGBColor.from_string(TEAL)
    accent.line.fill.background()

    _add_ppt_text_box(
        slide, clean_output_text(topic), 0.7, 1.45, 11.8, 1.45,
        font_size=34, bold=True, color=WHITE
    )
    _add_ppt_text_box(
        slide,
        clean_output_text(f"{engine}" + (f"  |  {ship}" if ship else "")),
        0.75, 3.05, 11.0, 0.8,
        font_size=20, color="C9D9E6"
    )
    _add_ppt_text_box(
        slide,
        "MarineWise AI • Technical Training • Manual-first evidence",
        0.75, 5.55, 8.5, 0.55,
        font_size=12, bold=True, color="9CCFD2"
    )
    add_footer(slide, 1, "Primary source: supplied manuals. Secondary source: reliable technical web research.")

    # --------------------------------------------------------
    # CONTENT SLIDES: driven by the actual plan
    # --------------------------------------------------------
    for slide_no, item in enumerate(deck_plan, start=2):
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        title = clean_output_text(item.get("title") or f"{topic} — Section {slide_no - 1}")
        purpose = clean_output_text(item.get("purpose"))
        bullets = item.get("bullets", [])
        if not isinstance(bullets, list):
            bullets = [bullets]
        bullets = [clean_output_text(x) for x in bullets if clean_output_text(x)]
        visual_type = clean_output_text(item.get("visual_type")).lower()

        add_header(slide, title, visual_type or "Training", slide_no)

        # Choose visuals by topic/slide purpose, not one fixed architecture.
        wants_visual = visual_type not in {"none", "objectives", "questions", "references"}
        visual_item = next_image() if wants_visual else None

        if visual_item:
            add_picture_card(
                slide,
                visual_item["bytes"],
                clean_output_text(visual_item.get("title")) or "Technical reference",
                (
                    f"{clean_output_text(visual_item.get('source'))} — page {visual_item.get('page')}"
                    if visual_item.get("source")
                    else clean_output_text(visual_item.get("title")) or "Online technical image"
                ),
                0.65, 1.28, 6.9, 5.55,
            )
            _add_ppt_text_box(
                slide,
                purpose or "Evidence-based technical explanation.",
                7.85, 1.35, 4.75, 0.85,
                font_size=13, bold=True, color=NAVY,
            )
            _add_ppt_bullets(
                slide,
                bullets or ["Refer to the supplied source for the detailed procedure."],
                7.8, 2.25, 4.85, 3.25,
                font_size=14,
            )
            add_source_badge(slide, visual_item, 8.0, 5.8, 4.4)
        elif visual_type in {"process", "flow", "flowchart", "process_diagram", "diagram"}:
            steps = bullets[:5] or ["Input", "Control", "Process", "Feedback", "Output"]
            add_process_shapes(slide, steps, left=0.65, top=2.0)
            _add_ppt_text_box(
                slide,
                purpose or "Trace the topic-specific operating sequence.",
                0.9, 1.15, 11.5, 0.65,
                font_size=14, bold=True, color=NAVY, align=1,
            )
        elif visual_type == "references":
            refs = _source_lines(sources, web_sources)
            _add_ppt_bullets(
                slide,
                refs or ["No source records were returned."],
                0.75, 1.35, 11.8, 5.3,
                font_size=13,
            )
        else:
            _add_ppt_text_box(
                slide,
                purpose or "Technical training content",
                0.8, 1.3, 11.7, 0.75,
                font_size=15, bold=True, color=NAVY,
            )
            _add_ppt_bullets(
                slide,
                bullets or ["Use the cited source material for the topic-specific details."],
                0.8, 2.15, 11.6, 4.15,
                font_size=15,
            )

        # Add a compact source strip on every slide so the deck never loses
        # provenance when a user exports or prints individual slides.
        source_hint = clean_output_text(item.get("source_preference"))
        if source_hint:
            _add_ppt_text_box(
                slide,
                f"Evidence mode: {source_hint}",
                0.65, 6.72, 5.0, 0.25,
                font_size=7, color=GRAY,
            )

    # --------------------------------------------------------
    # FINAL SOURCE SLIDE
    # --------------------------------------------------------
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header(slide, "References & Source Traceability", "Sources", len(prs.slides) + 1)
    refs = _source_lines(sources, web_sources)
    if not refs:
        refs = ["No external source records were returned.", "Verify all technical details against the current manufacturer manual."]
    _add_ppt_bullets(slide, refs[:12], 0.7, 1.25, 11.9, 5.55, font_size=11)

    output = io.BytesIO()
    prs.save(output)
    return output.getvalue()


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
        # Topic is the primary training input. Engine model and ship are
        # optional context fields so the Training Agent can also handle
        # workshop/marine-mechanical topics that are not engine-specific.
        topic = st.text_input(
            "Training Topic *",
            placeholder="Bow Thruster / Shaft Alignment / Propeller / Cooling System",
            help=(
                "Enter the technical subject first. This can be an engine topic, "
                "marine propulsion topic, workshop topic, deck machinery topic, "
                "or another mechanical training subject."
            ),
        )

        engine = st.text_input(
            "Engine Model (Optional)",
            placeholder="MTU 16V 4000 M90",
            help=(
                "Optional. Use this when the training needs to be specific to an "
                "engine/model. Leave blank for general mechanical or marine topics."
            ),
        )

        ship = st.text_input(
            "Ship (Optional)",
            placeholder="MV Example",
            help=(
                "Optional. Use this when the training should be specific to a ship."
            ),
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

    if not topic.strip():
        st.warning(
            "Enter a training topic first. Engine Model and Ship are optional."
        )
        return

    # Build a clean optional context string. Do not force an engine model into
    # retrieval, research, prompts, or output when the user leaves it blank.
    training_context = " ".join(
        part.strip()
        for part in [engine, ship]
        if part and part.strip()
    )
    retrieval_prefix = f"{training_context} " if training_context else ""

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
            primary_results = search_index(
                rag,
                f"{retrieval_prefix}{topic} system components operation troubleshooting",
                get_embedder(),
                k=10,
            )
            visual_results = search_index(
                rag,
                f"{retrieval_prefix}{topic} diagram schematic architecture components",
                get_embedder(),
                k=10,
            )
            # Merge both retrieval passes so the deck can use several distinct
            # manual pages instead of repeatedly showing one architecture page.
            merged = []
            seen_keys = set()
            for item in list(primary_results) + list(visual_results):
                key = (item.get("source"), item.get("page"), item.get("chunk_id", item.get("text", ""))[:80])
                if key not in seen_keys:
                    seen_keys.add(key)
                    merged.append(item)

            context, relevant = retrieve_context(
                merged,
                min_score=0.25,
                max_chunks=10,
                max_chars=9000,
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

    # Source gate: manual evidence is preferred. If no manual evidence exists,
    # a reliable web result is required before any downloadable technical output
    # is created. This prevents unsupported model-only technical content.
    if not context.strip() and not web_sources:
        st.error(
            "No usable source evidence was found. Upload/build the manual index "
            "or configure TAVILY_API_KEY so MarineWise can research reliable "
            "technical sources before generating PDF/PPT/Word output."
        )
        return

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
        if generate_training_presentation_plan is not None:
            plan_raw = generate_training_presentation_plan(
                selected_provider(),
                engine,
                ship,
                topic,
                context,
                research,
            )
            plan = parse_training_plan(plan_raw)
        else:
            # Safe local fallback if the optional planner function is unavailable.
            plan = [
                {
                    "title": f"{topic} — System Overview",
                    "purpose": f"Explain the requested {topic} system using source evidence.",
                    "bullets": [
                        f"Purpose and boundaries of {topic}",
                        f"Major {topic} components supported by the source",
                        "Relevant interfaces and flow path",
                    ],
                    "visual_type": "manual_page",
                    "visual_query": f"{engine} {topic} system diagram schematic",
                    "source_preference": "manual_first",
                },
                {
                    "title": f"How {topic} Works",
                    "purpose": "Explain the topic-specific operating sequence.",
                    "bullets": [
                        "Initiating condition / input",
                        "Main process",
                        "Control or regulation",
                        "Feedback / output",
                    ],
                    "visual_type": "process_diagram",
                    "visual_query": f"{engine} {topic} operating sequence flow diagram",
                    "source_preference": "manual_first",
                },
                {
                    "title": f"{topic} Components",
                    "purpose": "Identify and explain the relevant components.",
                    "bullets": [
                        "Function",
                        "Operating role",
                        "Inspection points",
                    ],
                    "visual_type": "component_cards",
                    "visual_query": f"{engine} {topic} components technical diagram",
                    "source_preference": "manual_first",
                },
                {
                    "title": "Technician Checks",
                    "purpose": "Provide evidence-supported inspection checks.",
                    "bullets": [
                        "Confirm the symptom or condition",
                        "Inspect relevant components and connections",
                        "Verify approved measurements",
                        "Record findings and source page",
                    ],
                    "visual_type": "checklist",
                    "visual_query": f"{engine} {topic} inspection manual",
                    "source_preference": "manual_first",
                },
                {
                    "title": f"{topic} Fault Diagnosis",
                    "purpose": "Connect symptoms to evidence-supported verification.",
                    "bullets": [
                        "Symptom",
                        "Relevant area",
                        "Verification",
                        "Corrective action where documented",
                    ],
                    "visual_type": "diagnostic_flow",
                    "visual_query": f"{engine} {topic} troubleshooting",
                    "source_preference": "manual_first",
                },
                {
                    "title": f"{topic} Maintenance & Safety",
                    "purpose": "Cover only topic-relevant maintenance and safety information.",
                    "bullets": [
                        "Isolation / lockout where documented",
                        "PPE and hazards where documented",
                        "Maintenance requirements",
                        "Warnings and limits from the source",
                    ],
                    "visual_type": "safety_callout",
                    "visual_query": f"{engine} {topic} maintenance safety",
                    "source_preference": "manual_first",
                },
                {
                    "title": f"{topic} Knowledge Check",
                    "purpose": "Confirm topic-specific understanding.",
                    "bullets": [
                        f"What is the normal {topic} sequence?",
                        f"Which {topic} components require inspection?",
                        "Which observation confirms the suspected condition?",
                    ],
                    "visual_type": "questions",
                    "visual_query": "",
                    "source_preference": "manual_first",
                },
                {
                    "title": "References",
                    "purpose": "Show all evidence used.",
                    "bullets": [
                        "Supplied manufacturer manual pages",
                        "Reliable web sources used for gaps",
                    ],
                    "visual_type": "references",
                    "visual_query": "",
                    "source_preference": "all_sources",
                },
            ]

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
        plan,
    )

    st.markdown(
        "### Topic Visual / Architecture"
    )

    if manual_images:
        st.image(
            manual_images[0]["bytes"],
            caption=(
                f"Primary manual visual: {manual_images[0].get('source')} — "
                f"page {manual_images[0].get('page')}"
            ),
            use_container_width=True,
        )
    else:
        st.image(
            diagram,
            caption=(
                "Generated topic-specific training diagram based on the "
                "presentation plan and source evidence."
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
                relevant,
                web_sources,
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
    st.subheader("2B. Quiz Generator")
    st.caption("Generate technician questions and an answer key.")

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
            "Generate Quiz",
            type="primary",
        )

    if not submitted:
        return
    if not topic.strip():
        st.warning("Enter a topic.")
        return

    rag = st.session_state.get("rag")
    context = ""
    relevant: list[dict[str, Any]] = []

    if rag:
        with st.spinner("Searching manuals first..."):
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
                max_chars=6500,
            )

    # If the manual does not support the requested topic, use the same reliable
    # web research path used by 2A instead of silently falling back to model memory.
    research: dict[str, Any] = {"results": [], "images": []}
    tavily_key = get_secret("TAVILY_API_KEY")
    if tavily_key and (not context.strip() or len(context) < 600):
        with st.spinner("Manual evidence is limited — researching reliable technical sources online..."):
            research = run_training_web_research(
                selected_provider(),
                "",
                topic,
                "",
                context,
                tavily_key,
            ) if run_training_web_research is not None else research

    web_sources = research.get("results", []) or []

    if not context.strip() and not web_sources:
        st.error(
            "No usable manual evidence or reliable web source was found for this topic. "
            "Build the manual index or configure TAVILY_API_KEY before generating the quiz."
        )
        return

    with st.spinner("Generating quiz and answer key..."):
        quiz_text = run_agent(
            """
Create a marine technician training quiz.

Follow the requested question type and count exactly.
Include an ANSWER KEY at the end.

SOURCE PRIORITY:
1. Supplied manufacturer/manual excerpts are primary.
2. Reliable web evidence is secondary and may only fill gaps not covered by the manual.
3. Do not invent manufacturer-specific values, limits, procedures, or component details.
4. If a detail is not supported by the supplied evidence, phrase the question generically or omit it.

Make every question directly relevant to the requested topic.
For MCQ questions, use exactly one correct answer and plausible distractors.
""",
            (
                f"Topic: {topic}\n"
                f"Type: {qtype}\n"
                f"Questions: {count}\n\n"
                f"PRIMARY MANUAL EXCERPTS:\n{context}\n\n"
                f"SECONDARY WEB SOURCES:\n{web_sources[:8]}"
            ),
            "training",
        )

    st.markdown("### Quiz")
    st.write(quiz_text)

    st.download_button(
        "Download Quiz PDF",
        make_quiz_pdf(
            topic,
            qtype,
            quiz_text,
            relevant,
            web_sources,
        ),
        "marinewise_quiz.pdf",
        "application/pdf",
    )

    render_sources(relevant)
    if web_sources:
        st.markdown("**Secondary web research sources:**")
        for source in web_sources[:8]:
            title = safe_text(source.get("title"))
            url = safe_text(source.get("url"))
            if title and url:
                st.markdown(f"- {title}: {url}")


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

        diagram = make_training_diagram(
            engine.strip()
            or "Marine engine",
            remedial_topic,
            remedial_plan,
        )

        st.markdown(
            "### Targeted Training Diagram"
        )

        st.image(
            diagram,
            caption=(
                "Topic-specific retraining visual based on the identified "
                "improvement areas and source-grounded training plan."
            ),
            use_container_width=True,
        )

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
                relevant,
                web_sources,
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
# MARINE AI COMMAND CENTER
# ============================================================

def _command_center_web_text(
    research: dict[str, Any],
) -> str:
    """
    Convert approved Tavily results into bounded text
    for the CrewAI agents.
    """

    parts = []

    for number, item in enumerate(
        research.get("results", [])[:10],
        start=1,
    ):

        title = safe_text(
            item.get("title")
        )

        url = safe_text(
            item.get("url")
        )

        content = safe_text(
            item.get("content")
        )

        parts.append(
            f"WEB SOURCE {number}\n"
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Content: {content}"
        )

    return "\n\n".join(parts)


def marine_ai_command_center_page() -> None:

    st.markdown(
        '<div class="main-title">'
        "Marine AI Command Center"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="main-subtitle">'
        "CrewAI-powered collaboration between specialist marine agents"
        "</div>",
        unsafe_allow_html=True,
    )

    st.info(
        "The Marine AI Command Center is the multi-agent "
        "orchestration layer of MarineWise. It searches the "
        "supplied manuals first and then passes evidence between "
        "specialist CrewAI agents. Online research is never started "
        "automatically."
    )

    # ========================================================
    # AGENT FLOW
    # ========================================================

    st.markdown(
        "### CrewAI Agent Workflow"
    )

    flow_cols = st.columns(5)

    flow = [
        ("1", "Orchestrator"),
        ("2", "Evidence Analyst"),
        ("3", "Web Reviewer"),
        ("4", "Marine Specialist"),
        ("5", "Learning Agent"),
    ]

    for col, (
        number,
        label,
    ) in zip(
        flow_cols,
        flow,
    ):

        col.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{number}</div>
                <div class="metric-label">{label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div style="
            text-align:center;
            margin:15px 0;
            font-size:18px;
            font-weight:600;
        ">
            Orchestrator
            →
            Evidence Analyst
            →
            Marine Specialist
            →
            Learning Agent
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "If approved web evidence is available, the Web Evidence "
        "Reviewer is inserted between the Evidence Analyst and "
        "Marine Specialist."
    )

    # ========================================================
    # INPUT
    # ========================================================

    st.markdown(
        "### 1. Define the Technical Task"
    )

    c1, c2 = st.columns(2)

    with c1:

        topic = st.text_input(
            "Topic *",
            placeholder=(
                "e.g. Bow thruster hydraulic system"
            ),
            key="cc_topic",
        )

        engine_model = st.text_input(
            "Engine / Equipment Model",
            placeholder="Optional",
            key="cc_engine",
        )

    with c2:

        ship = st.text_input(
            "Ship",
            placeholder="Optional",
            key="cc_ship",
        )

        objective = st.text_area(
            "Command Center Objective",
            placeholder=(
                "What should the agents investigate, explain, "
                "troubleshoot, or turn into a learning plan?"
            ),
            height=110,
            key="cc_objective",
        )

    # ========================================================
    # RAG STATUS
    # ========================================================

    rag = st.session_state.get(
        "rag"
    )

    if rag:

        st.success(
            f"Manual-first RAG is ready: "
            f"{len(rag['records'])} chunks and "
            f"{len(rag.get('page_images', {}))} page visuals."
        )

    else:

        st.warning(
            "No FAISS manual index is available. "
            "Build the manual index first for "
            "source-grounded Command Center results."
        )

    # ========================================================
    # RUN BUTTON
    # ========================================================

    run_button = st.button(
        "Run Marine AI Command Center",
        type="primary",
        use_container_width=True,
        key="cc_run",
    )

    if run_button:

        if not topic.strip():

            st.warning(
                "Enter a topic first."
            )

            return

        context = ""
        relevant = []

        # ----------------------------------------------------
        # MANUAL-FIRST RAG
        # ----------------------------------------------------

        if rag:

            query_parts = [
                topic.strip()
            ]

            if engine_model.strip():
                query_parts.append(
                    engine_model.strip()
                )

            if ship.strip():
                query_parts.append(
                    ship.strip()
                )

            query = " ".join(
                query_parts
            )

            with st.spinner(
                "Step 1 — searching the supplied manuals..."
            ):

                results = search_index(
                    rag,
                    query,
                    get_embedder(),
                    k=8,
                )

                context, relevant = (
                    retrieve_context(
                        results,
                        min_score=0.28,
                        max_chunks=6,
                        max_chars=7000,
                    )
                )

            st.session_state.last_retrieved = (
                relevant
            )

        # ----------------------------------------------------
        # CREWAI
        # ----------------------------------------------------

        try:

            with st.spinner(
                "Step 2 — CrewAI agents are collaborating..."
            ):

                result = run_marine_command_center(
                    provider=selected_provider(),
                    topic=topic,
                    engine_model=engine_model,
                    ship=ship,
                    objective=objective,
                    manual_context=context,
                    web_context="",
                    api_key=require_provider_key(),
                )

            st.session_state.command_center_result = (
                result
            )

            st.session_state.command_center_context = (
                context
            )

            st.session_state.command_center_sources = (
                relevant
            )

            st.session_state.command_center_web = (
                None
            )

            st.session_state.command_center_case = {
                "topic": topic,
                "engine_model": engine_model,
                "ship": ship,
                "objective": objective,
            }

        except Exception as exc:

            st.error(
                f"Command Center failed: {exc}"
            )

            if (
                "CrewAI is not installed"
                in str(exc)
            ):

                st.code(
                    "pip install -U crewai",
                    language="bash",
                )

            return

    # ========================================================
    # GET RESULT
    # ========================================================

    result = st.session_state.get(
        "command_center_result"
    )

    context = st.session_state.get(
        "command_center_context",
        "",
    )

    sources = st.session_state.get(
        "command_center_sources",
        [],
    )

    web_research = st.session_state.get(
        "command_center_web"
    )

    case = st.session_state.get(
        "command_center_case",
        {},
    )

    if not result:
        return

    # ========================================================
    # RESULT
    # ========================================================

    st.markdown("---")

    st.markdown(
        "### CrewAI Collaboration Result"
    )

    agents_used = result.get(
        "agent_names",
        [],
    )

    if agents_used:

        st.caption(
            "Agents used: "
            + " → ".join(
                agents_used
            )
        )

    result_tabs = st.tabs(
        [
            "Final Marine Insight",
            "Agent-to-Agent Handoffs",
            "Manual Evidence",
        ]
    )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    with result_tabs[0]:

        st.markdown(
            result.get(
                "final",
                "No final result was returned.",
            )
        )

        if result.get(
            "used_web_evidence"
        ):

            st.caption(
                "Approved supplementary web evidence "
                "was included in this run."
            )

        else:

            st.caption(
                "No online research was used in this run."
            )

    # ========================================================
    # AGENT HANDOFFS
    # ========================================================

    with result_tabs[1]:

        task_outputs = result.get(
            "task_outputs",
            [],
        )

        if not task_outputs:

            st.info(
                "No individual task outputs were returned."
            )

        for index, item in enumerate(
            task_outputs,
            start=1,
        ):

            task_name = item.get(
                "task",
                "CrewAI Task",
            )

            output = item.get(
                "output",
                "",
            )

            with st.expander(
                f"Agent handoff {index}: {task_name}",
                expanded=(
                    index
                    == len(task_outputs)
                ),
            ):

                st.write(
                    output
                )

    # ========================================================
    # MANUAL EVIDENCE
    # ========================================================

    with result_tabs[2]:

        if sources:

            for source in sources:

                source_name = source.get(
                    "source",
                    "Manual",
                )

                page = source.get(
                    "page",
                    "?",
                )

                st.markdown(
                    f"**{source_name} — page {page}**"
                )

                st.write(
                    source.get(
                        "text",
                        "",
                    )
                )

        elif context:

            st.write(
                context
            )

        else:

            st.warning(
                "No manual evidence was retrieved."
            )

    # ========================================================
    # HUMAN-IN-THE-LOOP WEB FALLBACK
    # ========================================================

    if not web_research:

        st.markdown("---")

        st.markdown(
            "### Need Supplementary Online Research?"
        )

        st.write(
            "The current CrewAI result is based on the "
            "supplied manual evidence. If the agents "
            "identified an evidence gap, you can explicitly "
            "approve Tavily research here."
        )

        tavily_key = get_secret(
            "TAVILY_API_KEY"
        )

        if not tavily_key:

            st.caption(
                "TAVILY_API_KEY is not configured, "
                "so online research is unavailable."
            )

        elif st.button(
            "Yes — Search Online with Tavily",
            key="cc_approve_web",
        ):

            try:

                with st.spinner(
                    "Searching approved supplementary sources..."
                ):

                    research = (
                        run_training_web_research(
                            provider=selected_provider(),
                            engine=case.get(
                                "engine_model",
                                "",
                            ),
                            topic=case.get(
                                "topic",
                                "",
                            ),
                            ship=case.get(
                                "ship",
                                "",
                            ),
                            manual_context=context,
                            tavily_api_key=tavily_key,
                        )
                    )

                web_text = (
                    _command_center_web_text(
                        research
                    )
                )

                st.session_state.command_center_web = (
                    research
                )

                if not web_text:

                    st.warning(
                        "No web evidence was returned. "
                        "The manual-only result remains available."
                    )

                else:

                    with st.spinner(
                        "Passing approved web evidence "
                        "to the CrewAI team..."
                    ):

                        updated = (
                            run_marine_command_center(
                                provider=selected_provider(),
                                topic=case.get(
                                    "topic",
                                    "",
                                ),
                                engine_model=case.get(
                                    "engine_model",
                                    "",
                                ),
                                ship=case.get(
                                    "ship",
                                    "",
                                ),
                                objective=case.get(
                                    "objective",
                                    "",
                                ),
                                manual_context=context,
                                web_context=web_text,
                                api_key=require_provider_key(),
                            )
                        )

                    st.session_state.command_center_result = (
                        updated
                    )

                    st.rerun()

            except Exception as exc:

                st.error(
                    f"Approved web research failed: {exc}"
                )

    # ========================================================
    # APPROVED WEB SOURCES
    # ========================================================

    if web_research:

        st.markdown(
            "### Approved Supplementary Web Evidence"
        )

        for item in web_research.get(
            "results",
            [],
        )[:8]:

            title = (
                safe_text(
                    item.get(
                        "title"
                    )
                )
                or "Web source"
            )

            url = safe_text(
                item.get(
                    "url"
                )
            )

            content = safe_text(
                item.get(
                    "content"
                )
            )

            with st.expander(
                title
            ):

                if url:
                    st.markdown(
                        url
                    )

                st.write(
                    content
                )

    # ========================================================
    # CLEAR RESULT
    # ========================================================

    if st.button(
        "Clear Command Center Result",
        key="cc_clear",
    ):

        for key in (
            "command_center_result",
            "command_center_context",
            "command_center_sources",
            "command_center_web",
            "command_center_case",
        ):

            st.session_state.pop(
                key,
                None,
            )

        st.rerun()
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
    """Create a clean, justified A4 PDF using Times New Roman when available."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        Image as RLImage,
        KeepTogether,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
    )

    font_name = _register_reportlab_fonts()
    output = io.BytesIO()
    document = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=clean_output_text(f"MarineWise AI — {topic}"),
        author="MarineWise AI",
    )

    title_style = ParagraphStyle(
        "MarineTitle",
        fontName=font_name,
        fontSize=22,
        leading=27,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#17324D"),
        spaceAfter=8,
    )
    meta_style = ParagraphStyle(
        "MarineMeta",
        fontName=font_name,
        fontSize=14,
        leading=20,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#344054"),
        spaceAfter=12,
    )
    body_style = ParagraphStyle(
        "MarineBody",
        fontName=font_name,
        fontSize=14,
        leading=20,
        alignment=TA_JUSTIFY,
        textColor=colors.HexColor("#17202A"),
        spaceAfter=8,
        allowWidows=0,
        allowOrphans=0,
    )
    heading_style = ParagraphStyle(
        "MarineHeading",
        fontName=font_name,
        fontSize=17,
        leading=21,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#17324D"),
        spaceBefore=10,
        spaceAfter=7,
    )
    source_style = ParagraphStyle(
        "MarineSource",
        fontName=font_name,
        fontSize=11,
        leading=16,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#475467"),
        spaceAfter=5,
    )

    story = [
        Paragraph("MarineWise AI — Technical Training", title_style),
        Paragraph(
            safe_paragraph(
                clean_output_text(
                    f"Engine: {engine} | Ship: {ship or 'Not specified'} | Topic: {topic}"
                )
            ),
            meta_style,
        ),
    ]

    if diagram:
        try:
            img = RLImage(io.BytesIO(diagram), width=170 * mm, height=74 * mm)
            story.extend([img, Spacer(1, 8)])
        except Exception:
            pass

    for part in split_text(clean_output_text(content), 1500):
        story.append(Paragraph(safe_paragraph(part), body_style))

    source_lines = _source_lines(sources, web_sources)
    if source_lines:
        story.append(PageBreak())
        story.append(Paragraph("Sources & References", heading_style))
        story.append(
            Paragraph(
                safe_paragraph(
                    "Primary source: supplied manufacturer manuals. "
                    "Secondary source: reliable technical web research where the manual did not provide the required information."
                ),
                body_style,
            )
        )
        for line in source_lines:
            story.append(Paragraph(safe_paragraph(line), source_style))

    document.build(story)
    return output.getvalue()


def make_training_docx(
    engine: str,
    ship: str,
    topic: str,
    content: str,
    diagram: bytes,
    sources: list[dict[str, Any]] | None = None,
    web_sources: list[dict[str, Any]] | None = None,
) -> bytes:
    """Create a professional Word document with explicit 14pt Times New Roman body text."""
    from docx import Document
    from docx.enum.section import WD_SECTION
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Inches, Pt

    document = Document()
    section = document.sections[0]
    section.top_margin = Inches(0.7)
    section.bottom_margin = Inches(0.7)
    section.left_margin = Inches(0.8)
    section.right_margin = Inches(0.8)

    normal = document.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(14)
    try:
        normal._element.rPr.rFonts.set(normal._element.rPr.rFonts.get_or_add_ascii(), "Times New Roman")
        normal._element.rPr.rFonts.set(normal._element.rPr.rFonts.get_or_add_hAnsi(), "Times New Roman")
    except Exception:
        pass

    title = document.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run(clean_output_text("MarineWise AI — Technical Training"))
    _set_docx_run_font(run, "Times New Roman", 20, True)

    _add_docx_paragraph(
        document,
        f"Engine: {engine} | Ship: {ship or 'Not specified'} | Topic: {topic}",
        size=14,
        bold=False,
        align=1,
    )

    if diagram:
        try:
            p = document.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.add_run().add_picture(io.BytesIO(diagram), width=Inches(6.4))
        except Exception:
            pass

    for part in split_text(clean_output_text(content), 1500):
        _add_docx_paragraph(document, part, size=14, bold=False, align=3)

    source_lines = _source_lines(sources, web_sources)
    if source_lines:
        document.add_page_break()
        heading = document.add_paragraph()
        heading.alignment = WD_ALIGN_PARAGRAPH.LEFT
        run = heading.add_run("Sources & References")
        _set_docx_run_font(run, "Times New Roman", 17, True)
        _add_docx_paragraph(
            document,
            "Primary source: supplied manufacturer manuals. Secondary source: reliable technical web research where required.",
            size=14,
            align=3,
        )
        for line in source_lines:
            _add_docx_paragraph(document, line, size=12, align=0)

    output = io.BytesIO()
    document.save(output)
    return output.getvalue()


def make_quiz_pdf(
    topic: str,
    qtype: str,
    text: str,
    sources: list[dict[str, Any]] | None = None,
    web_sources: list[dict[str, Any]] | None = None,
) -> bytes:
    """Create a clean quiz PDF with 14pt justified Times New Roman body text."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

    font_name = _register_reportlab_fonts()
    output = io.BytesIO()
    document = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=clean_output_text(f"MarineWise AI — {topic} Quiz"),
        author="MarineWise AI",
    )

    title_style = ParagraphStyle(
        "QuizTitle", fontName=font_name, fontSize=21, leading=26,
        alignment=TA_CENTER, textColor=colors.HexColor("#17324D"), spaceAfter=8,
    )
    meta_style = ParagraphStyle(
        "QuizMeta", fontName=font_name, fontSize=14, leading=20,
        alignment=TA_CENTER, textColor=colors.HexColor("#344054"), spaceAfter=12,
    )
    body_style = ParagraphStyle(
        "QuizBody", fontName=font_name, fontSize=14, leading=20,
        alignment=TA_JUSTIFY, textColor=colors.HexColor("#17202A"), spaceAfter=9,
    )
    source_style = ParagraphStyle(
        "QuizSource", fontName=font_name, fontSize=11, leading=16,
        alignment=TA_LEFT, textColor=colors.HexColor("#475467"), spaceAfter=5,
    )

    story = [
        Paragraph("MarineWise AI — Technical Assessment / Quiz", title_style),
        Paragraph(
            safe_paragraph(clean_output_text(f"Topic: {topic} | Type: {qtype}")),
            meta_style,
        ),
        Spacer(1, 6),
    ]

    quiz_text_clean = clean_output_text(text)
    answer_match = re.search(r"(?is)\bANSWER\s+KEY\b", quiz_text_clean)
    question_text = quiz_text_clean
    answer_text = ""
    if answer_match:
        question_text = quiz_text_clean[:answer_match.start()].strip()
        answer_text = quiz_text_clean[answer_match.end():].strip()

    # Keep questions separate from the answer key so the printed assessment is
    # clean and the key never appears halfway through a question block.
    for part in split_text(question_text, 1200):
        story.append(Paragraph(safe_paragraph(part), body_style))

    story.append(PageBreak())
    story.append(Paragraph("Answer Key", title_style))
    if answer_text:
        for part in split_text(answer_text, 1200):
            story.append(Paragraph(safe_paragraph(part), body_style))
    else:
        story.append(
            Paragraph(
                safe_paragraph("No explicit answer key marker was returned by the quiz generator."),
                body_style,
            )
        )

    source_lines = _source_lines(sources, web_sources)
    if source_lines:
        story.append(Spacer(1, 10))
        story.append(Paragraph("Source Verification", title_style))
        for line in source_lines:
            story.append(Paragraph(safe_paragraph(line), source_style))

    document.build(story)
    return output.getvalue()


def make_remedial_ppt(
    text: str,
) -> bytes:
    """Legacy remedial PPT helper retained for compatibility, with safe formatting."""
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.util import Inches, Pt

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    parts = split_text(clean_output_text(text), 900)
    for index, part in enumerate(parts[:8], start=1):
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        add_slide_background(slide)
        add_top_bar(slide, f"Remedial Training — Part {index}", "Remedial")
        _add_ppt_text_box(
            slide, part, 0.75, 1.35, 11.8, 5.2,
            font_size=16, color=DARK, align=3
        )
        add_footer(slide, index, "MarineWise AI — Remedial Training")

    output = io.BytesIO()
    prs.save(output)
    return output.getvalue()


# ============================================================
# MAIN APPLICATION
# ============================================================


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

        # ----------------------------------------------------
        # PAGE 1 — TROUBLESHOOTING AGENT
        # ----------------------------------------------------

        if page == "Troubleshooting Agent":

            troubleshooting_page()

        # ----------------------------------------------------
        # PAGE 2 — TRAINING AGENT
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # PAGE 3 — MARINE AI COMMAND CENTER
        # ----------------------------------------------------

        elif page == "Marine AI Command Center":

            marine_ai_command_center_page()

        # ----------------------------------------------------
        # PAGE 4 — LEARNING
        # ----------------------------------------------------

        elif page == "Learning":

            learning_page()

    except Exception as exc:

        import traceback

        message = str(exc)

        st.error(
            f"MarineWise AI encountered an error: {exc}"
        )

        with st.expander(
            "Show technical error details"
        ):

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
