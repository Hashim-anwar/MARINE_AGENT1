"""
MarineWise AI - AI agents and web research.

This file handles:
- Groq AI
- Gemini AI
- Tavily web search
- Troubleshooting agent
- Technical training agent
- Training web research
- Professional PPT slide planning

No CrewAI or LiteLLM is used.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

import requests


# ============================================================
# CONFIGURATION
# ============================================================

GROQ_MODEL = "openai/gpt-oss-120b"

# Gemini model can be changed later without changing the app.
GEMINI_MODEL = "gemini-3.8-flash"

TAVILY_URL = "https://api.tavily.com/search"


# Troubleshooting is intentionally kept small because
# troubleshooting questions often contain large manual excerpts.
TROUBLESHOOTING_MAX_PROMPT_CHARS = 6500
TROUBLESHOOTING_MAX_COMPLETION = 700

# Training needs more context than troubleshooting.
TRAINING_MAX_PROMPT_CHARS = 18000
TRAINING_MAX_COMPLETION = 1400

# Web-search answer generation.
WEB_MAX_PROMPT_CHARS = 7000
WEB_MAX_COMPLETION = 900

# Presentation planning needs enough room to create
# a complete structured slide plan.
PRESENTATION_PLAN_MAX_PROMPT_CHARS = 22000
PRESENTATION_PLAN_MAX_COMPLETION = 3000


# ============================================================
# SIMPLE AGENT DEFINITIONS
# ============================================================

@dataclass
class MarineAgent:
    """Simple description of a MarineWise AI agent."""

    name: str
    role: str
    goal: str


TROUBLESHOOTING_AGENT = MarineAgent(
    name="Marine Troubleshooting Agent",
    role="Marine engine troubleshooting technician assistant",
    goal=(
        "Help technicians diagnose marine engine alarms and defects using "
        "the supplied manuals first and clearly identify the source."
    ),
)

TRAINING_AGENT = MarineAgent(
    name="Marine Technical Training Agent",
    role="Marine engine technical instructor",
    goal=(
        "Create accurate, practical and visually understandable technical "
        "training material for marine engineers and technicians."
    ),
)


def make_troubleshooting_agent() -> MarineAgent:
    """Return the troubleshooting agent definition."""
    return TROUBLESHOOTING_AGENT


def make_training_agent() -> MarineAgent:
    """Return the training agent definition."""
    return TRAINING_AGENT


# ============================================================
# API KEY HELPERS
# ============================================================

def _get_key(provider: str, api_key: str | None = None) -> str:
    """
    Return an API key.

    The Streamlit app normally passes the key from st.secrets.
    Environment variables are also supported.
    """

    if api_key:
        return api_key.strip()

    provider = provider.strip().lower()

    if provider == "groq":
        key = os.getenv("GROQ_API_KEY")
    elif provider == "gemini":
        key = os.getenv("GEMINI_API_KEY")
    else:
        key = None

    if not key:
        raise RuntimeError(
            f"No API key was supplied for {provider.title()}. "
            f"Configure the appropriate API key in Streamlit Secrets."
        )

    return key.strip()


def _get_tavily_key(api_key: str | None = None) -> str:
    """Get Tavily API key."""

    if api_key:
        return api_key.strip()

    key = os.getenv("TAVILY_API_KEY")

    if not key:
        raise RuntimeError(
            "TAVILY_API_KEY is not configured. "
            "Add it to Streamlit Secrets."
        )

    return key.strip()


# ============================================================
# TEXT / CONTEXT HELPERS
# ============================================================

def _bounded(text: str | None, limit: int) -> str:
    """
    Keep prompts below provider context limits.

    This is especially important for retrieved PDF chunks.
    """

    if not text:
        return ""

    text = str(text)

    if len(text) <= limit:
        return text

    return text[:limit].rstrip() + "\n\n[Context truncated for safety.]"


def _limits(agent_kind: str) -> tuple[int, int]:
    """Return prompt and completion limits."""

    kind = (agent_kind or "").strip().lower()

    if kind == "troubleshooting":
        return (
            TROUBLESHOOTING_MAX_PROMPT_CHARS,
            TROUBLESHOOTING_MAX_COMPLETION,
        )

    if kind == "training":
        return (
            TRAINING_MAX_PROMPT_CHARS,
            TRAINING_MAX_COMPLETION,
        )

    if kind == "presentation":
        return (
            PRESENTATION_PLAN_MAX_PROMPT_CHARS,
            PRESENTATION_PLAN_MAX_COMPLETION,
        )

    if kind == "web":
        return (
            WEB_MAX_PROMPT_CHARS,
            WEB_MAX_COMPLETION,
        )

    return (
        TRAINING_MAX_PROMPT_CHARS,
        TRAINING_MAX_COMPLETION,
    )


def _clean_text(text: str) -> str:
    """Clean excessive whitespace."""

    return re.sub(r"\n{3,}", "\n\n", text or "").strip()


# ============================================================
# GROQ
# ============================================================

def _run_groq(
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    max_tokens: int,
) -> str:
    """
    Run Groq directly.

    Important:
    We intentionally do NOT use CrewAI or LiteLLM.
    """

    from groq import Groq

    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0.2,
        reasoning_effort="low",
        max_completion_tokens=max_tokens,
    )

    if not response.choices:
        raise RuntimeError("Groq returned no response.")

    message = response.choices[0].message

    content = getattr(message, "content", None)

    if not content:
        raise RuntimeError("Groq returned an empty response.")

    return str(content).strip()


# ============================================================
# GEMINI
# ============================================================

def _run_gemini(
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    max_tokens: int,
) -> str:
    """Run Gemini using the official google-genai SDK."""

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    combined_prompt = (
        "SYSTEM INSTRUCTIONS:\n"
        f"{system_prompt}\n\n"
        "USER REQUEST:\n"
        f"{user_prompt}"
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=combined_prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=max_tokens,
        ),
    )

    text = getattr(response, "text", None)

    if not text:
        raise RuntimeError("Gemini returned an empty response.")

    return str(text).strip()


# ============================================================
# GENERIC MARINE AGENT
# ============================================================

def run_marine_agent(
    provider: str,
    agent_kind: str,
    system_prompt: str,
    user_prompt: str,
    api_key: str,
) -> str:
    """
    Run a MarineWise AI agent using Groq or Gemini.

    Parameters match the current app.py.
    """

    provider = (provider or "Groq").strip().lower()

    prompt_limit, completion_limit = _limits(agent_kind)

    safe_system = _bounded(system_prompt, 5000)

    remaining = max(1000, prompt_limit - len(safe_system))

    safe_user = _bounded(user_prompt, remaining)

    if provider == "groq":
        return _run_groq(
            safe_system,
            safe_user,
            _get_key("groq", api_key),
            completion_limit,
        )

    if provider == "gemini":
        return _run_gemini(
            safe_system,
            safe_user,
            _get_key("gemini", api_key),
            completion_limit,
        )

    raise ValueError(
        f"Unsupported AI provider: {provider}. "
        "Choose Groq or Gemini."
    )


# ============================================================
# TAVILY SEARCH
# ============================================================

def _tavily_search(
    query: str,
    api_key: str,
    max_results: int = 5,
    include_images: bool = False,
) -> dict[str, Any]:
    """
    Search Tavily.

    Returns:
        {
            "results": [...],
            "images": [...]
        }
    """

    query = _clean_text(query)

    if not query:
        return {
            "results": [],
            "images": [],
        }

    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "advanced",
        "max_results": max_results,
        "include_answer": False,
        "include_raw_content": False,
        "include_images": include_images,
    }

    response = requests.post(
        TAVILY_URL,
        json=payload,
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    results = data.get("results") or []
    images = data.get("images") or []

    clean_results: list[dict[str, Any]] = []

    for item in results:
        if not isinstance(item, dict):
            continue

        title = str(item.get("title") or "").strip()
        url = str(item.get("url") or "").strip()
        content = str(
            item.get("content")
            or item.get("snippet")
            or ""
        ).strip()

        if not title and not url and not content:
            continue

        clean_results.append(
            {
                "title": title,
                "url": url,
                "content": _bounded(content, 1800),
            }
        )

    clean_images: list[str] = []

    for image in images:
        if isinstance(image, str):
            image_url = image.strip()

            if image_url:
                clean_images.append(image_url)

        elif isinstance(image, dict):
            image_url = str(
                image.get("url")
                or image.get("image_url")
                or ""
            ).strip()

            if image_url:
                clean_images.append(image_url)

    # Remove duplicate images while keeping order.
    clean_images = list(dict.fromkeys(clean_images))

    return {
        "results": clean_results,
        "images": clean_images,
    }


# ============================================================
# GENERAL WEB SEARCH
# ============================================================

def run_web_search(
    provider: str,
    system_prompt: str,
    user_prompt: str,
    api_key: str,
) -> str:
    """
    Search Tavily and then summarize the findings with Groq/Gemini.

    This function is used by the Troubleshooting page when the
    technician explicitly asks to search online.
    """

    tavily_key = _get_tavily_key()

    research = _tavily_search(
        user_prompt,
        tavily_key,
        max_results=5,
        include_images=False,
    )

    results = research.get("results", [])

    if not results:
        return (
            "No reliable web search results were returned. "
            "Please verify the engine model and search again."
        )

    evidence_parts: list[str] = []

    for i, result in enumerate(results, start=1):
        evidence_parts.append(
            f"SOURCE {i}\n"
            f"Title: {result.get('title', '')}\n"
            f"URL: {result.get('url', '')}\n"
            f"Content: {result.get('content', '')}"
        )

    evidence = "\n\n".join(evidence_parts)

    combined_user_prompt = (
        f"{user_prompt}\n\n"
        "WEB SEARCH EVIDENCE:\n"
        f"{_bounded(evidence, WEB_MAX_PROMPT_CHARS)}\n\n"
        "Answer using the evidence above. "
        "Do not invent specifications. "
        "Clearly distinguish information found online from information "
        "contained in the supplied manuals. "
        "Include source URLs where useful."
    )

    return run_marine_agent(
        provider,
        "web",
        system_prompt,
        combined_user_prompt,
        api_key,
    )


# ============================================================
# TRAINING WEB RESEARCH
# ============================================================

def run_training_web_research(
    provider: str,
    engine: str,
    topic: str,
    ship: str,
    manual_context: str = "",
    tavily_api_key: str | None = None,
) -> dict[str, Any]:
    """
    Perform deeper online research for professional training material.

    The manual remains the primary source.

    Tavily is used to find:
    - technical explanations
    - diagrams
    - component information
    - maintenance information
    - safety information
    - useful images

    Returns a research bundle containing:
        results
        images
        queries
    """

    del provider  # Provider is retained for API compatibility.

    engine = _clean_text(engine)
    topic = _clean_text(topic)
    ship = _clean_text(ship)
    manual_context = _clean_text(manual_context)

    tavily_key = _get_tavily_key(tavily_api_key)

    queries: list[str] = []

    # --------------------------------------------------------
    # Query 1 - Main subject
    # --------------------------------------------------------

    if engine:
        queries.append(
            f"{engine} {topic} marine engine technical information"
        )
    else:
        queries.append(
            f"marine diesel engine {topic} technical information"
        )

    # --------------------------------------------------------
    # Query 2 - Diagram / system
    # --------------------------------------------------------

    if engine:
        queries.append(
            f"{engine} {topic} system diagram schematic"
        )
    else:
        queries.append(
            f"marine diesel {topic} system diagram schematic"
        )

    # --------------------------------------------------------
    # Query 3 - Training material
    # --------------------------------------------------------

    queries.append(
        f"marine diesel engine {topic} training components operation"
    )

    # --------------------------------------------------------
    # Query 4 - Maintenance / safety
    # --------------------------------------------------------

    queries.append(
        f"marine diesel {topic} maintenance troubleshooting safety"
    )

    # Add ship context when useful.
    if ship:
        queries.append(
            f"{ship} marine engine {topic} technical"
        )

    # Avoid excessive API calls.
    queries = list(dict.fromkeys(queries))[:4]

    all_results: list[dict[str, Any]] = []
    all_images: list[str] = []

    seen_urls: set[str] = set()
    seen_images: set[str] = set()

    for query in queries:
        try:
            research = _tavily_search(
                query,
                tavily_key,
                max_results=5,
                include_images=True,
            )
        except Exception:
            # One failed query should not destroy the entire
            # training-generation workflow.
            continue

        for result in research.get("results", []):
            url = str(result.get("url") or "").strip()

            # Deduplicate URLs.
            if url and url in seen_urls:
                continue

            if url:
                seen_urls.add(url)

            all_results.append(
                {
                    "title": str(result.get("title") or ""),
                    "url": url,
                    "content": _bounded(
                        str(result.get("content") or ""),
                        1600,
                    ),
                }
            )

        for image_url in research.get("images", []):
            image_url = str(image_url).strip()

            if not image_url:
                continue

            if image_url in seen_images:
                continue

            seen_images.add(image_url)
            all_images.append(image_url)

    # Limit evidence so the presentation planner does not
    # receive an enormous prompt.
    all_results = all_results[:16]
    all_images = all_images[:12]

    # Include a compact note that manual material is primary.
    if manual_context:
        manual_note = (
            "Manual context was available and should be treated as the "
            "primary technical source. Online research is supplementary."
        )
    else:
        manual_note = (
            "No manual context was supplied. Online information should "
            "be verified against the engine manufacturer's current manual."
        )

    return {
        "engine": engine,
        "topic": topic,
        "ship": ship,
        "queries": queries,
        "results": all_results,
        "images": all_images,
        "manual_note": manual_note,
    }


# ============================================================
# JSON EXTRACTION
# ============================================================

def _extract_json(text: str) -> Any | None:
    """
    Extract JSON from an AI response.

    Handles:
    - plain JSON
    - ```json ... ```
    - JSON surrounded by explanatory text
    """

    if not text:
        return None

    cleaned = text.strip()

    # Remove markdown fences.
    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    # First attempt: whole response.
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # Second attempt: locate first JSON object.
    object_match = re.search(
        r"\{.*\}",
        cleaned,
        flags=re.DOTALL,
    )

    if object_match:
        try:
            return json.loads(object_match.group(0))
        except Exception:
            pass

    # Third attempt: locate first JSON array.
    array_match = re.search(
        r"\[.*\]",
        cleaned,
        flags=re.DOTALL,
    )

    if array_match:
        try:
            return json.loads(array_match.group(0))
        except Exception:
            pass

    return None


# ============================================================
# PRESENTATION PLAN FALLBACK
# ============================================================

def _fallback_presentation_plan(
    engine: str,
    ship: str,
    topic: str,
) -> dict[str, Any]:
    """
    Safe fallback if the AI does not return valid JSON.
    """

    return {
        "presentation_title": (
            f"{engine} — {topic}"
            if engine
            else f"Marine Engine Training — {topic}"
        ),
        "subtitle": ship or "Technical Training",
        "slides": [
            {
                "title": "Learning Objectives",
                "purpose": "Introduce the skills technicians should gain.",
                "bullets": [
                    f"Understand the {topic} system",
                    "Identify the main components",
                    "Explain the operating sequence",
                    "Recognize common faults and checks",
                ],
                "visual_type": "diagram",
                "visual_query": f"marine engine {topic} system diagram",
                "source_preference": "manual_first",
                "speaker_note": (
                    "Introduce the topic and explain why it matters "
                    "during normal operation and troubleshooting."
                ),
            },
            {
                "title": "System Overview",
                "purpose": "Show the overall system and major components.",
                "bullets": [
                    "Identify the major components",
                    "Follow the main flow path",
                    "Relate components to engine operation",
                ],
                "visual_type": "manual_page",
                "visual_query": f"{engine} {topic} diagram",
                "source_preference": "manual_first",
                "speaker_note": "Use the supplied manual where available.",
            },
            {
                "title": "How the System Works",
                "purpose": "Explain the operating sequence.",
                "bullets": [
                    "Input / supply",
                    "Main process",
                    "Control and monitoring",
                    "Return / output",
                ],
                "visual_type": "process_diagram",
                "visual_query": f"marine diesel {topic} flow diagram",
                "source_preference": "manual_first",
                "speaker_note": "Trace the system from beginning to end.",
            },
            {
                "title": "Component Spotlight",
                "purpose": "Explain important components.",
                "bullets": [
                    "Component function",
                    "Normal operating role",
                    "Important inspection points",
                ],
                "visual_type": "photo_or_diagram",
                "visual_query": f"marine engine {topic} components",
                "source_preference": "manual_or_web",
                "speaker_note": "Use labelled imagery where possible.",
            },
            {
                "title": "Technician Checks",
                "purpose": "Give technicians a practical inspection sequence.",
                "bullets": [
                    "Confirm the reported symptom",
                    "Check obvious external conditions",
                    "Verify relevant pressures / temperatures",
                    "Check sensors, connections and components",
                ],
                "visual_type": "checklist",
                "visual_query": "",
                "source_preference": "manual_first",
                "speaker_note": "Always follow the engine manufacturer's procedure.",
            },
            {
                "title": "Fault Diagnosis",
                "purpose": "Connect symptoms to verification steps.",
                "bullets": [
                    "Symptom",
                    "Possible system area",
                    "Verification",
                    "Corrective action",
                ],
                "visual_type": "diagnostic_flow",
                "visual_query": f"marine diesel {topic} troubleshooting",
                "source_preference": "manual_first",
                "speaker_note": "Avoid replacing components before verification.",
            },
            {
                "title": "Maintenance & Safety",
                "purpose": "Cover routine maintenance and hazards.",
                "bullets": [
                    "Follow manufacturer maintenance intervals",
                    "Use correct PPE",
                    "Isolate energy sources where required",
                    "Allow hot components to cool before work",
                ],
                "visual_type": "safety_callout",
                "visual_query": f"marine diesel engine {topic} maintenance safety",
                "source_preference": "manual_first",
                "speaker_note": "Use the engine manual's exact safety instructions.",
            },
            {
                "title": "Technician Knowledge Check",
                "purpose": "Reinforce the main learning points.",
                "bullets": [
                    "What is the normal flow path?",
                    "Which components require inspection?",
                    "What could cause the reported symptom?",
                ],
                "visual_type": "questions",
                "visual_query": "",
                "source_preference": "generated",
                "speaker_note": "Use these questions for a short classroom discussion.",
            },
            {
                "title": "References",
                "purpose": "Identify the technical sources used.",
                "bullets": [
                    "Supplied engine manuals",
                    "Manufacturer documentation",
                    "Reputable marine engineering references",
                ],
                "visual_type": "references",
                "visual_query": "",
                "source_preference": "all_sources",
                "speaker_note": "Verify technical details against the current manual.",
            },
        ],
    }


# ============================================================
# PROFESSIONAL TRAINING PRESENTATION PLANNER
# ============================================================

def generate_training_presentation_plan(
    provider: str,
    engine: str,
    ship: str,
    topic: str,
    manual_context: str,
    web_evidence: dict[str, Any],
) -> dict[str, Any]:
    """
    Ask the selected AI provider to design a professional
    technical training presentation.

    The output is structured JSON consumed by app.py.
    """

    engine = _clean_text(engine)
    ship = _clean_text(ship)
    topic = _clean_text(topic)
    manual_context = _clean_text(manual_context)

    web_evidence = web_evidence or {}

    results = web_evidence.get("results", [])
    images = web_evidence.get("images", [])

    # --------------------------------------------------------
    # Build compact web evidence.
    # --------------------------------------------------------

    web_parts: list[str] = []

    for i, result in enumerate(results[:16], start=1):
        web_parts.append(
            f"WEB SOURCE {i}\n"
            f"Title: {result.get('title', '')}\n"
            f"URL: {result.get('url', '')}\n"
            f"Evidence: {result.get('content', '')}"
        )

    web_text = "\n\n".join(web_parts)

    image_text = "\n".join(
        f"- {url}"
        for url in images[:12]
    )

    system_prompt = """
You are the MarineWise AI Professional Training Presentation Planner.

You create technically accurate presentation structures for marine
engineers and technicians.

The supplied engine manual material is the PRIMARY technical source.

Online research is SECONDARY and must never override the manufacturer
manual when the manual contains a specific instruction or specification.

Create an 8 to 10 slide professional training presentation.

The presentation must be:
- technically clear
- concise
- practical
- visually driven
- suitable for marine engineering technicians
- suitable for a professional PowerPoint deck

Do not create walls of text.

Each slide should have:
- title
- purpose
- 3 to 5 concise bullets maximum
- visual_type
- visual_query
- source_preference
- speaker_note

Possible visual_type values:
- title
- objectives
- manual_page
- photo_or_diagram
- process_diagram
- diagnostic_flow
- checklist
- safety_callout
- questions
- references
- component_cards

Use diagrams when a system process needs to be explained.

Use manual_page when the supplied manual has a relevant page or figure.

Use photo_or_diagram when an online technical image would help.

Use process_diagram for flow paths.

Use diagnostic_flow for troubleshooting logic.

Use safety_callout for hazards and precautions.

Return ONLY valid JSON.
No markdown.
No ``` fences.
""".strip()

    user_prompt = f"""
ENGINE:
{engine}

SHIP:
{ship or "Not specified"}

TRAINING TOPIC:
{topic}

MANUAL CONTEXT:
{_bounded(manual_context, 10000)}

ONLINE RESEARCH:
{_bounded(web_text, 8000)}

ONLINE IMAGE URLS:
{_bounded(image_text, 3500)}

Create the presentation plan.

Recommended structure:
1. Title
2. Learning objectives
3. System overview
4. How the system works
5. Component spotlight
6. System / process diagram
7. Fault diagnosis / technician checks
8. Maintenance and safety
9. Knowledge check
10. References

You may combine or adjust slides when technically appropriate,
but keep the final presentation between 8 and 10 slides.

JSON format:

{{
  "presentation_title": "...",
  "subtitle": "...",
  "slides": [
    {{
      "title": "...",
      "purpose": "...",
      "bullets": ["...", "...", "..."],
      "visual_type": "...",
      "visual_query": "...",
      "source_preference": "...",
      "speaker_note": "..."
    }}
  ]
}}
""".strip()

    # --------------------------------------------------------
    # Keep the planner prompt bounded.
    # --------------------------------------------------------

    user_prompt = _bounded(
        user_prompt,
        PRESENTATION_PLAN_MAX_PROMPT_CHARS,
    )

    try:
        raw = run_marine_agent(
            provider,
            "presentation",
            system_prompt,
            user_prompt,
            _get_key(
                provider,
                os.getenv(
                    "GROQ_API_KEY"
                    if provider.lower() == "groq"
                    else "GEMINI_API_KEY"
                ),
            ),
        )
    except Exception:
        return _fallback_presentation_plan(
            engine,
            ship,
            topic,
        )

    parsed = _extract_json(raw)

    if not isinstance(parsed, dict):
        return _fallback_presentation_plan(
            engine,
            ship,
            topic,
        )

    slides = parsed.get("slides")

    if not isinstance(slides, list) or not slides:
        return _fallback_presentation_plan(
            engine,
            ship,
            topic,
        )

    # --------------------------------------------------------
    # Normalize the AI-generated slide plan.
    # --------------------------------------------------------

    normalized_slides: list[dict[str, Any]] = []

    for slide in slides[:10]:
        if not isinstance(slide, dict):
            continue

        title = str(slide.get("title") or "").strip()

        if not title:
            continue

        bullets = slide.get("bullets", [])

        if not isinstance(bullets, list):
            bullets = [str(bullets)]

        clean_bullets = []

        for bullet in bullets[:5]:
            bullet = str(bullet).strip()

            if bullet:
                clean_bullets.append(bullet)

        normalized_slides.append(
            {
                "title": title,
                "purpose": str(
                    slide.get("purpose") or ""
                ).strip(),
                "bullets": clean_bullets,
                "visual_type": str(
                    slide.get("visual_type") or "photo_or_diagram"
                ).strip(),
                "visual_query": str(
                    slide.get("visual_query") or ""
                ).strip(),
                "source_preference": str(
                    slide.get("source_preference")
                    or "manual_first"
                ).strip(),
                "speaker_note": str(
                    slide.get("speaker_note")
                    or ""
                ).strip(),
            }
        )

    # If AI produced fewer than 8 usable slides,
    # use the reliable fallback structure.
    if len(normalized_slides) < 8:
        return _fallback_presentation_plan(
            engine,
            ship,
            topic,
        )

    return {
        "presentation_title": str(
            parsed.get("presentation_title")
            or f"{engine} — {topic}"
        ).strip(),
        "subtitle": str(
            parsed.get("subtitle")
            or ship
            or "Technical Training"
        ).strip(),
        "slides": normalized_slides,
    }
