"""
MarineWise AI - AI agents, Tavily research, and presentation planning.

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
GEMINI_MODEL = "gemini-3.8-flash"

TAVILY_URL = "https://api.tavily.com/search"

TROUBLESHOOTING_MAX_PROMPT_CHARS = 6500
TROUBLESHOOTING_MAX_COMPLETION = 700

TRAINING_MAX_PROMPT_CHARS = 18000
TRAINING_MAX_COMPLETION = 1400

WEB_MAX_PROMPT_CHARS = 7000
WEB_MAX_COMPLETION = 900

PRESENTATION_PLAN_MAX_PROMPT_CHARS = 22000
PRESENTATION_PLAN_MAX_COMPLETION = 3000


# ============================================================
# AGENTS
# ============================================================

@dataclass
class MarineAgent:
    name: str
    role: str
    goal: str


TROUBLESHOOTING_AGENT = MarineAgent(
    name="Marine Troubleshooting Agent",
    role="Marine engine troubleshooting technician assistant",
    goal=(
        "Help technicians diagnose marine engine alarms and defects "
        "using supplied manuals first."
    ),
)

TRAINING_AGENT = MarineAgent(
    name="Marine Technical Training Agent",
    role="Marine engine technical instructor",
    goal=(
        "Create accurate, practical and visually understandable "
        "technical training material."
    ),
)


def make_troubleshooting_agent() -> MarineAgent:
    return TROUBLESHOOTING_AGENT


def make_training_agent() -> MarineAgent:
    return TRAINING_AGENT


# ============================================================
# SAFE TEXT HELPERS
# ============================================================

def _to_text(value: Any) -> str:
    """
    Safely convert anything into text.

    This fixes the "'dict' object has no attribute 'strip'"
    problem by never calling .strip() directly on unknown data.
    """

    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, dict):
        # Useful for research dictionaries.
        parts = []

        for key, item in value.items():
            if item is None:
                continue

            if isinstance(item, (dict, list)):
                item = json.dumps(
                    item,
                    ensure_ascii=False,
                    default=str,
                )

            parts.append(
                f"{key}: {item}"
            )

        return "\n".join(parts).strip()

    if isinstance(value, list):
        parts = []

        for item in value:
            text = _to_text(item)

            if text:
                parts.append(text)

        return "\n".join(parts).strip()

    return str(value).strip()


def _clean_text(value: Any) -> str:
    """Normalize whitespace safely."""

    text = _to_text(value)

    text = text.replace("\x00", " ")

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def _bounded(value: Any, limit: int) -> str:
    """Convert to text and safely limit its size."""

    text = _to_text(value)

    if len(text) <= limit:
        return text

    return (
        text[:limit].rstrip()
        + "\n\n[Context truncated for safety.]"
    )


# ============================================================
# API KEYS
# ============================================================

def _get_key(
    provider: str,
    api_key: str | None = None,
) -> str:

    if api_key:
        return _to_text(api_key)

    provider = _clean_text(provider).lower()

    if provider == "groq":
        key = os.getenv("GROQ_API_KEY")

    elif provider == "gemini":
        key = os.getenv("GEMINI_API_KEY")

    else:
        key = None

    if not key:
        raise RuntimeError(
            f"No API key configured for {provider.title()}."
        )

    return _to_text(key)


def _get_tavily_key(
    api_key: str | None = None,
) -> str:

    if api_key:
        return _to_text(api_key)

    key = os.getenv("TAVILY_API_KEY")

    if not key:
        raise RuntimeError(
            "TAVILY_API_KEY is not configured."
        )

    return _to_text(key)


# ============================================================
# LIMITS
# ============================================================

def _limits(
    agent_kind: str,
) -> tuple[int, int]:

    kind = _clean_text(agent_kind).lower()

    if kind == "troubleshooting":
        return (
            TROUBLESHOOTING_MAX_PROMPT_CHARS,
            TROUBLESHOOTING_MAX_COMPLETION,
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


# ============================================================
# GROQ
# ============================================================

def _run_groq(
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    max_tokens: int,
) -> str:

    from groq import Groq

    client = Groq(
        api_key=_to_text(api_key)
    )

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": _to_text(system_prompt),
            },
            {
                "role": "user",
                "content": _to_text(user_prompt),
            },
        ],
        temperature=0.2,
        reasoning_effort="low",
        max_completion_tokens=max_tokens,
    )

    if not response.choices:
        raise RuntimeError(
            "Groq returned no response."
        )

    content = getattr(
        response.choices[0].message,
        "content",
        None,
    )

    if not content:
        raise RuntimeError(
            "Groq returned an empty response."
        )

    return _to_text(content)


# ============================================================
# GEMINI
# ============================================================

def _run_gemini(
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    max_tokens: int,
) -> str:

    from google import genai
    from google.genai import types

    client = genai.Client(
        api_key=_to_text(api_key)
    )

    prompt = (
        "SYSTEM INSTRUCTIONS:\n"
        f"{_to_text(system_prompt)}\n\n"
        "USER REQUEST:\n"
        f"{_to_text(user_prompt)}"
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=max_tokens,
        ),
    )

    text = getattr(
        response,
        "text",
        None,
    )

    if not text:
        raise RuntimeError(
            "Gemini returned an empty response."
        )

    return _to_text(text)


# ============================================================
# MAIN AGENT RUNNER
# ============================================================

def run_marine_agent(
    provider: str,
    agent_kind: str,
    system_prompt: str,
    user_prompt: str,
    api_key: str,
) -> str:

    provider = _clean_text(provider).lower()

    prompt_limit, completion_limit = _limits(
        agent_kind
    )

    safe_system = _bounded(
        system_prompt,
        5000,
    )

    remaining = max(
        1000,
        prompt_limit - len(safe_system),
    )

    safe_user = _bounded(
        user_prompt,
        remaining,
    )

    key = _get_key(
        provider,
        api_key,
    )

    if provider == "groq":
        return _run_groq(
            safe_system,
            safe_user,
            key,
            completion_limit,
        )

    if provider == "gemini":
        return _run_gemini(
            safe_system,
            safe_user,
            key,
            completion_limit,
        )

    raise ValueError(
        f"Unsupported AI provider: {provider}"
    )


# ============================================================
# TAVILY SEARCH
# ============================================================

def _tavily_search(
    query: Any,
    api_key: str,
    max_results: int = 5,
    include_images: bool = False,
) -> dict[str, Any]:

    query_text = _clean_text(query)

    if not query_text:
        return {
            "results": [],
            "images": [],
        }

    payload = {
        "api_key": _to_text(api_key),
        "query": query_text,
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

    clean_results = []

    for item in data.get("results", []):

        if not isinstance(item, dict):
            continue

        clean_results.append(
            {
                "title": _to_text(
                    item.get("title")
                ),
                "url": _to_text(
                    item.get("url")
                ),
                "content": _bounded(
                    item.get("content")
                    or item.get("snippet"),
                    1800,
                ),
            }
        )

    clean_images = []

    for image in data.get("images", []):

        if isinstance(image, str):
            url = image.strip()

        elif isinstance(image, dict):
            url = _to_text(
                image.get("url")
                or image.get("image_url")
            )

        else:
            url = ""

        if url:
            clean_images.append(url)

    return {
        "results": clean_results,
        "images": list(
            dict.fromkeys(clean_images)
        ),
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

    tavily_key = _get_tavily_key()

    research = _tavily_search(
        user_prompt,
        tavily_key,
        max_results=5,
        include_images=False,
    )

    results = research.get(
        "results",
        [],
    )

    if not results:
        return (
            "No reliable web search results were returned. "
            "Please verify the engine model and try again."
        )

    evidence = []

    for number, result in enumerate(
        results,
        start=1,
    ):

        evidence.append(
            f"SOURCE {number}\n"
            f"Title: {_to_text(result.get('title'))}\n"
            f"URL: {_to_text(result.get('url'))}\n"
            f"Content: {_to_text(result.get('content'))}"
        )

    prompt = (
        f"{_to_text(user_prompt)}\n\n"
        "WEB SEARCH EVIDENCE:\n"
        f"{_bounded(evidence, WEB_MAX_PROMPT_CHARS)}\n\n"
        "Use the evidence above. "
        "Do not invent specifications. "
        "Clearly state that the information comes from the web."
    )

    return run_marine_agent(
        provider,
        "web",
        system_prompt,
        prompt,
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

    engine = _clean_text(engine)
    topic = _clean_text(topic)
    ship = _clean_text(ship)
    manual_context = _to_text(manual_context)

    tavily_key = _get_tavily_key(
        tavily_api_key
    )

    queries = []

    if engine:
        queries.append(
            f"{engine} {topic} marine engine technical information"
        )
        queries.append(
            f"{engine} {topic} system diagram schematic"
        )
    else:
        queries.append(
            f"marine diesel engine {topic} technical information"
        )
        queries.append(
            f"marine diesel {topic} system diagram schematic"
        )

    queries.append(
        f"marine diesel engine {topic} training components operation"
    )

    queries.append(
        f"marine diesel {topic} maintenance troubleshooting safety"
    )

    queries = list(
        dict.fromkeys(queries)
    )[:4]

    all_results = []
    all_images = []

    seen_urls = set()
    seen_images = set()

    for query in queries:

        try:
            research = _tavily_search(
                query,
                tavily_key,
                max_results=5,
                include_images=True,
            )

        except Exception:
            continue

        for result in research.get(
            "results",
            [],
        ):

            url = _to_text(
                result.get("url")
            )

            if url and url in seen_urls:
                continue

            if url:
                seen_urls.add(url)

            all_results.append(
                {
                    "title": _to_text(
                        result.get("title")
                    ),
                    "url": url,
                    "content": _bounded(
                        result.get("content"),
                        1600,
                    ),
                }
            )

        for image in research.get(
            "images",
            [],
        ):

            image_url = _to_text(image)

            if (
                image_url
                and image_url not in seen_images
            ):
                seen_images.add(
                    image_url
                )
                all_images.append(
                    image_url
                )

    return {
        "engine": engine,
        "topic": topic,
        "ship": ship,
        "queries": queries,
        "results": all_results[:16],
        "images": all_images[:12],
        "manual_context": _bounded(
            manual_context,
            10000,
        ),
        "manual_note": (
            "Manual information is the primary source. "
            "Online research is supplementary."
            if manual_context
            else
            "Verify online information against the current "
            "manufacturer manual."
        ),
    }


# ============================================================
# JSON PARSER
# ============================================================

def _extract_json(text: Any) -> Any | None:

    if not isinstance(text, str):
        text = _to_text(text)

    cleaned = text.strip()

    cleaned = re.sub(
        r"^```json\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"^```\s*",
        "",
        cleaned,
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned,
    )

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    match = re.search(
        r"\{.*\}",
        cleaned,
        flags=re.DOTALL,
    )

    if match:
        try:
            return json.loads(
                match.group(0)
            )
        except Exception:
            pass

    return None


# ============================================================
# FALLBACK PRESENTATION
# ============================================================

def _fallback_presentation_plan(
    engine: str,
    ship: str,
    topic: str,
) -> dict[str, Any]:

    return {
        "presentation_title": (
            f"{engine} — {topic}"
            if engine
            else f"Marine Engine Training — {topic}"
        ),
        "subtitle": (
            ship
            if ship
            else "Technical Training"
        ),
        "slides": [
            {
                "title": "Learning Objectives",
                "purpose": "Introduce the learning goals.",
                "bullets": [
                    f"Understand the {topic} system",
                    "Identify major components",
                    "Explain the operating sequence",
                    "Recognize common faults",
                ],
                "visual_type": "objectives",
                "visual_query": "",
                "source_preference": "manual_first",
                "speaker_note": "Introduce the training topic.",
            },
            {
                "title": "System Overview",
                "purpose": "Introduce the complete system.",
                "bullets": [
                    "Major components",
                    "System boundaries",
                    "Main flow path",
                ],
                "visual_type": "manual_page",
                "visual_query": f"{engine} {topic} diagram",
                "source_preference": "manual_first",
                "speaker_note": "Use the manual figure when available.",
            },
            {
                "title": "How the System Works",
                "purpose": "Explain the operating sequence.",
                "bullets": [
                    "Input",
                    "Main process",
                    "Control and monitoring",
                    "Output / return",
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
                    "Function",
                    "Operating role",
                    "Inspection points",
                ],
                "visual_type": "component_cards",
                "visual_query": f"marine engine {topic} components",
                "source_preference": "manual_or_web",
                "speaker_note": "Use relevant technical images.",
            },
            {
                "title": "Technician Checks",
                "purpose": "Provide a practical inspection sequence.",
                "bullets": [
                    "Confirm the symptom",
                    "Check external conditions",
                    "Verify measurements",
                    "Inspect relevant components",
                ],
                "visual_type": "checklist",
                "visual_query": "",
                "source_preference": "manual_first",
                "speaker_note": "Follow the manufacturer's procedure.",
            },
            {
                "title": "Fault Diagnosis",
                "purpose": "Connect symptoms to checks.",
                "bullets": [
                    "Symptom",
                    "Possible area",
                    "Verification",
                    "Corrective action",
                ],
                "visual_type": "diagnostic_flow",
                "visual_query": f"marine diesel {topic} troubleshooting",
                "source_preference": "manual_first",
                "speaker_note": "Verify before replacing components.",
            },
            {
                "title": "Maintenance & Safety",
                "purpose": "Cover maintenance and hazards.",
                "bullets": [
                    "Follow maintenance intervals",
                    "Use required PPE",
                    "Isolate energy sources",
                    "Beware of hot and pressurized systems",
                ],
                "visual_type": "safety_callout",
                "visual_query": f"marine diesel {topic} maintenance safety",
                "source_preference": "manual_first",
                "speaker_note": "Use exact manual safety instructions.",
            },
            {
                "title": "Knowledge Check",
                "purpose": "Reinforce learning.",
                "bullets": [
                    "What is the normal flow path?",
                    "Which components need inspection?",
                    "What can cause the symptom?",
                ],
                "visual_type": "questions",
                "visual_query": "",
                "source_preference": "generated",
                "speaker_note": "Use as a technician discussion.",
            },
            {
                "title": "References",
                "purpose": "Show technical sources.",
                "bullets": [
                    "Supplied engine manuals",
                    "Manufacturer documentation",
                    "Technical marine references",
                ],
                "visual_type": "references",
                "visual_query": "",
                "source_preference": "all_sources",
                "speaker_note": "Verify details against the current manual.",
            },
        ],
    }


# ============================================================
# PRESENTATION PLANNER
# ============================================================

def generate_training_presentation_plan(
    provider: str,
    engine: str,
    ship: str,
    topic: str,
    manual_context: Any,
    web_evidence: Any,
) -> dict[str, Any]:

    engine = _clean_text(engine)
    ship = _clean_text(ship)
    topic = _clean_text(topic)

    # IMPORTANT:
    # manual_context may accidentally arrive as a dict.
    # Convert it safely instead of calling .strip().
    manual_context = _to_text(
        manual_context
    )

    # IMPORTANT:
    # web_evidence is intentionally a dictionary.
    # Never call .strip() on it.
    if isinstance(
        web_evidence,
        dict,
    ):
        research = web_evidence
    else:
        research = {
            "results": [],
            "images": [],
        }

    results = research.get(
        "results",
        [],
    )

    images = research.get(
        "images",
        [],
    )

    # --------------------------------------------------------
    # Web evidence
    # --------------------------------------------------------

    evidence_parts = []

    for number, result in enumerate(
        results[:16],
        start=1,
    ):

        if not isinstance(
            result,
            dict,
        ):
            continue

        evidence_parts.append(
            f"WEB SOURCE {number}\n"
            f"Title: {_to_text(result.get('title'))}\n"
            f"URL: {_to_text(result.get('url'))}\n"
            f"Evidence: {_to_text(result.get('content'))}"
        )

    web_text = "\n\n".join(
        evidence_parts
    )

    image_text = "\n".join(
        f"- {_to_text(url)}"
        for url in images[:12]
    )

    # --------------------------------------------------------
    # System prompt
    # --------------------------------------------------------

    system_prompt = """
You are the MarineWise AI Professional Training Presentation Planner.

Create a professional technical training presentation for marine
engineers and technicians.

The supplied manufacturer manual is the PRIMARY technical source.

Online research is SECONDARY.

Do not override specific manufacturer instructions with generic
online information.

Create 8 to 10 slides.

Each slide must contain:
- title
- purpose
- 3 to 5 concise bullets
- visual_type
- visual_query
- source_preference
- speaker_note

Use:
manual_page
photo_or_diagram
process_diagram
diagnostic_flow
checklist
safety_callout
component_cards
questions
references

Do not create walls of text.

Return ONLY valid JSON.
""".strip()

    # --------------------------------------------------------
    # JSON example kept OUTSIDE the f-string.
    # This prevents the previous format-specifier error.
    # --------------------------------------------------------

    json_example = """
{
  "presentation_title": "Example",
  "subtitle": "Example",
  "slides": [
    {
      "title": "Example Slide",
      "purpose": "Explain the topic",
      "bullets": [
        "Point one",
        "Point two",
        "Point three"
      ],
      "visual_type": "process_diagram",
      "visual_query": "marine engine system diagram",
      "source_preference": "manual_first",
      "speaker_note": "Explain the diagram."
    }
  ]
}
"""

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

Create an 8 to 10 slide professional training presentation.

Recommended structure:

1. Title
2. Learning objectives
3. System overview
4. How the system works
5. Component spotlight
6. System/process diagram
7. Fault diagnosis and technician checks
8. Maintenance and safety
9. Knowledge check
10. References

Use the manual first whenever relevant.

JSON format example:

{json_example}
""".strip()

    user_prompt = _bounded(
        user_prompt,
        PRESENTATION_PLAN_MAX_PROMPT_CHARS,
    )

    # --------------------------------------------------------
    # Run AI
    # --------------------------------------------------------

    try:

        provider_clean = _clean_text(
            provider
        )

        if provider_clean.lower() == "groq":
            environment_key = os.getenv(
                "GROQ_API_KEY"
            )
        else:
            environment_key = os.getenv(
                "GEMINI_API_KEY"
            )

        raw = run_marine_agent(
            provider_clean,
            "presentation",
            system_prompt,
            user_prompt,
            environment_key or "",
        )

    except Exception:
        return _fallback_presentation_plan(
            engine,
            ship,
            topic,
        )

    # --------------------------------------------------------
    # Parse JSON
    # --------------------------------------------------------

    parsed = _extract_json(
        raw
    )

    if not isinstance(
        parsed,
        dict,
    ):
        return _fallback_presentation_plan(
            engine,
            ship,
            topic,
        )

    slides = parsed.get(
        "slides",
        [],
    )

    if not isinstance(
        slides,
        list,
    ):
        return _fallback_presentation_plan(
            engine,
            ship,
            topic,
        )

    # --------------------------------------------------------
    # Normalize slides
    # --------------------------------------------------------

    normalized = []

    for slide in slides[:10]:

        if not isinstance(
            slide,
            dict,
        ):
            continue

        title = _clean_text(
            slide.get("title")
        )

        if not title:
            continue

        bullets = slide.get(
            "bullets",
            [],
        )

        if not isinstance(
            bullets,
            list,
        ):
            bullets = [
                bullets
            ]

        clean_bullets = []

        for bullet in bullets[:5]:

            bullet_text = _clean_text(
                bullet
            )

            if bullet_text:
                clean_bullets.append(
                    bullet_text
                )

        normalized.append(
            {
                "title": title,

                "purpose": _clean_text(
                    slide.get("purpose")
                ),

                "bullets": clean_bullets,

                "visual_type": (
                    _clean_text(
                        slide.get(
                            "visual_type"
                        )
                    )
                    or "photo_or_diagram"
                ),

                "visual_query": _clean_text(
                    slide.get(
                        "visual_query"
                    )
                ),

                "source_preference": (
                    _clean_text(
                        slide.get(
                            "source_preference"
                        )
                    )
                    or "manual_first"
                ),

                "speaker_note": _clean_text(
                    slide.get(
                        "speaker_note"
                    )
                ),
            }
        )

    if len(normalized) < 8:
        return _fallback_presentation_plan(
            engine,
            ship,
            topic,
        )

    return {
        "presentation_title": (
            _clean_text(
                parsed.get(
                    "presentation_title"
                )
            )
            or f"{engine} — {topic}"
        ),

        "subtitle": (
            _clean_text(
                parsed.get(
                    "subtitle"
                )
            )
            or ship
            or "Technical Training"
        ),

        "slides": normalized,
    }
