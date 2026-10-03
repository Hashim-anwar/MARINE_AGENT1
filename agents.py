"""
MarineWise AI - AI agents, Tavily research, presentation planning, and CrewAI orchestration.

CrewAI is used by the Marine AI Command Center. Existing direct Groq/Gemini functions remain available for the original app pages.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

import requests

# CrewAI is intentionally optional at import time so the existing
# Troubleshooting/Training pages can still load before the dependency is installed.
# The Command Center raises a clear installation error if CrewAI is unavailable.
try:
    from crewai import Agent, Crew, Process, Task, LLM
    from crewai.llms.base_llm import BaseLLM
    from crewai.tools import BaseTool
    CREWAI_AVAILABLE = True
    CREWAI_IMPORT_ERROR = ""
except Exception as _crewai_exc:
    Agent = Crew = Process = Task = LLM = None
    BaseTool = None
    BaseLLM = None
    CREWAI_AVAILABLE = False
    CREWAI_IMPORT_ERROR = str(_crewai_exc)


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

# ============================================================
# CREWAI MARINE AI COMMAND CENTER
# ============================================================
#
# This is a REAL CrewAI implementation:
#   - Agent objects are created with crewai.Agent
#   - Task objects are created with crewai.Task
#   - A Crew object orchestrates them
#   - Process.sequential passes completed task outputs into later tasks
#   - The approved Tavily search is exposed as a real CrewAI BaseTool
#
# The original direct SDK functions above are intentionally preserved so
# Troubleshooting, Training, Quiz and Presentation pages are not disturbed.
# The new Command Center uses CrewAI end-to-end.

if CREWAI_AVAILABLE:
    class MarineTavilySearchTool(BaseTool):
        """CrewAI tool that searches Tavily for approved marine research."""

        name: str = "marine_tavily_search"
        description: str = (
            "Search Tavily for current marine-engine technical information. "
            "Use only when the user has already approved online research. "
            "Return titles, URLs and concise evidence."
        )

        def _run(self, query: str) -> str:
            try:
                key = _get_tavily_key()
                data = _tavily_search(
                    query,
                    key,
                    max_results=5,
                    include_images=False,
                )
                results = data.get("results", [])
                if not results:
                    return "No Tavily results were returned."

                blocks = []
                for i, item in enumerate(results, start=1):
                    blocks.append(
                        f"SOURCE {i}\n"
                        f"Title: {_to_text(item.get('title'))}\n"
                        f"URL: {_to_text(item.get('url'))}\n"
                        f"Evidence: {_to_text(item.get('content'))}"
                    )
                return _bounded("\n\n".join(blocks), 7000)
            except Exception as exc:
                return f"Tavily search failed: {exc}"


_CrewAIBaseLLM = BaseLLM if CREWAI_AVAILABLE else object


class MarineGroqCrewLLM(_CrewAIBaseLLM):
    """CrewAI adapter that calls Groq directly without CrewAI model rewriting.

    This is intentionally used only by the CrewAI Command Center. The rest of
    MarineWise continues using the existing direct Groq/Gemini functions.
    """

    llm_type: str = "marinewise_groq"

    def __init__(self, model: str, api_key: str, temperature: float = 0.2):
        super().__init__(model=model, temperature=temperature)
        self.api_key = api_key
        self.endpoint = "https://api.groq.com/openai/v1/chat/completions"

    def call(
        self,
        messages,
        tools=None,
        callbacks=None,
        available_functions=None,
        from_task=None,
        from_agent=None,
        response_model=None,
        **kwargs,
    ):
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        # CrewAI may attach internal metadata to messages. In particular,
        # recent CrewAI versions can add `cache_breakpoint`, which is a
        # CrewAI/provider-side field and is NOT accepted by Groq's Chat
        # Completions endpoint. Send only the message fields Groq supports.
        sanitized_messages = []
        for message in messages:
            if isinstance(message, dict):
                clean_message = {}
                for key in (
                    "role",
                    "content",
                    "name",
                    "tool_call_id",
                    "tool_calls",
                ):
                    if key in message and message[key] is not None:
                        clean_message[key] = message[key]
                sanitized_messages.append(clean_message)
            else:
                sanitized_messages.append(message)

        payload = {
            "model": self.model,
            "messages": sanitized_messages,
            "temperature": self.temperature if self.temperature is not None else 0.2,
            "reasoning_effort": "low",
        }

        max_tokens = kwargs.get("max_tokens") or kwargs.get("max_completion_tokens")
        if max_tokens:
            payload["max_completion_tokens"] = int(max_tokens)

        if self.stop:
            payload["stop"] = self.stop

        # The Command Center already supplies approved web evidence to the
        # research agent. We deliberately do not enable function calling here;
        # this avoids another provider-routing layer while keeping Tavily as a
        # separately available CrewAI tool for future extension.
        # Other CrewAI agents do not require tools for their core workflow.

        response = requests.post(
            self.endpoint,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=180,
        )

        if not response.ok:
            try:
                detail = response.json()
            except Exception:
                detail = response.text
            raise RuntimeError(
                f"Groq CrewAI request failed with HTTP {response.status_code}: {detail}"
            )

        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"Groq returned no choices: {data}")

        message = choices[0].get("message") or {}
        content = message.get("content")

        if content is None:
            # Some model/API responses can expose content in a different form.
            # Return a useful error rather than letting CrewAI fail obscurely.
            raise RuntimeError(f"Groq returned no message content: {data}")

        return str(content)

    def supports_function_calling(self) -> bool:
        return False

    def supports_stop_words(self) -> bool:
        return True

    def get_context_window_size(self) -> int:
        return 131072


def _crewai_llm(provider: str, api_key: str):
    """Create the CrewAI LLM used by every MarineWise Command Center agent."""
    if not CREWAI_AVAILABLE:
        raise RuntimeError(
            "CrewAI is not installed. Install the 'crewai' package before "
            "using Marine AI Command Center."
        )

    provider_clean = _clean_text(provider).lower()
    key = _get_key(provider_clean, api_key)

    if provider_clean == "groq":
        # IMPORTANT:
        # Do NOT pass Groq GPT-OSS through CrewAI's OpenAI provider here.
        # CrewAI can normalize `openai/gpt-oss-120b` to `gpt-oss-120b`,
        # which Groq correctly rejects. This custom adapter sends the exact
        # Groq model ID directly to Groq's OpenAI-compatible endpoint.
        return MarineGroqCrewLLM(
            model=GROQ_MODEL,
            api_key=key,
            temperature=0.2,
        )

    if provider_clean == "gemini":
        return LLM(
            model=f"gemini/{GEMINI_MODEL}",
            api_key=key,
        )

    raise ValueError(f"Unsupported CrewAI provider: {provider_clean}")


def _crew_output_text(value: Any) -> str:
    """Extract readable text from CrewAI CrewOutput/TaskOutput objects."""
    if value is None:
        return ""

    raw = getattr(value, "raw", None)
    if raw:
        return _to_text(raw)

    output = getattr(value, "output", None)
    if output:
        return _to_text(output)

    return _to_text(value)


def _task_output_record(task: Any) -> dict[str, str]:
    """Convert a CrewAI task output to the Command Center UI format."""
    output = getattr(task, "output", None)
    return {
        "agent": _to_text(getattr(getattr(task, "agent", None), "role", "")),
        "task": _bounded(
            _to_text(getattr(task, "description", "")),
            500,
        ),
        "output": _bounded(_crew_output_text(output), 5000),
    }


def run_marine_command_center(
    provider: str,
    topic: str,
    engine_model: str = "",
    ship: str = "",
    objective: str = "",
    manual_context: Any = "",
    web_context: Any = "",
    api_key: str = "",
) -> dict[str, Any]:
    """
    Run the MarineWise multi-agent Command Center with REAL CrewAI.

    Collaboration chain:

        Orchestrator Agent
              |
              v
        Manual/RAG Analyst
              |
              +----> Research Agent + Tavily (ONLY when web evidence exists,
              |      which means the user already approved web research)
              |
              v
        Marine Technical Specialist
              |
              v
        Technician Action Agent
              |
              v
        Crew final output

    The manual evidence is supplied by MarineWise's existing FAISS/RAG layer.
    The Research Agent is not created at all when web_context is empty, so
    the Command Center cannot silently perform online research.
    """

    if not CREWAI_AVAILABLE:
        raise RuntimeError(
            "CrewAI is not installed. Add 'crewai' to requirements.txt and "
            "install it before using Marine AI Command Center. "
            f"Import detail: {CREWAI_IMPORT_ERROR}"
        )

    provider_clean = _clean_text(provider).lower()
    topic_clean = _clean_text(topic)
    engine_clean = _clean_text(engine_model)
    ship_clean = _clean_text(ship)
    objective_clean = _clean_text(objective)
    manual_text = _bounded(manual_context, 9000)
    web_text = _bounded(web_context, 8000)

    if not topic_clean:
        raise ValueError("A technical topic is required for the Command Center.")

    if not manual_text:
        manual_text = (
            "NO MANUAL/RAG EVIDENCE WAS RETRIEVED. Do not invent manufacturer "
            "specifications, limits, procedures, locations or measurements."
        )

    # Presence of web_context is the approval gate used by the current app:
    # the app only creates this context after the user clicks Search Online.
    web_approved = bool(web_text)

    llm = _crewai_llm(provider_clean, api_key)

    case_summary = (
        f"TOPIC: {topic_clean}\n"
        f"ENGINE/EQUIPMENT MODEL: {engine_clean or 'Not specified'}\n"
        f"SHIP: {ship_clean or 'Not specified'}\n"
        f"TECHNICAL OBJECTIVE: {objective_clean or 'Not specified'}"
    )

    # --------------------------------------------------------
    # 1. Orchestrator
    # --------------------------------------------------------
    orchestrator = Agent(
        role="Marine AI Orchestrator",
        goal=(
            "Coordinate a disciplined marine technical investigation. "
            "Keep manual/RAG evidence primary, identify evidence gaps, "
            "and pass a clear plan to the specialist agents."
        ),
        backstory=(
            "You are the lead coordinator of MarineWise. You do not invent "
            "technical facts. You organize the evidence and decide which "
            "specialist should address each part of the case."
        ),
        llm=llm,
        allow_delegation=False,
        verbose=False,
        max_iter=3,
    )

    orchestration_task = Task(
        description=(
            f"Review this marine technical case:\n{case_summary}\n\n"
            "Create a concise collaboration plan. Identify the key technical "
            "questions, the evidence that must be verified, and what the "
            "downstream Marine Technical Specialist must not assume."
        ),
        expected_output=(
            "A concise technical investigation plan with evidence priorities "
            "and explicit warnings about unsupported assumptions."
        ),
        agent=orchestrator,
    )

    # --------------------------------------------------------
    # 2. Manual/RAG Analyst
    # --------------------------------------------------------
    manual_agent = Agent(
        role="Marine Manual and RAG Evidence Analyst",
        goal=(
            "Extract only technically supported facts from the supplied "
            "MarineWise manual/RAG evidence and identify missing evidence."
        ),
        backstory=(
            "You are a marine documentation specialist. You distinguish "
            "manufacturer evidence from generic knowledge and never fill a "
            "missing specification with a guess."
        ),
        llm=llm,
        allow_delegation=False,
        verbose=False,
        max_iter=3,
    )

    manual_task = Task(
        description=(
            f"CASE:\n{case_summary}\n\n"
            f"MANUAL/RAG EVIDENCE:\n{manual_text}\n\n"
            "Analyze the supplied evidence. Extract relevant components, "
            "operating relationships, procedures, checks, warnings, "
            "measurements and source references only when actually supported. "
            "Explicitly list important information that the evidence does not "
            "support. Do not use generic marine knowledge to fill gaps."
        ),
        expected_output=(
            "A manual-grounded evidence brief containing supported facts, "
            "relevant source details, and an explicit evidence-gap list."
        ),
        agent=manual_agent,
        context=[orchestration_task],
    )

    agents = [orchestrator, manual_agent]
    tasks = [orchestration_task, manual_task]
    research_agent = None
    research_task = None

    # --------------------------------------------------------
    # 3. Optional Research Agent + real Tavily tool
    # --------------------------------------------------------
    if web_approved:
        tavily_tool = MarineTavilySearchTool()

        research_agent = Agent(
            role="Marine Web Research Agent",
            goal=(
                "Review approved online evidence and, when useful, validate "
                "or supplement the case using the Tavily search tool. "
                "Clearly separate web evidence from manual evidence."
            ),
            backstory=(
                "You are a marine technical researcher. Online information is "
                "secondary to the supplied manual. You search only because "
                "the user has explicitly approved online research."
            ),
            llm=llm,
            tools=[tavily_tool],
            allow_delegation=False,
            verbose=False,
            max_iter=3,
        )

        research_task = Task(
            description=(
                f"CASE:\n{case_summary}\n\n"
                f"APPROVED WEB EVIDENCE:\n{web_text}\n\n"
                "Review the approved web evidence. You may use the attached "
                "Tavily tool to validate or supplement important points, but "
                "do not replace manufacturer-specific instructions with generic "
                "web content. For every important web-derived point, retain the "
                "source title/URL when available. Flag conflicts with the "
                "manual rather than deciding silently which source is correct."
            ),
            expected_output=(
                "A concise secondary research brief with source URLs, validated "
                "facts, unresolved conflicts and limitations."
            ),
            agent=research_agent,
            context=[orchestration_task, manual_task],
            tools=[tavily_tool],
        )
        agents.append(research_agent)
        tasks.append(research_task)

    # --------------------------------------------------------
    # 4. Marine Technical Specialist
    # --------------------------------------------------------
    specialist = Agent(
        role="Marine Technical Specialist",
        goal=(
            "Synthesize the evidence into a technically disciplined answer "
            "for a marine engineer or technician without inventing unsupported "
            "specifications or procedures."
        ),
        backstory=(
            "You are an experienced marine technical specialist. You reason "
            "from evidence, distinguish confirmed facts from possibilities, "
            "and prioritize safe, manufacturer-supported actions."
        ),
        llm=llm,
        allow_delegation=False,
        verbose=False,
        max_iter=4,
    )

    specialist_context = [orchestration_task, manual_task]
    if research_task is not None:
        specialist_context.append(research_task)

    specialist_task = Task(
        description=(
            f"CASE:\n{case_summary}\n\n"
            "Using the upstream agent outputs, build the technical answer. "
            "Explain the relevant system/components, operating sequence, "
            "diagnostic logic or training points as appropriate to the topic. "
            "Separate confirmed manual facts, secondary web evidence and "
            "reasonable but unconfirmed possibilities. Never invent values, "
            "alarm limits, component locations, tolerances or maintenance "
            "intervals. If evidence is insufficient, say so."
        ),
        expected_output=(
            "A technically grounded synthesis suitable for the final technician "
            "response, with evidence status and limitations clearly stated."
        ),
        agent=specialist,
        context=specialist_context,
    )
    agents.append(specialist)
    tasks.append(specialist_task)

    # --------------------------------------------------------
    # 5. Technician Action Agent
    # --------------------------------------------------------
    action_agent = Agent(
        role="Marine Technician Action and Learning Agent",
        goal=(
            "Turn the specialist synthesis into a clear, practical and safe "
            "technician-facing response while preserving evidence boundaries."
        ),
        backstory=(
            "You are the final MarineWise technician communication specialist. "
            "Your output must be useful at the worksite or in training, but you "
            "must never turn an unsupported assumption into an instruction."
        ),
        llm=llm,
        allow_delegation=False,
        verbose=False,
        max_iter=4,
    )

    action_task = Task(
        description=(
            f"CASE:\n{case_summary}\n\n"
            "Create the final MarineWise response from the specialist output. "
            "Use clear headings and practical steps where appropriate. Include "
            "safety/isolation considerations when supported by the evidence. "
            "Clearly label unsupported or unverified items. End with a short "
            "'Evidence and limitations' section stating that manufacturer "
            "documentation remains authoritative. Do not fabricate sources."
        ),
        expected_output=(
            "A professional technician-facing final response with clear "
            "technical reasoning, practical actions, safety notes, and evidence "
            "limitations."
        ),
        agent=action_agent,
        context=[specialist_task],
    )
    agents.append(action_agent)
    tasks.append(action_task)

    # --------------------------------------------------------
    # REAL CREWAI CREW
    # --------------------------------------------------------
    crew = Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential,
        verbose=False,
    )

    result = crew.kickoff()
    final_text = _crew_output_text(result)

    task_outputs = [
        _task_output_record(task)
        for task in tasks
        if getattr(task, "output", None) is not None
    ]

    return {
        "final": final_text,
        "agent_names": [
            _to_text(getattr(agent, "role", ""))
            for agent in agents
        ],
        "task_outputs": task_outputs,
        "used_crewai": True,
        "used_web_evidence": web_approved,
        "used_tavily_tool": web_approved,
        "provider": provider_clean,
        "topic": topic_clean,
        "engine_model": engine_clean,
        "ship": ship_clean,
        "objective": objective_clean,
    }
