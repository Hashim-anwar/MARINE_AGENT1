"""MarineWise AI provider and agent layer.

Two logical agents:

1. Troubleshooting Agent
2. Technical Training Agent

AI providers:

- Groq
- Gemini

Online search:

- Tavily

Groq and Gemini are called directly with their official SDKs.
CrewAI/LiteLLM is intentionally not used for the model transport because
the previous LiteLLM path produced Groq-specific request errors.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import requests
from groq import Groq


GROQ_MODEL = "openai/gpt-oss-120b"

# Keep this configurable because Gemini model availability can change.
GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.8-flash",
)


# ---------------------------------------------------------------------
# Separate request budgets
# ---------------------------------------------------------------------

# Troubleshooting gets a deliberately small budget.
TROUBLESHOOTING_MAX_PROMPT_CHARS = 6500
TROUBLESHOOTING_MAX_COMPLETION = 700

# Training has its own independent budget.
TRAINING_MAX_PROMPT_CHARS = 18000
TRAINING_MAX_COMPLETION = 1400

# Online search has its own budget.
WEB_MAX_PROMPT_CHARS = 7000
WEB_MAX_COMPLETION = 900


@dataclass(frozen=True)
class MarineAgent:
    name: str
    role: str
    goal: str
    backstory: str


TROUBLESHOOTING_AGENT = MarineAgent(
    name="Troubleshooting Agent",
    role="Marine Engine Troubleshooting Specialist",
    goal=(
        "Answer marine engine troubleshooting questions "
        "from supplied evidence without inventing facts."
    ),
    backstory=(
        "A careful senior marine technician who preserves "
        "source and page references."
    ),
)


TRAINING_AGENT = MarineAgent(
    name="Training Agent",
    role="Marine Technical Training Specialist",
    goal=(
        "Create, explain, assess, and remediate marine "
        "technician training using supplied evidence."
    ),
    backstory=(
        "An experienced marine instructor who creates "
        "practical training, quizzes, scoring feedback, "
        "and remedial lessons."
    ),
)


def make_troubleshooting_agent() -> MarineAgent:
    return TROUBLESHOOTING_AGENT


def make_training_agent() -> MarineAgent:
    return TRAINING_AGENT


def _get_key(
    provider: str,
    api_key: str | None = None,
) -> str:

    if api_key:
        return api_key

    if provider == "Groq":
        env_name = "GROQ_API_KEY"
    elif provider == "Gemini":
        env_name = "GEMINI_API_KEY"
    else:
        raise ValueError(
            "Provider must be Groq or Gemini."
        )

    key = os.getenv(env_name)

    if not key:
        raise RuntimeError(
            f"{env_name} is not configured."
        )

    return key


def _bounded(
    text: str,
    limit: int,
) -> str:

    text = text or ""

    if len(text) <= limit:
        return text

    return (
        text[:limit]
        + "\n\n"
        "[MarineWise: context truncated for request safety.]"
    )


def _limits(
    agent_kind: str,
) -> tuple[int, int]:

    if agent_kind == "troubleshooting":
        return (
            TROUBLESHOOTING_MAX_PROMPT_CHARS,
            TROUBLESHOOTING_MAX_COMPLETION,
        )

    if agent_kind == "training":
        return (
            TRAINING_MAX_PROMPT_CHARS,
            TRAINING_MAX_COMPLETION,
        )

    raise ValueError(
        f"Unknown agent kind: {agent_kind}"
    )


# ---------------------------------------------------------------------
# Groq
# ---------------------------------------------------------------------

def _run_groq(
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    max_tokens: int,
) -> str:

    client = Groq(
        api_key=api_key
    )

    response = (
        client.chat.completions.create(
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
    )

    answer = (
        response.choices[0]
        .message.content
    )

    return answer or "No answer was returned."


# ---------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------

def _run_gemini(
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    max_tokens: int,
) -> str:

    from google import genai
    from google.genai import types

    client = genai.Client(
        api_key=api_key
    )

    response = (
        client.models.generate_content(
            model=GEMINI_MODEL,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.2,
                max_output_tokens=max_tokens,
            ),
        )
    )

    return (
        response.text
        or "No answer was returned."
    )


# ---------------------------------------------------------------------
# Main agent runner
# ---------------------------------------------------------------------

def run_marine_agent(
    provider: str,
    agent_kind: str,
    system_prompt: str,
    user_prompt: str,
    api_key: str | None = None,
) -> str:
    """Run Troubleshooting or Training Agent."""

    if provider not in {
        "Groq",
        "Gemini",
    }:
        raise ValueError(
            "Provider must be Groq or Gemini."
        )

    if agent_kind == "troubleshooting":
        agent = make_troubleshooting_agent()

    elif agent_kind == "training":
        agent = make_training_agent()

    else:
        raise ValueError(
            f"Unknown agent kind: {agent_kind}"
        )

    full_system = (
        f"You are the MarineWise {agent.name}.\n\n"
        f"ROLE: {agent.role}\n"
        f"GOAL: {agent.goal}\n"
        f"BACKGROUND: {agent.backstory}\n\n"
        f"{system_prompt}"
    )

    prompt_limit, completion_limit = _limits(
        agent_kind
    )

    bounded_user_prompt = _bounded(
        user_prompt,
        prompt_limit,
    )

    key = _get_key(
        provider,
        api_key,
    )

    if provider == "Groq":

        return _run_groq(
            full_system,
            bounded_user_prompt,
            key,
            completion_limit,
        )

    return _run_gemini(
        full_system,
        bounded_user_prompt,
        key,
        completion_limit,
    )


# ---------------------------------------------------------------------
# Tavily
# ---------------------------------------------------------------------

def _get_tavily_key() -> str:

    key = os.getenv(
        "TAVILY_API_KEY"
    )

    if not key:
        raise RuntimeError(
            "TAVILY_API_KEY is not configured."
        )

    return key


def _tavily_search(
    query: str,
    api_key: str,
    max_results: int = 5,
) -> list[dict[str, Any]]:

    response = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": api_key,
            "query": query,
            "search_depth": "advanced",
            "max_results": max_results,
            "include_answer": False,
            "include_raw_content": False,
        },
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    return data.get(
        "results",
        [],
    )


def run_web_search(
    provider: str,
    system_prompt: str,
    user_prompt: str,
    api_key: str | None = None,
) -> str:
    """Search with Tavily and summarize using Groq or Gemini."""

    if provider not in {
        "Groq",
        "Gemini",
    }:
        raise ValueError(
            "Provider must be Groq or Gemini."
        )

    tavily_key = _get_tavily_key()

    results = _tavily_search(
        user_prompt,
        tavily_key,
        max_results=5,
    )

    if not results:
        return (
            "No relevant online results were found "
            "by Tavily."
        )

    evidence_parts = []

    for number, item in enumerate(
        results[:5],
        start=1,
    ):

        title = item.get(
            "title",
            "Untitled",
        )

        url = item.get(
            "url",
            "",
        )

        content = _bounded(
            item.get(
                "content",
                "",
            ),
            1000,
        )

        evidence_parts.append(
            f"WEB SOURCE {number}\n"
            f"TITLE: {title}\n"
            f"URL: {url}\n"
            f"CONTENT:\n{content}"
        )

    evidence = _bounded(
        "\n\n".join(
            evidence_parts
        ),
        5000,
    )

    final_system = (
        f"{system_prompt}\n\n"
        "The web evidence below came from Tavily. "
        "Use only that evidence for online factual claims. "
        "Clearly state that the answer is based on online "
        "sources. Do not invent information. "
        "Include source URLs when useful."
    )

    final_user = _bounded(
        f"USER REQUEST:\n"
        f"{user_prompt}\n\n"
        f"TAVILY WEB RESULTS:\n"
        f"{evidence}",
        WEB_MAX_PROMPT_CHARS,
    )

    key = _get_key(
        provider,
        api_key,
    )

    if provider == "Groq":

        return _run_groq(
            final_system,
            final_user,
            key,
            WEB_MAX_COMPLETION,
        )

    return _run_gemini(
        final_system,
        final_user,
        key,
        WEB_MAX_COMPLETION,
    )
