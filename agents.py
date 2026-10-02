"""MarineWise AI provider and agent layer.

Two logical agents are kept separate:
- Troubleshooting Agent
- Technical Training Agent

The model transport can be selected between Groq and Gemini.  Both providers
are called directly with their official Python SDKs; CrewAI/LiteLLM is not used
for the model request because the previous LiteLLM path caused Groq-specific
request incompatibilities.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import requests
from groq import Groq

GROQ_MODEL = "openai/gpt-oss-120b"
GEMINI_MODEL = "gemini-3.8-flash"

# Deliberately different limits: troubleshooting is the strict path that was
# causing request-limit errors; training keeps a larger independent budget.
TROUBLESHOOTING_MAX_PROMPT_CHARS = 6500
TROUBLESHOOTING_MAX_COMPLETION = 700
TRAINING_MAX_PROMPT_CHARS = 18000
TRAINING_MAX_COMPLETION = 1400
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
    goal="Answer marine engine troubleshooting questions from supplied evidence without inventing facts.",
    backstory="A careful senior marine technician who preserves source and page references.",
)

TRAINING_AGENT = MarineAgent(
    name="Training Agent",
    role="Marine Technical Training Specialist",
    goal="Create, explain, assess, and remediate marine technician training using supplied evidence.",
    backstory="An experienced marine instructor who creates practical training, quizzes, scoring feedback, and remedial lessons.",
)


def make_troubleshooting_agent() -> MarineAgent:
    return TROUBLESHOOTING_AGENT


def make_training_agent() -> MarineAgent:
    return TRAINING_AGENT


def _get_key(provider: str, api_key: str | None = None) -> str:
    if api_key:
        return api_key
    env_name = "GROQ_API_KEY" if provider == "Groq" else "GEMINI_API_KEY"
    key = os.getenv(env_name)
    if not key:
        raise RuntimeError(f"{env_name} is not configured.")
    return key


def _limits(agent_kind: str) -> tuple[int, int]:
    if agent_kind == "troubleshooting":
        return TROUBLESHOOTING_MAX_PROMPT_CHARS, TROUBLESHOOTING_MAX_COMPLETION
    return TRAINING_MAX_PROMPT_CHARS, TRAINING_MAX_COMPLETION


def _bounded(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n[Context truncated by MarineWise for request safety.]"


def _run_groq(system_prompt: str, user_prompt: str, api_key: str, max_tokens: int) -> str:
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        reasoning_effort="low",
        max_completion_tokens=max_tokens,
    )
    return response.choices[0].message.content or "No answer was returned."


def _run_gemini(system_prompt: str, user_prompt: str, api_key: str, max_tokens: int) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.2,
            max_output_tokens=max_tokens,
        ),
    )
    return response.text or "No answer was returned."


def run_marine_agent(
    provider: str,
    agent_kind: str,
    system_prompt: str,
    user_prompt: str,
    api_key: str | None = None,
) -> str:
    """Run a MarineWise logical agent using Groq or Gemini directly."""
    factories = {
        "troubleshooting": make_troubleshooting_agent,
        "training": make_training_agent,
    }
    if agent_kind not in factories:
        raise ValueError(f"Unknown agent kind: {agent_kind}")
    if provider not in {"Groq", "Gemini"}:
        raise ValueError("Provider must be Groq or Gemini.")

    agent = factories[agent_kind]()
    full_system = (
        f"You are the MarineWise {agent.name}.\n"
        f"ROLE: {agent.role}\nGOAL: {agent.goal}\nBACKGROUND: {agent.backstory}\n\n"
        f"{system_prompt}"
    )
    prompt_limit, completion_limit = _limits(agent_kind)
    user_prompt = _bounded(user_prompt, prompt_limit)
    key = _get_key(provider, api_key)

    if provider == "Groq":
        return _run_groq(full_system, user_prompt, key, completion_limit)
    return _run_gemini(full_system, user_prompt, key, completion_limit)


def _get_tavily_key() -> str:
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        raise RuntimeError("TAVILY_API_KEY is not configured.")
    return key


def _tavily_search(query: str, api_key: str, max_results: int = 5) -> list[dict[str, Any]]:
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
    return data.get("results", [])


def run_web_search(
    provider: str,
    system_prompt: str,
    user_prompt: str,
    api_key: str | None = None,
) -> str:
    """Search the web with Tavily, then summarize results with Groq or Gemini.

    Tavily is deliberately separate from the selected model provider. The
    selected Groq/Gemini model only receives a small set of search results.
    """
    if provider not in {"Groq", "Gemini"}:
        raise ValueError("Provider must be Groq or Gemini.")

    tavily_results = _tavily_search(user_prompt, _get_tavily_key(), max_results=5)
    if not tavily_results:
        return "No relevant web results were found by Tavily."

    evidence_parts = []
    for i, item in enumerate(tavily_results[:5], start=1):
        title = item.get("title", "Untitled")
        url = item.get("url", "")
        content = _bounded(item.get("content", ""), 1200)
        evidence_parts.append(f"WEB SOURCE {i}: {title}\nURL: {url}\n{content}")
    evidence = _bounded("\n\n".join(evidence_parts), 6500)

    key = _get_key(provider, api_key)
    final_system = (
        f"{system_prompt}\n\n"
        "Use only the supplied Tavily search results for web facts. "
        "Clearly state that the answer comes from online sources. "
        "Include source URLs when useful. Do not invent missing details."
    )
    final_user = _bounded(
        f"USER REQUEST:\n{user_prompt}\n\nTAVILY RESULTS:\n{evidence}",
        WEB_MAX_PROMPT_CHARS,
    )

    if provider == "Groq":
        client = Groq(api_key=key)
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": final_system},
                {"role": "user", "content": final_user},
            ],
            temperature=0.2,
            reasoning_effort="low",
            max_completion_tokens=WEB_MAX_COMPLETION,
        )
        return response.choices[0].message.content or "No web answer was returned."

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=key)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=final_user,
        config=types.GenerateContentConfig(
            system_instruction=final_system,
            temperature=0.2,
            max_output_tokens=WEB_MAX_COMPLETION,
        ),
    )
    return response.text or "No web answer was returned."

