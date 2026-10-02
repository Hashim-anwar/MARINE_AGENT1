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

from groq import Groq
import requests

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
TAVILY_MAX_RESULTS = 5
TAVILY_RESULT_CHARS = 700


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


def _get_tavily_key(api_key: str | None = None) -> str:
    key = api_key or os.getenv("TAVILY_API_KEY")
    if not key:
        raise RuntimeError("TAVILY_API_KEY is not configured.")
    return key


def _tavily_search(query: str, tavily_key: str) -> list[dict[str, str]]:
    response = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": tavily_key,
            "query": query,
            "topic": "general",
            "search_depth": "basic",
            "max_results": TAVILY_MAX_RESULTS,
            "include_answer": False,
            "include_raw_content": False,
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    return [
        {
            "title": str(item.get("title", "Untitled")),
            "url": str(item.get("url", "")),
            "content": str(item.get("content", ""))[:TAVILY_RESULT_CHARS],
        }
        for item in payload.get("results", [])
        if item.get("url")
    ]


def run_web_search(
    provider: str,
    system_prompt: str,
    user_prompt: str,
    api_key: str | None = None,
    tavily_api_key: str | None = None,
) -> str:
    """Search the public web with Tavily, then synthesize with Groq or Gemini.

    Tavily is deliberately a separate web-search layer. It is only called by
    workflows that explicitly request online research; it does not alter the
    local FAISS/manual RAG index.
    """
    if provider not in {"Groq", "Gemini"}:
        raise ValueError("Provider must be Groq or Gemini.")

    model_key = _get_key(provider, api_key)
    search_key = _get_tavily_key(tavily_api_key)
    query = _bounded(user_prompt, WEB_MAX_PROMPT_CHARS)
    results = _tavily_search(query, search_key)
    if not results:
        return "Tavily did not return any web results for this query."

    evidence_parts = []
    for i, item in enumerate(results, 1):
        evidence_parts.append(
            f"SOURCE {i}\nTITLE: {item['title']}\nURL: {item['url']}\n"
            f"SNIPPET: {item['content']}"
        )
    web_evidence = _bounded("\n\n".join(evidence_parts), 4200)

    synthesis_system = (
        f"{system_prompt}\n\n"
        "Use ONLY the Tavily search results supplied below for web-specific claims. "
        "Do not claim that you opened or verified a page beyond the supplied snippets. "
        "Mention the source URLs when useful. Clearly label the result as web-sourced. "
    )
    synthesis_user = (
        f"USER REQUEST:\n{query}\n\n"
        f"TAVILY SEARCH RESULTS:\n{web_evidence}"
    )

    if provider == "Groq":
        return _run_groq(
            synthesis_system,
            synthesis_user,
            model_key,
            WEB_MAX_COMPLETION,
        )
    return _run_gemini(
        synthesis_system,
        synthesis_user,
        model_key,
        WEB_MAX_COMPLETION,
    )

