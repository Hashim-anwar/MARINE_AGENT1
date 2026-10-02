"""MarineWise AI agent layer.

The app exposes two clear agent roles:
1. Troubleshooting Agent
2. Technical Training Agent

Important compatibility fix:
The previous version executed CrewAI through LiteLLM with Groq. That path can
send a `cache_breakpoint` field that the Groq Chat Completions endpoint rejects.
The agents below therefore use the official Groq Python client directly.
The role prompts and routing remain separated as two agents, but the fragile
CrewAI/LiteLLM runtime dependency is removed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from groq import Groq

MODEL = "openai/gpt-oss-120b"
MAX_PROMPT_CHARS = 22000
MAX_COMPLETION_TOKENS = 1800


@dataclass(frozen=True)
class MarineAgent:
    name: str
    role: str
    goal: str
    backstory: str


TROUBLESHOOTING_AGENT = MarineAgent(
    name="Troubleshooting Agent",
    role="Marine Engine Troubleshooting Specialist",
    goal="Answer marine engine troubleshooting questions from supplied manual evidence only.",
    backstory=(
        "A careful senior marine technician. Never invent manual-specific facts "
        "and always preserve source file and page references."
    ),
)

TRAINING_AGENT = MarineAgent(
    name="Training Agent",
    role="Marine Technical Training Specialist",
    goal=(
        "Create, explain, assess, and remediate marine technician training using "
        "supplied manual evidence whenever available."
    ),
    backstory=(
        "An experienced marine technical instructor and assessor. Explain systems "
        "step by step, create quizzes, score assessments against supplied keys, "
        "identify learning gaps, and produce practical remedial training."
    ),
)


def _client() -> Groq:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not configured.")
    return Groq(api_key=key)


def make_troubleshooting_agent() -> MarineAgent:
    return TROUBLESHOOTING_AGENT


def make_training_agent() -> MarineAgent:
    return TRAINING_AGENT


def run_marine_agent(agent_kind: str, system_prompt: str, user_prompt: str) -> str:
    """Run one MarineWise agent using the official Groq Python client.

    This deliberately avoids the CrewAI/LiteLLM adapter because Groq currently
    rejects the `cache_breakpoint` field that that adapter may add to requests.
    """
    agents = {
        "troubleshooting": make_troubleshooting_agent,
        "training": make_training_agent,
    }
    if agent_kind not in agents:
        raise ValueError(f"Unknown agent kind: {agent_kind}")

    agent = agents[agent_kind]()
    full_system = (
        f"You are the MarineWise {agent.name}.\n"
        f"ROLE: {agent.role}\n"
        f"GOAL: {agent.goal}\n"
        f"BACKGROUND: {agent.backstory}\n\n"
        f"{system_prompt}"
    )

    # Keep prompts bounded. A long PDF can otherwise push the request beyond
    # the practical request/token limits even though the model has a large
    # theoretical context window.
    if len(user_prompt) > MAX_PROMPT_CHARS:
        user_prompt = user_prompt[:MAX_PROMPT_CHARS] + "\n[Manual context truncated for safety.]"

    response = _client().chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": full_system},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        reasoning_effort="low",
        max_completion_tokens=MAX_COMPLETION_TOKENS,
    )
    return response.choices[0].message.content or "No answer was returned."
