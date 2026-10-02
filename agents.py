"""CrewAI agent layer for MarineWise AI.

MarineWise uses two clear agents in the same app:
1. Troubleshooting Agent
2. Technical Training Agent

The Technical Training Agent handles all three training workflows: training
material, quiz generation, and assessment scoring/remedial training.

The Groq Python client remains the direct client used by the app for the
optional web-search fallback.

GROQ_API_KEY is supplied at runtime by app.py from Streamlit Secrets or the
environment. Nothing is hardcoded here.
"""

from __future__ import annotations

import os

from crewai import Agent, Crew, Task

MODEL = "groq/openai/gpt-oss-120b"


def _check_key() -> None:
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is not configured.")


def make_troubleshooting_agent() -> Agent:
    return Agent(
        role="Marine Engine Troubleshooting Specialist",
        goal="Answer marine engine troubleshooting questions from supplied manual evidence only.",
        backstory=(
            "A careful senior marine technician. You never invent manual-specific "
            "facts and always preserve source file and page references."
        ),
        llm=MODEL,
        verbose=False,
        allow_delegation=False,
    )


def make_training_agent() -> Agent:
    return Agent(
        role="Marine Technical Training Agent",
        goal=(
            "Create, explain, assess, and remediate marine technician training "
            "using supplied manual evidence whenever available."
        ),
        backstory=(
            "An experienced marine technical instructor and assessor. You explain "
            "systems step by step, create clear quizzes, score assessments against "
            "provided keys, identify learning gaps, and produce practical remedial "
            "training. You never invent manual-specific values."
        ),
        llm=MODEL,
        verbose=False,
        allow_delegation=False,
    )


def run_marine_agent(
    agent_kind: str,
    system_prompt: str,
    user_prompt: str,
) -> str:
    """Run one CrewAI agent with one focused task."""
    _check_key()

    factories = {
        "troubleshooting": make_troubleshooting_agent,
        "training": make_training_agent,
    }
    if agent_kind not in factories:
        raise ValueError(f"Unknown agent kind: {agent_kind}")

    agent = factories[agent_kind]()
    task = Task(
        description=f"{system_prompt}\n\nUSER REQUEST:\n{user_prompt}",
        expected_output="A concise, technically careful answer following the instructions exactly.",
        agent=agent,
    )
    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=False,
    )
    result = crew.kickoff()
    return str(result)
