"""
EXPERIMENTAL — CrewAI demo. Not part of the v1 supported surface.

The CrewAI adapter at processguard.experimental.crewai does NOT reliably
capture tool-call events on current CrewAI versions (it was written against
the older LangChain-style step_callback shape). As a result, this demo will
likely produce zero detections even though the agent loops — which is the
honest reason CrewAI is deferred to v1.1.

This file is kept in `experimental/` for two reasons:
  1. As scaffolding for whoever picks up the CrewAI rewrite in v1.1.
  2. As a concrete reproducer for the gap: run this and see for yourself
     that the detectors do not fire on a CrewAI crew that is clearly stuck.

Requires:
    pip install processguard[experimental-crewai]
    export ANTHROPIC_API_KEY=...

Run:
    python experimental/crewai_demo.py
"""

from __future__ import annotations

from crewai import Agent, Task, Crew
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from processguard import ProcessGuard, PolicyAction
from processguard.experimental.crewai import CrewAIAdapter


# ── fake tool that never makes progress ───────────────────────────────────────

class SearchInput(BaseModel):
    query: str = Field(description="Search query")


class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = "Search the web."
    args_schema: type[BaseModel] = SearchInput

    def _run(self, query: str) -> str:
        print(f"  [tool] web_search({query!r})")
        return "Generic search results. No new information found. Consider refining your query."


# ── crew ──────────────────────────────────────────────────────────────────────

def build_crew():
    search = WebSearchTool()

    researcher = Agent(
        role="Senior Researcher",
        goal="Find detailed information about RAG architectures",
        backstory="Expert at finding information online.",
        tools=[search],
        llm="claude-haiku-4-5-20251001",
        verbose=False,
        max_iter=15,
    )

    task = Task(
        description="Research RAG architectures in 2026. Summarise the state of the art.",
        expected_output="A 3-paragraph summary of RAG architectures.",
        agent=researcher,
    )

    return Crew(agents=[researcher], tasks=[task], verbose=False)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    crew = build_crew()

    # processguard.attach() rejects CrewAI in v1 — wire the experimental
    # adapter manually instead.
    guard = ProcessGuard(default_policy=PolicyAction.STEER, db_path=":memory:")
    CrewAIAdapter(guard).attach(crew)

    print("\n=== ProcessGuard CrewAI Demo (experimental) ===")
    print("Task: Research RAG architectures in 2026\n")
    print("NOTE: This adapter is known not to capture tool-call events on")
    print("current CrewAI versions. Expect zero detections even if the crew")
    print("clearly loops. This is the gap that v1.1 will close.\n")

    try:
        result = crew.kickoff(inputs={"topic": "RAG architectures 2026"})
        print(f"\n--- Final output ---\n{result}")
    except Exception as e:
        print(f"\n[halted] {e}")

    print(f"\nDetections: {len(guard.policy.detections)}")
    for d in guard.policy.detections:
        print(f"  {d.failure_mode} {d.failure_name} (conf={d.confidence:.2f})")


if __name__ == "__main__":
    main()
