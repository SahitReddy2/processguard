"""
CrewAI demo — FM-1.3 step repetition + no-progress loop caught at runtime.

Requires:
    pip install processguard[crewai]
    export ANTHROPIC_API_KEY=...

Run:
    python examples/crewai_demo.py
"""

from __future__ import annotations

from crewai import Agent, Task, Crew
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

import processguard
from processguard import PolicyAction, PolicyConfig


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

    guard = processguard.attach(
        crew,
        default_policy=PolicyAction.STEER,
        db_path=":memory:",
    )

    print("\n=== ProcessGuard CrewAI Demo ===")
    print("Task: Research RAG architectures in 2026\n")

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
