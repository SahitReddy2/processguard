"""
Real LangGraph demo — non-synthetic ReAct agent with ProcessGuard attached.

This is the v0.1.1 validation demo. The agent is a standard LangGraph
ReAct loop (`langgraph.prebuilt.create_react_agent`) using Claude Haiku 4.5
and DuckDuckGo search. The task is open-ended on purpose — compare three
LLM techniques and recommend one for a specific use case — so we can see
how the detectors behave on a real, non-rigged run.

If DuckDuckGo fails on the first call (network error, rate limit, package
missing), the search tool transparently falls back to a small in-memory
knowledge base for the rest of the run. Per the v0.1.1 plan, *junk* results
from DDG are still treated as valid data (the agent might loop on them,
which is the thing we want to observe); the in-memory fallback only kicks
in on actual exceptions, not on low-quality results.

Requires:
    pip install processguard[langgraph] langchain-google-genai duckduckgo-search
    export GOOGLE_API_KEY=...

Run:
    python examples/real_langgraph_demo.py

Model note: this demo uses Gemini 2.5 Flash (free tier on Google AI Studio).
The two LLM-judge detectors (FM-2.6 ReasoningActionMismatch, FM-3.1
PrematureTermination) are wired to the Anthropic SDK and are disabled in
this run via llm_detectors=False — their behaviour is audited separately
in docs/judge_audit.md. The three rule-based detectors (StepRepetition,
UnawareTermination, NoProgressLoop) are framework-independent and run as
normal.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# ── halt if the API key is missing (do this before any imports that fan out) ─

if not os.getenv("GOOGLE_API_KEY"):
    sys.exit("GOOGLE_API_KEY not set; cannot run real LangGraph demo.")

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

import processguard
from processguard import PolicyAction


# ── search tool: DuckDuckGo with in-memory fallback ──────────────────────────

try:
    from duckduckgo_search import DDGS as _DDGSClass
    _DDGS_AVAILABLE = True
except ImportError:
    _DDGSClass = None
    _DDGS_AVAILABLE = False


_IN_MEMORY_KB = [
    {
        "title": "Retrieval-Augmented Generation: Overview",
        "text": (
            "RAG combines a retrieval system (typically vector search over a "
            "document corpus) with a generative LLM. At inference time, the "
            "system retrieves passages relevant to the user's query and "
            "includes them in the LLM's prompt as context. RAG is strong "
            "when the knowledge base is large, frequently updated, or "
            "domain-specific, because adding new information means "
            "re-indexing rather than retraining. Common failure modes are "
            "poor retrieval quality, irrelevant passages, and "
            "context-window overflow."
        ),
    },
    {
        "title": "Fine-tuning LLMs for Domain Tasks",
        "text": (
            "Fine-tuning trains a base LLM on a curated dataset of "
            "input-output pairs from the target domain. It bakes domain "
            "behavior into the model weights and produces fast, consistent "
            "outputs at inference time without needing retrieval. Costs "
            "include compute for training, dataset curation effort, and the "
            "need to re-train when domain knowledge updates. LoRA and "
            "QLoRA reduce training cost substantially by training small "
            "adapter layers rather than full weights."
        ),
    },
    {
        "title": "Prompt Engineering Techniques",
        "text": (
            "Prompt engineering uses careful instruction design — system "
            "prompts, few-shot examples, chain-of-thought scaffolding — to "
            "steer a frozen LLM. It has zero training cost, instant "
            "deployment, and full reversibility, but is limited by the "
            "base model's existing knowledge and instruction-following "
            "capability. Common patterns include role prompts ('you are an "
            "expert X'), output schemas, and example-driven prompting."
        ),
    },
    {
        "title": "RAG vs Fine-tuning: when to choose which",
        "text": (
            "Choose RAG when knowledge changes frequently, the corpus is "
            "large, you need source attribution, or you have many narrow "
            "domains. Choose fine-tuning when the task requires specific "
            "output formatting, latency matters more than freshness, or "
            "the domain knowledge is stable. Many production systems "
            "combine both — fine-tune the model for tone and formatting, "
            "then use RAG for factual lookups."
        ),
    },
    {
        "title": "Internal Documentation QA Systems",
        "text": (
            "For internal company documentation QA, three considerations "
            "dominate: documents change weekly or daily, the audience "
            "needs source citations for trust, and the corpus is too "
            "large to fit in any prompt. These properties make RAG the "
            "default architectural choice. Fine-tuning is rarely worth "
            "the operational cost for a documentation QA use case unless "
            "the company has thousands of historical Q&A pairs and an "
            "extreme latency requirement."
        ),
    },
]


def _in_memory_search(query: str) -> str:
    q_words = set(query.lower().split())
    scored = sorted(
        _IN_MEMORY_KB,
        key=lambda d: len(q_words & set(d["text"].lower().split())),
        reverse=True,
    )
    top = scored[:3]
    return "\n\n".join(f"[{d['title']}] {d['text']}" for d in top)


_search_calls = 0
_ddg_failed_once = False
_per_call_meta: list[dict] = []   # populated by web_search; copied into status JSON


@tool
def web_search(query: str) -> str:
    """Search the web for information on a topic."""
    global _search_calls, _ddg_failed_once
    _search_calls += 1
    print(f"  [tool] web_search({query!r})  call #{_search_calls}")

    if _DDGS_AVAILABLE and not _ddg_failed_once:
        try:
            with _DDGSClass() as ddgs:
                results = list(ddgs.text(query, max_results=3))
            _per_call_meta.append({
                "call":         _search_calls,
                "query":        query,
                "source":       "ddg",
                "ddg_results":  len(results),
                "result_titles": [r.get("title", "") for r in results],
            })
            if results:
                return "\n\n".join(
                    f"{r.get('title','')}: {r.get('body','')}"
                    for r in results
                )
            return "(no results)"
        except Exception as e:
            print(
                f"  [search] DDG failed ({type(e).__name__}: {e}); "
                "falling back to in-memory KB for the rest of this run"
            )
            _ddg_failed_once = True
            _per_call_meta.append({
                "call":      _search_calls,
                "query":     query,
                "source":    "in_memory_fallback",
                "ddg_error": f"{type(e).__name__}: {e}",
            })
            return _in_memory_search(query)

    _per_call_meta.append({
        "call":   _search_calls,
        "query":  query,
        "source": "in_memory_fallback",
    })
    return _in_memory_search(query)


# ── event capture: tee every event into a JSONL file ─────────────────────────

EVENT_LOG = Path(__file__).parent / "real_run_log.jsonl"


def _tee_emit(guard):
    log = EVENT_LOG.open("w", encoding="utf-8")
    _orig = guard._emit

    def emit(event):
        log.write(json.dumps({
            "ts":          event.timestamp,
            "trace_id":    event.trace_id,
            "span_id":     event.span_id,
            "event_type":  event.event_type.value,
            "agent_name":  event.agent_name,
            "tool_name":   event.tool_name,
            "tool_args":   event.tool_args,
            "tool_result": (event.tool_result[:1500] if event.tool_result else None),
            "content":     (event.content[:1500] if event.content else None),
        }) + "\n")
        log.flush()
        return _orig(event)

    guard._emit = emit
    return log


# ── status JSON: makes findings report unambiguous ───────────────────────────

STATUS_PATH = Path(__file__).parent / "real_run_status.json"


def _classify_exception(e: Exception) -> str:
    cls = type(e).__name__
    if cls == "GraphRecursionError" or "Recursion" in cls:
        return "recursion_limit"
    return "exception"


def _final_message_text(content) -> str:
    """LangChain message .content can be a string OR (for Gemini) a list of
    content blocks like [{'type': 'text', 'text': '...'}]. Always return a
    string so len() means 'characters of text', not 'number of blocks'."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for blk in content:
            if isinstance(blk, dict) and blk.get("type") == "text":
                parts.append(blk.get("text", ""))
            else:
                parts.append(str(blk))
        return "\n".join(p for p in parts if p)
    return str(content)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", max_output_tokens=1024)
    agent = create_react_agent(model=llm, tools=[web_search])

    guard = processguard.attach(
        agent,
        default_policy=PolicyAction.LOG,    # observe only — don't steer
        db_path=":memory:",
        verbose=True,
        llm_detectors=False,                # FM-2.6 + FM-3.1 use anthropic SDK; not in this run
    )

    log_handle = _tee_emit(guard)

    task = (
        "Compare retrieval-augmented generation, fine-tuning, and prompt "
        "engineering for a domain-specific question-answering system over "
        "internal company documentation. Recommend one approach with "
        "reasoning."
    )

    print("\n=== ProcessGuard real LangGraph demo ===")
    print(f"Task: {task}\n")

    final_state    = "completed"
    exc_type       = None
    exc_message    = None
    final_text     = None

    try:
        result = agent.invoke(
            {"messages": [HumanMessage(content=task)]},
            config={"recursion_limit": 20},
        )
        final_text = _final_message_text(result["messages"][-1].content)
        suffix = "…" if len(final_text) > 2000 else ""
        print(f"\n--- Final output ---\n{final_text[:2000]}{suffix}")
    except Exception as e:
        final_state = _classify_exception(e)
        exc_type    = type(e).__name__
        exc_message = str(e)
        print(f"\n[run ended] {exc_type}: {exc_message}")

    log_handle.close()

    print(f"\nTotal web_search calls: {_search_calls}")
    print(f"DDG fallback engaged:   {_ddg_failed_once}")
    print(f"\nDetections: {len(guard.policy.detections)}")
    for d in guard.policy.detections:
        print(f"  {d.failure_mode:12s} {d.failure_name:30s} conf={d.confidence:.2f}")

    # ── write status JSON ────────────────────────────────────────────────────
    status = {
        "task":                task,
        "model":               "gemini-2.5-flash",
        "llm_judge_detectors_enabled": False,
        "recursion_limit":     20,
        "final_state":         final_state,
        "exception_type":      exc_type,
        "exception_message":   exc_message,
        "ddg_available":       _DDGS_AVAILABLE,
        "ddg_failed":          _ddg_failed_once,
        "ddg_used":            _DDGS_AVAILABLE and any(c.get("source") == "ddg" for c in _per_call_meta),
        "total_search_calls":  _search_calls,
        "per_call":            _per_call_meta,
        "detections": [
            {
                "failure_mode": d.failure_mode,
                "failure_name": d.failure_name,
                "confidence":   d.confidence,
                "agent_name":   d.agent_name,
                "evidence":     d.evidence,
            }
            for d in guard.policy.detections
        ],
        "final_output_length": len(final_text) if final_text else None,
    }
    STATUS_PATH.write_text(json.dumps(status, indent=2), encoding="utf-8")

    print(f"\nEvent log:  {EVENT_LOG}")
    print(f"Status JSON: {STATUS_PATH}")


if __name__ == "__main__":
    main()
