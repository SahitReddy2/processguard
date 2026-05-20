import uuid
import pytest
from processguard.core.event import AgentEvent, EventType


def make_event(
    event_type: EventType,
    agent_name: str = "researcher",
    trace_id: str | None = None,
    tool_name: str | None = None,
    tool_args: dict | None = None,
    tool_result: str | None = None,
    content: str | None = None,
) -> AgentEvent:
    return AgentEvent(
        trace_id   = trace_id or "trace-test",
        span_id    = str(uuid.uuid4()),
        event_type = event_type,
        agent_name = agent_name,
        tool_name  = tool_name,
        tool_args  = tool_args,
        tool_result= tool_result,
        content    = content,
    )


def tool_call(query: str, tool: str = "web_search", trace_id: str = "trace-test") -> AgentEvent:
    return make_event(EventType.TOOL_CALL, tool_name=tool, tool_args={"query": query}, trace_id=trace_id)


def tool_result(text: str, trace_id: str = "trace-test") -> AgentEvent:
    return make_event(EventType.TOOL_RESULT, tool_result=text, trace_id=trace_id)
