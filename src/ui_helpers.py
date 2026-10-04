"""Pure helpers for the Streamlit UI (no streamlit import, so they are easy to test)."""

AGENT_NAMES = {"billing": "Billing Agent", "technical": "Technical Agent",
               "feature_request": "Feature Request Agent", "general": "General Agent"}

REASON_TEXT = {
    "ok": "Confident: sent to the agent for the predicted intent.",
    "low_confidence": "Confidence is below the threshold: sent to the General agent.",
    "low_confidence_kept_agent": "Confidence is below the threshold, mid-conversation: kept the current agent.",
    "unknown_words": "Mostly unknown words, so the confidence is not trusted: sent to the General agent.",
    "classifier_error": "The classifier failed: sent to the General agent.",
    "invalid_output": "The classifier returned an invalid result: sent to the General agent.",
    "empty_input": "Empty message.",
}


def new_usage():
    return {"messages": 0, "groq": 0, "ollama": 0, "fallbacks": 0, "tool_calls": 0, "errors": 0}


def record_usage(usage, result):
    """Update the session counters from one RouterResult."""
    usage["messages"] += 1
    if result.agent is None:
        usage["errors"] += 1
        return
    a = result.agent
    usage[a.provider] = usage.get(a.provider, 0) + 1
    usage["fallbacks"] += bool(a.used_fallback)
    usage["tool_calls"] += len(a.tool_calls)


def provider_label(agent_result):
    return f"{agent_result.provider} ({agent_result.model})"


def attempts_text(attempts):
    """'groq: rate_limit' style summary of providers that failed before the one that answered."""
    return ", ".join(f"{a['provider']}: {a['category']}" for a in attempts)


def message_caption(result):
    """Small caption shown under each assistant message."""
    d = result.routing
    parts = [AGENT_NAMES.get(d.selected_agent, d.selected_agent)]
    if result.agent is not None:
        parts.append(result.agent.provider + (" (fallback)" if result.agent.used_fallback else ""))
        if result.agent.tool_calls:
            parts.append("tools: " + ", ".join(c["tool"] for c in result.agent.tool_calls))
    return " · ".join(parts)
