"""A scripted chat model for tests: no network, but it behaves like a tool-calling LLM."""
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field, PrivateAttr


def tool_call_msg(name, args, call_id="call_1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


class ScriptedChatModel(BaseChatModel):
    """Returns the scripted AIMessages in order (the last one repeats). Records every prompt it saw."""
    script: list = Field(default_factory=list)
    seen: list = Field(default_factory=list)
    seen_tool_names: list = Field(default_factory=list)
    _i: int = PrivateAttr(default=0)

    @property
    def _llm_type(self):
        return "scripted"

    def bind_tools(self, tools, **kwargs):          # required by create_tool_calling_agent
        self.seen_tool_names = [t.name for t in tools]
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(messages)
        msg = self.script[min(self._i, len(self.script) - 1)]
        self._i += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])


class FakeBot:
    """Stands in for SupportBot in UI tests: scripted routing results, records how it was called."""

    def __init__(self):
        from src.memory.history import ConversationMemory
        self.memory = ConversationMemory()
        self.calls = []
        self.resets = 0

    def ask(self, message, customer_id="demo_customer", threshold=None, providers=None, simulate_outage=()):
        from src.agents.agents import AgentResult
        from src.routing.router import RouterResult, RoutingDecision
        self.calls.append({"message": message, "customer_id": customer_id, "threshold": threshold,
                           "simulate_outage": set(simulate_outage)})
        text = message.lower()
        if {"groq", "ollama"} <= set(simulate_outage):
            return RouterResult(RoutingDecision("billing", 0.93, "billing", False, "ok", threshold=0.6, known_ratio=1.0), None,
                                "I'm having trouble reaching the AI service right now. Please try again in a minute.")
        if "boom" in text:
            return RouterResult(RoutingDecision("billing", 0.9, "billing", False, "ok"), None,
                                "I'm having trouble reaching the AI service right now. Please try again in a minute.")
        if "asdf" in text:
            decision = RoutingDecision("general", 0.875, "general", False, "unknown_words", threshold=0.6, known_ratio=0.0)
            agent = AgentResult("I'm not sure I understand. What do you need?", "general", [], "groq", "fake-model")
        elif "banner" in text:
            decision = RoutingDecision("billing", 0.321, "general", True, "low_confidence", threshold=0.6, known_ratio=0.5)
            agent = AgentResult("That banner usually means the file is too large.", "general", [], "groq", "fake-model")
        else:
            decision = RoutingDecision("billing", 0.93, "billing", False, "ok", threshold=0.6, known_ratio=1.0)
            calls = [{"tool": "check_invoice", "args": {"customer_id": customer_id},
                      "result": {"amount_due_usd": 500.0}}]
            fallback = "groq" in simulate_outage
            agent = AgentResult("You owe $500.00 on invoice INV-2041.", "billing", calls,
                                "ollama" if fallback else "groq", "fake-model", used_fallback=fallback,
                                attempts=[{"provider": "groq", "category": "simulated_outage"}] if fallback else [])
        self.memory.add(message, agent.output, decision.selected_agent)
        return RouterResult(decision, agent)

    def reset(self):
        self.resets += 1
        self.memory.clear()
