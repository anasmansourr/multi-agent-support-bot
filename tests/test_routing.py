import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src import config
from src.llm.providers import LLMProvider
from src.routing.router import Router, build_chain, decide
from tests.fakes import ScriptedChatModel, tool_call_msg


class FakePredictor:
    def __init__(self, intent="billing", confidence=0.95, boom=False, raw=None):
        self.intent, self.confidence, self.boom, self.raw = intent, confidence, boom, raw
        self.calls = []

    def predict(self, text):
        self.calls.append(text)
        if self.boom:
            raise RuntimeError("model exploded")
        return self.raw or {"intent": self.intent, "confidence": self.confidence}


def provider(model):
    return LLMProvider("groq", lambda: "fake-model", lambda: model)


def system_text(model):
    return model.seen[0][0].content


# ------------------------------------------------------------------ decide(): pure routing rule
@pytest.mark.parametrize("intent", config.INTENTS)
def test_each_intent_goes_to_its_own_agent(intent):
    d = decide("x", FakePredictor(intent, 0.9))
    assert d.selected_agent == intent and not d.fallback_applied and d.reason == "ok"


def test_low_confidence_goes_to_general_but_keeps_the_predicted_intent():
    d = decide("x", FakePredictor("billing", 0.40))
    assert d.selected_agent == "general" and d.predicted_intent == "billing"
    assert d.fallback_applied and d.reason == "low_confidence"


def test_threshold_boundary_is_strictly_below():
    assert decide("x", FakePredictor("technical", 0.60), threshold=0.60).selected_agent == "technical"
    assert decide("x", FakePredictor("technical", 0.5999), threshold=0.60).selected_agent == "general"


def test_threshold_is_configurable_in_one_place(monkeypatch):
    assert decide("x", FakePredictor("billing", 0.62)).selected_agent == "billing"        # default 0.60
    monkeypatch.setattr(config, "CONFIDENCE_THRESHOLD", 0.70)
    assert decide("x", FakePredictor("billing", 0.62)).selected_agent == "general"
    assert decide("x", FakePredictor("billing", 0.62), threshold=0.5).selected_agent == "billing"   # override


def test_low_confidence_general_is_not_reported_as_a_fallback():
    d = decide("x", FakePredictor("general", 0.30))
    assert d.selected_agent == "general" and not d.fallback_applied


@pytest.mark.parametrize("raw", [{"intent": "refunds", "confidence": 0.9}, {"intent": "billing", "confidence": float("nan")},
                                 {"intent": "billing", "confidence": 1.7}, {"confidence": 0.9}, {"intent": "billing"}])
def test_invalid_classifier_output_goes_to_general(raw):
    d = decide("x", FakePredictor(raw=raw))
    assert d.selected_agent == "general" and d.reason in ("invalid_output", "classifier_error")


def test_classifier_exception_does_not_crash():
    d = decide("x", FakePredictor(boom=True))
    assert d.selected_agent == "general" and d.reason == "classifier_error"


# ------------------------------------------------------------------ the LangChain chain
def test_chain_runs_the_agent_chosen_by_the_classifier_and_no_other():
    pred = FakePredictor("technical", 0.9)
    model = ScriptedChatModel(script=[AIMessage(content="Try restarting.")])
    out = build_chain(pred).invoke({"input": "the app crashes", "chat_history": [], "customer_id": "demo_customer",
                                    "providers": [provider(model)]})
    assert out["routing"].selected_agent == "technical" and out["agent"].intent == "technical"
    assert "Technical Support Agent" in system_text(model)
    assert set(model.seen_tool_names) == {"check_service_status"}
    assert len(model.seen) == 1                     # the LLM was called once, for the answer, NOT to classify


def test_low_confidence_message_runs_the_general_agent():
    pred = FakePredictor("billing", 0.30)
    model = ScriptedChatModel(script=[AIMessage(content="Could you tell me more?")])
    r = Router(pred).handle("hmm", providers=[provider(model)])
    assert r.routing.fallback_applied and r.agent.intent == "general"
    assert "General Agent" in system_text(model) and "Billing Agent" not in system_text(model)


def test_billing_route_really_calls_the_tool():
    model = ScriptedChatModel(script=[tool_call_msg("check_invoice", {"customer_id": "demo_customer"}),
                                      AIMessage(content="You owe $500.00.")])
    r = Router(FakePredictor("billing", 0.95)).handle("it says i owe 500", providers=[provider(model)])
    assert r.agent.tool_calls[0]["result"]["amount_due_usd"] == 500.0


def test_classifier_runs_exactly_once_per_message():
    pred = FakePredictor("general", 0.9)
    model = ScriptedChatModel(script=[AIMessage(content="hi")])
    Router(pred).handle("hello there", providers=[provider(model)])
    assert pred.calls == ["hello there"]            # classified exactly once


def test_history_is_passed_to_whichever_agent_is_selected():
    """Memory must survive a handoff: a technical turn after a billing turn still sees the billing history."""
    history = [HumanMessage("I have a billing problem."), AIMessage("Sure, what seems to be wrong?")]
    model = ScriptedChatModel(script=[AIMessage(content="ok")])
    Router(FakePredictor("technical", 0.9)).handle("the page shows an error", chat_history=history,
                                                   providers=[provider(model)])
    assert [m.content for m in model.seen[0][1:3]] == ["I have a billing problem.", "Sure, what seems to be wrong?"]


def test_all_providers_failing_returns_error_but_keeps_the_routing_info():
    r = Router(FakePredictor("billing", 0.95)).handle("hi", providers=[provider(ScriptedChatModel(script=[]))],
                                                      simulate_outage={"groq"})
    assert r.agent is None and "try again" in r.error.lower()
    assert r.routing.selected_agent == "billing"          # the sidebar can still show the routing
