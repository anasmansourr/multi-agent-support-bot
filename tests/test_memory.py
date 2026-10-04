import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src import config
from src.bot import SupportBot
from src.llm.providers import LLMProvider
from src.memory.history import ConversationMemory
from src.routing.router import Router, decide
from tests.fakes import ScriptedChatModel
from tests.test_routing import FakePredictor


# ------------------------------------------------------------------ the memory object
def test_messages_alternate_human_ai_in_order():
    m = ConversationMemory()
    m.add("I have a billing problem.", "Sure, what seems to be wrong?", "billing")
    m.add("It says I owe 500", "You owe $500.", "billing")
    msgs = m.messages()
    assert [type(x) for x in msgs] == [HumanMessage, AIMessage] * 2
    assert [x.content for x in msgs][:2] == ["I have a billing problem.", "Sure, what seems to be wrong?"]


def test_history_is_bounded_to_the_last_n_turns():
    m = ConversationMemory(max_turns=3)
    for i in range(7):
        m.add(f"q{i}", f"a{i}", "general")
    assert [t.user for t in m.turns] == ["q4", "q5", "q6"] and m.dropped == 4


def test_long_messages_are_clipped():
    m = ConversationMemory(max_chars=50)
    m.add("x" * 500, "y" * 500, "general")
    assert len(m.turns[0].user) <= 54 and m.turns[0].user.endswith("...")


def test_last_agent_and_clear():
    m = ConversationMemory()
    assert m.last_agent() is None
    m.add("a", "b", "billing")
    m.add("c", "d", "technical")
    assert m.last_agent() == "technical"
    m.clear()
    assert len(m) == 0 and m.messages() == [] and m.last_agent() is None


def test_summarizer_hook_condenses_dropped_turns_instead_of_forgetting_them():
    calls = []

    def summarizer(old_summary, dropped):
        calls.append([t.user for t in dropped])
        return (old_summary + " | " if old_summary else "") + "; ".join(t.user for t in dropped)

    m = ConversationMemory(max_turns=2, summarizer=summarizer)
    for q in ["one", "two", "three", "four"]:
        m.add(q, "ok", "general")
    assert m.summary == "one | two" and len(m) == 2
    first = m.messages()[0]
    assert isinstance(first, HumanMessage) and "one | two" in first.content   # summary leads the history


def test_a_failing_summarizer_never_breaks_the_chat():
    def boom(*a):
        raise RuntimeError("llm down")
    m = ConversationMemory(max_turns=1, summarizer=boom)
    m.add("a", "b", "general")
    m.add("c", "d", "general")
    assert len(m) == 1 and m.summary == ""


# ------------------------------------------------------------------ follow-up routing
def low(intent="general", conf=0.30):
    return FakePredictor(intent, conf)


def test_low_confidence_first_message_goes_to_general():
    d = decide("hmm", low("billing"), active_agent=None)
    assert d.selected_agent == "general" and d.reason == "low_confidence"


def test_low_confidence_followup_keeps_the_current_specialist():
    d = decide("yes please", low("general"), active_agent="feature_request")
    assert d.selected_agent == "feature_request" and d.reason == "low_confidence_kept_agent"


def test_confident_classification_still_switches_agent_that_is_the_handoff():
    d = decide("the page shows an error", FakePredictor("technical", 0.9), active_agent="billing")
    assert d.selected_agent == "technical" and d.reason == "ok"


def test_strict_mode_ignores_the_active_agent(monkeypatch):
    monkeypatch.setattr(config, "FOLLOWUP_STICKY", False)
    d = decide("yes please", low("general"), active_agent="feature_request")
    assert d.selected_agent == "general" and d.reason == "low_confidence"
    assert decide("x", low(), active_agent="billing", sticky=True).selected_agent == "billing"   # explicit override


def test_active_general_agent_is_not_sticky():
    assert decide("x", low("billing"), active_agent="general").selected_agent == "general"


def test_invalid_classifier_output_is_never_sticky():
    d = decide("x", FakePredictor(boom=True), active_agent="billing")
    assert d.selected_agent == "general" and d.reason == "classifier_error"


# ------------------------------------------------------------------ end to end with fake LLMs
def provider(model):
    return LLMProvider("groq", lambda: "fake", lambda: model)


class SequencePredictor:
    """Returns a different (intent, confidence) per call - a scripted conversation."""
    def __init__(self, seq):
        self.seq = list(seq)

    def predict(self, text):
        intent, conf = self.seq.pop(0)
        return {"intent": intent, "confidence": conf}


def test_memory_survives_an_agent_handoff():
    """Turn 1 handled by billing, turn 2 routed to technical: technical must still see turn 1."""
    billing_model = ScriptedChatModel(script=[AIMessage(content="Sure, what seems to be wrong with your payment?")])
    tech_model = ScriptedChatModel(script=[AIMessage(content="Let's check the app. What error do you see?")])
    bot = SupportBot(Router(SequencePredictor([("billing", 0.95), ("technical", 0.90)])))

    r1 = bot.ask("I have a billing problem with my payment", providers=[provider(billing_model)])
    r2 = bot.ask("The payment went through but the app shows an error", providers=[provider(tech_model)])

    assert (r1.routing.selected_agent, r2.routing.selected_agent) == ("billing", "technical")
    prompt = tech_model.seen[0]
    assert "Technical Support Agent" in prompt[0].content
    assert [m.content for m in prompt[1:3]] == ["I have a billing problem with my payment",
                                                "Sure, what seems to be wrong with your payment?"]
    assert prompt[-1].content == "The payment went through but the app shows an error"
    assert [t.agent for t in bot.memory.turns] == ["billing", "technical"]


def test_it_is_resolved_from_history_in_the_same_conversation():
    model = ScriptedChatModel(script=[AIMessage(content="Sure."), AIMessage(content="You owe $500.")])
    bot = SupportBot(Router(SequencePredictor([("billing", 0.95), ("billing", 0.62)])))
    bot.ask("I have a billing problem.", providers=[provider(model)])
    bot.ask("It says I owe 500", providers=[provider(model)])
    second_turn_messages = [m.content for m in model.seen[1]]
    assert "I have a billing problem." in second_turn_messages and "Sure." in second_turn_messages


def test_followup_keeps_the_feature_agent_through_a_low_confidence_yes():
    model = ScriptedChatModel(script=[AIMessage(content="Want me to log it?"), AIMessage(content="Logged.")])
    bot = SupportBot(Router(SequencePredictor([("feature_request", 0.99), ("general", 0.40)])))
    bot.ask("please add a linux app", providers=[provider(model)])
    r = bot.ask("yes please", providers=[provider(model)])
    assert r.routing.selected_agent == "feature_request" and r.routing.reason == "low_confidence_kept_agent"
    assert "Feature Request Agent" in model.seen[1][0].content


def test_failed_turn_is_not_remembered():
    bot = SupportBot(Router(FakePredictor("billing", 0.95)))
    r = bot.ask("hi", providers=[provider(ScriptedChatModel(script=[]))], simulate_outage={"groq"})
    assert r.agent is None and len(bot.memory) == 0


def test_reset_clears_the_conversation():
    model = ScriptedChatModel(script=[AIMessage(content="hello")])
    bot = SupportBot(Router(FakePredictor("general", 0.9)))
    bot.ask("hello", providers=[provider(model)])
    assert len(bot.memory) == 1
    bot.reset()
    assert len(bot.memory) == 0 and bot.memory.last_agent() is None
