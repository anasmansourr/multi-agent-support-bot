"""Headless end-to-end tests of app.py using Streamlit's AppTest runner and a fake bot (no LLM, no models)."""
import pytest
from streamlit.testing.v1 import AppTest

from src import config
from src.agents.agents import AgentResult
from src.routing.router import RouterResult, RoutingDecision
from src.ui_helpers import message_caption, new_usage, record_usage
from tests.fakes import FakeBot


APP = str(config.ROOT_DIR / "app.py")


@pytest.fixture
def app():
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["bot"] = FakeBot()
    at.run()
    assert not at.exception, at.exception
    return at


def send(at, text):
    at.chat_input[0].set_value(text).run()
    assert not at.exception, at.exception
    return at


def sidebar_text(at):
    parts = [m.value for m in at.sidebar.markdown] + [c.value for c in at.sidebar.caption]
    parts += [m.value for m in at.sidebar.metric] + [w.value for w in at.sidebar.warning] + [e.value for e in at.sidebar.error]
    parts += [i.value for i in at.sidebar.info]
    return " | ".join(str(p) for p in parts)


def test_initial_page(app):
    assert app.title[0].value == "FlowDesk Support Assistant"
    assert len(app.chat_message) == 0
    assert "Send a message" in sidebar_text(app)
    assert app.chat_input[0].placeholder == "Type your message..."


def test_message_shows_in_chat_and_routing_in_sidebar(app):
    send(app, "It says I owe 500")
    assert [m.name for m in app.chat_message] == ["user", "assistant"]
    assert "You owe $500.00" in app.chat_message[1].markdown[0].value
    metrics = {m.label: m.value for m in app.sidebar.metric}
    assert metrics["Predicted intent"] == "billing" and metrics["Confidence"] == "0.930"
    assert metrics["Selected agent"] == "Billing Agent"
    text = sidebar_text(app)
    assert "groq (fake-model)" in text and "check_invoice" in text
    assert "Billing Agent" in app.chat_message[1].caption[0].value


def test_conversation_history_persists_across_turns(app):
    send(app, "I have a billing problem.")
    send(app, "It says I owe 500")
    assert len(app.chat_message) == 4
    assert [m.markdown[0].value for m in app.chat_message if m.name == "user"] == \
        ["I have a billing problem.", "It says I owe 500"]


def test_low_confidence_is_visible(app):
    send(app, "uploading a photo gives me a red banner saying failed")
    metrics = {m.label: m.value for m in app.sidebar.metric}
    assert metrics["Predicted intent"] == "billing" and metrics["Selected agent"] == "General Agent"
    assert metrics["Confidence"] == "0.321"
    assert "below the threshold" in sidebar_text(app)


def test_unknown_words_are_explained(app):
    send(app, "asdf qwerty")
    text = sidebar_text(app)
    assert "unknown words" in text.lower() and "Known words: 0%" in text


def test_outage_toggle_forces_the_fallback_and_shows_it(app):
    app.sidebar.toggle[0].set_value(True).run()
    assert any("outage simulation is ON" in w.value for w in app.sidebar.warning)
    send(app, "It says I owe 500")
    assert app.session_state["bot"].calls[-1]["simulate_outage"] == {"groq"}
    text = sidebar_text(app)
    assert "ollama (fake-model)" in text and "Fallback used" in text and "groq: simulated_outage" in text
    assert "ollama (fallback)" in app.chat_message[1].caption[0].value


def test_both_outage_toggles_show_the_friendly_error(app):
    app.sidebar.toggle[0].set_value(True).run()
    app.sidebar.toggle[1].set_value(True).run()
    assert any("Both providers are simulated DOWN" in w.value for w in app.sidebar.warning)
    send(app, "It says I owe 500")
    assert app.session_state["bot"].calls[-1]["simulate_outage"] == {"groq", "ollama"}
    assert "try again" in app.chat_message[1].markdown[0].value.lower()
    assert "Errors **1**" in sidebar_text(app)


def test_ollama_toggle_alone_changes_nothing_visible(app):
    app.sidebar.toggle[1].set_value(True).run()
    send(app, "It says I owe 500")
    assert app.session_state["bot"].calls[-1]["simulate_outage"] == {"ollama"}
    assert "You owe $500.00" in app.chat_message[1].markdown[0].value


def test_threshold_and_customer_are_passed_to_the_bot(app):
    app.sidebar.slider[0].set_value(0.8).run()
    app.sidebar.selectbox[0].set_value("cust_paid").run()
    send(app, "hello")
    call = app.session_state["bot"].calls[-1]
    assert call["threshold"] == pytest.approx(0.8) and call["customer_id"] == "cust_paid"


def test_failure_shows_a_friendly_message_never_a_traceback(app):
    send(app, "boom")
    assert "try again" in app.chat_message[1].markdown[0].value.lower()
    assert any("try again" in e.value.lower() for e in app.sidebar.error)
    assert not app.exception
    assert "Errors **1**" in sidebar_text(app)


def test_usage_counter(app):
    send(app, "It says I owe 500")
    app.sidebar.toggle[0].set_value(True).run()
    send(app, "It says I owe 500")
    send(app, "boom")
    text = sidebar_text(app)
    assert "Messages **3**" in text and "Groq **1**" in text and "Ollama **1**" in text
    assert "Fallbacks **1**" in text and "Tool calls **2**" in text and "Errors **1**" in text


def test_reset_clears_chat_sidebar_and_memory(app):
    send(app, "It says I owe 500")
    reset = next(b for b in app.sidebar.button if b.label == "Reset conversation")
    reset.click().run()
    assert len(app.chat_message) == 0 and "Send a message" in sidebar_text(app)
    assert app.session_state["bot"].resets == 1 and "Messages **0**" in sidebar_text(app)


def test_demo_button_sends_its_message(app):
    button = next(b for b in app.sidebar.button if b.label == "please add dark mode")
    button.click().run()
    assert app.session_state["bot"].calls[-1]["message"] == "please add dark mode"
    assert app.chat_message[0].markdown[0].value == "please add dark mode"


def test_missing_classifier_shows_a_clear_instruction(monkeypatch):
    from src.classifier.inference import IntentPredictor

    def missing(self, *a, **k):
        raise FileNotFoundError("no checkpoint")
    monkeypatch.setattr(IntentPredictor, "__init__", missing)
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert any("python -m src.classifier.train" in e.value for e in at.error)
    assert len(at.chat_input) == 0                       # the page stopped; no half-working chat


# ------------------------------------------------------------------ pure helpers
def test_usage_helper_counts_everything():
    u = new_usage()
    ok = RouterResult(RoutingDecision("billing", .9, "billing", False, "ok"),
                      AgentResult("x", "billing", [{"tool": "t"}, {"tool": "u"}], "ollama", "m", used_fallback=True))
    record_usage(u, ok)
    record_usage(u, RouterResult(RoutingDecision("x", 0, "general", False, "ok"), None, "err"))
    assert u == {"messages": 2, "groq": 0, "ollama": 1, "fallbacks": 1, "tool_calls": 2, "errors": 1}
    assert message_caption(ok) == "Billing Agent · ollama (fallback) · tools: t, u"
