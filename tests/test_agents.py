import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src import config
from src.agents.agents import build_agent, run_agent
from src.agents.prompts import ROLES, build_system_prompt
from src.agents.tools import TOOLS_BY_AGENT
from src.llm.providers import AllProvidersFailed, LLMProvider
from tests.fakes import ScriptedChatModel, tool_call_msg


def provider(name, model):
    return LLMProvider(name, lambda: f"{name}-model", lambda: model)


def billing_script():
    return [tool_call_msg("check_invoice", {"customer_id": "demo_customer"}),
            AIMessage(content="You currently owe $500.00 on invoice INV-2041, which is overdue.")]


def system_text(model):
    return model.seen[0][0].content


# ------------------------------------------------------------ the tool really runs
def test_billing_agent_really_executes_the_tool_and_answers_from_its_result():
    model = ScriptedChatModel(script=billing_script())
    r = run_agent("billing", "it says I owe 500", providers=[provider("groq", model)])
    assert [c["tool"] for c in r.tool_calls] == ["check_invoice"]
    assert r.tool_calls[0]["args"] == {"customer_id": "demo_customer"}
    assert r.tool_calls[0]["result"]["amount_due_usd"] == 500.0            # produced by the real function
    assert "500" in r.output and r.intent == "billing" and r.provider == "groq"
    # the model's 2nd turn received the tool result as a message
    second_prompt = " ".join(str(m.content) for m in model.seen[1])
    assert "INV-2041" in second_prompt


def test_feature_agent_tool_has_a_real_side_effect(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "FEATURE_REQUESTS_PATH", tmp_path / "fr.jsonl")
    model = ScriptedChatModel(script=[
        tool_call_msg("log_feature_request", {"title": "Linux app", "customer_id": "demo_customer"}),
        AIMessage(content="Logged as FR-0001.")])
    r = run_agent("feature_request", "please log a linux app request", providers=[provider("groq", model)])
    assert r.tool_calls[0]["result"]["ticket_id"] == "FR-0001"
    assert (tmp_path / "fr.jsonl").exists()


# ------------------------------------------------------------ each agent is different
def test_agents_have_different_prompts_and_only_their_own_tools():
    prompts = {i: build_system_prompt(i, [t.name for t in TOOLS_BY_AGENT[i]]) for i in config.INTENTS}
    assert len(set(prompts.values())) == 4 and set(ROLES) == set(config.INTENTS)
    assert "check_invoice" in prompts["billing"] and "check_invoice" not in prompts["technical"]
    assert "log_feature_request" in prompts["feature_request"] and "log_feature_request" not in prompts["billing"]
    assert "documentation does not provide enough information" in prompts["technical"]     # anti-hallucination


def test_the_model_is_only_offered_its_agents_tools():
    for intent, expected in {"billing": {"check_invoice", "check_refund_eligibility"},
                             "technical": {"check_service_status", "search_docs"},
                             "feature_request": {"check_roadmap", "log_feature_request", "search_docs"},
                             "general": {"search_docs"}}.items():
        model = ScriptedChatModel(script=[AIMessage(content="ok")])
        run_agent(intent, "hi", providers=[provider("groq", model)])
        assert set(model.seen_tool_names) == expected


def test_an_agent_cannot_use_a_tool_it_does_not_own(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "FEATURE_REQUESTS_PATH", tmp_path / "fr.jsonl")
    model = ScriptedChatModel(script=[tool_call_msg("log_feature_request", {"title": "sneaky"}),
                                      AIMessage(content="Sorry, I can't do that.")])
    r = run_agent("billing", "log a feature", providers=[provider("groq", model)])
    assert "not a valid tool" in str(r.tool_calls[0]["result"])
    assert not (tmp_path / "fr.jsonl").exists()                          # nothing was written


def test_general_agent_can_answer_without_using_its_tool():
    model = ScriptedChatModel(script=[AIMessage(content="Hello! How can I help?")])
    r = run_agent("general", "hello", providers=[provider("groq", model)])
    assert r.output == "Hello! How can I help?" and r.tool_calls == []


def test_unknown_intent_goes_to_general():
    model = ScriptedChatModel(script=[AIMessage(content="hi")])
    assert run_agent("nonsense", "x", providers=[provider("groq", model)]).intent == "general"


# ------------------------------------------------------------ context reaches the model
def test_customer_id_and_history_are_in_the_prompt():
    model = ScriptedChatModel(script=[AIMessage(content="ok")])
    history = [HumanMessage("I have a billing problem."), AIMessage("Sure, what seems to be wrong?")]
    run_agent("billing", "It says I owe 500", chat_history=history, customer_id="cust_paid",
              providers=[provider("groq", model)])
    sent = model.seen[0]
    assert "cust_paid" in system_text(model)
    assert [m.content for m in sent[1:3]] == ["I have a billing problem.", "Sure, what seems to be wrong?"]
    assert sent[-1].content == "It says I owe 500"


# ------------------------------------------------------------ fallback + failures
def test_groq_failure_falls_back_to_ollama_and_still_calls_the_tool():
    import groq
    req = httpx.Request("POST", "https://api.groq.com/x")

    class RateLimited(ScriptedChatModel):
        def _generate(self, *a, **k):
            raise groq.RateLimitError("slow", response=httpx.Response(429, request=req), body=None)

    ollama_model = ScriptedChatModel(script=billing_script())
    r = run_agent("billing", "what do I owe", providers=[provider("groq", RateLimited(script=[])),
                                                          provider("ollama", ollama_model)])
    assert r.provider == "ollama" and r.used_fallback and r.attempts[0]["category"] == "rate_limit"
    assert r.tool_calls[0]["tool"] == "check_invoice"


def test_both_providers_failing_raises_a_safe_error():
    with pytest.raises(AllProvidersFailed) as info:
        run_agent("billing", "hi", providers=[provider("groq", ScriptedChatModel(script=[])),
                                              provider("ollama", ScriptedChatModel(script=[]))],
                  simulate_outage={"groq", "ollama"})
    assert "try again" in info.value.user_message.lower()


def test_build_agent_returns_an_agent_executor_for_tool_agents():
    from langchain.agents import AgentExecutor
    assert isinstance(build_agent("billing", ScriptedChatModel(script=[])), AgentExecutor)


def test_prompts_forbid_inventing_details_and_claiming_unbacked_actions():
    from src.agents.prompts import COMMON_RULES
    assert "menu names" in COMMON_RULES and "Never claim you did something" in COMMON_RULES
    assert "Do not suggest workarounds" in COMMON_RULES
    assert "unless the documentation says so explicitly" in COMMON_RULES
    technical = build_system_prompt("technical", ["check_service_status"])
    assert "ONLY come from documentation" in technical
    assert "forwarded or submitted" in build_system_prompt("billing", [])


def test_prompts_keep_conversations_self_contained_and_hide_internals():
    from src.agents.prompts import COMMON_RULES
    assert "Never mention tool names" in COMMON_RULES
    assert "Do not call tools yet" in COMMON_RULES
    assert "Finish the job in one reply" in COMMON_RULES
    feature = build_system_prompt("feature_request", ["check_roadmap", "log_feature_request"])
    assert "IMMEDIATELY in this same turn" in feature and "Do NOT log a new request" in feature
    assert "Never offer updates" in COMMON_RULES and "keep you posted" in COMMON_RULES
