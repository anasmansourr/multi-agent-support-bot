import httpx
import pytest

from src.llm import providers
from src.llm.providers import (AllProvidersFailed, LLMProvider, MissingAPIKeyError, call_with_fallback,
                               classify_llm_error, get_ollama_llm, ollama_installed_models)


def ok_provider(name, answer):
    return LLMProvider(name, lambda: f"{name}-model", lambda: f"{name}-llm:{answer}")


def broken_provider(name, exc):
    def factory():
        raise exc
    return LLMProvider(name, lambda: f"{name}-model", factory)


def rate_limit():
    import groq
    req = httpx.Request("POST", "https://api.groq.com/x")
    return groq.RateLimitError("slow down gsk_SECRET123abc", response=httpx.Response(429, request=req), body=None)


def run(llm):
    return f"answered by {llm}"


def test_primary_used_when_it_works():
    r = call_with_fallback(run, [ok_provider("groq", 1), ok_provider("ollama", 2)])
    assert r.provider == "groq" and r.value == "answered by groq-llm:1" and not r.used_fallback


def test_rate_limit_falls_back_to_ollama():
    r = call_with_fallback(run, [broken_provider("groq", rate_limit()), ok_provider("ollama", 2)])
    assert r.provider == "ollama" and r.used_fallback
    assert r.attempts[0]["provider"] == "groq" and r.attempts[0]["category"] == "rate_limit"


def test_failure_while_RUNNING_also_falls_back():
    calls = []

    def flaky_run(llm):
        calls.append(llm)
        if llm.startswith("groq"):
            raise rate_limit()
        return "ok from ollama"

    r = call_with_fallback(flaky_run, [ok_provider("groq", 1), ok_provider("ollama", 2)])
    assert r.provider == "ollama" and len(calls) == 2


def test_missing_key_falls_back():
    r = call_with_fallback(run, [broken_provider("groq", MissingAPIKeyError()), ok_provider("ollama", 2)])
    assert r.provider == "ollama" and r.attempts[0]["category"] == "missing_key"


def test_simulated_outage_skips_groq_even_though_it_works():
    r = call_with_fallback(run, [ok_provider("groq", 1), ok_provider("ollama", 2)], simulate_outage={"groq"})
    assert r.provider == "ollama" and r.attempts[0]["category"] == "simulated_outage"


def test_both_fail_gives_safe_message_and_no_secret():
    with pytest.raises(AllProvidersFailed) as info:
        call_with_fallback(run, [broken_provider("groq", rate_limit()),
                                 broken_provider("ollama", httpx.ConnectError("refused"))])
    exc = info.value
    assert [a["provider"] for a in exc.attempts] == ["groq", "ollama"]
    assert exc.attempts[1]["category"] == "network"
    blob = str(exc.user_message) + repr(exc.attempts)
    assert "SECRET123abc" not in blob and "Traceback" not in blob
    assert "try again" in exc.user_message.lower()


def test_default_provider_order_is_groq_then_ollama():
    assert [p.name for p in providers.default_providers()] == ["groq", "ollama"]


def test_ollama_llm_built_from_env_without_network(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "tiny:1b")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://example.invalid:1234")
    llm = get_ollama_llm()
    assert llm.model == "tiny:1b" and llm.base_url == "http://example.invalid:1234"
    assert llm.keep_alive == "30m"


def test_ollama_error_classification():
    import ollama
    assert classify_llm_error(ollama.ResponseError("model not found", 404))[0] == "model_not_found"
    assert classify_llm_error(httpx.ConnectError("refused"))[0] == "network"
    assert classify_llm_error(ConnectionRefusedError())[0] == "network"


def test_ollama_installed_models_parses_server_reply(monkeypatch):
    fake = httpx.Response(200, json={"models": [{"name": "llama3.2:3b"}, {"name": "qwen2.5:3b"}]},
                          request=httpx.Request("GET", "http://x/api/tags"))
    monkeypatch.setattr(httpx, "get", lambda *a, **k: fake)
    assert ollama_installed_models() == ["llama3.2:3b", "qwen2.5:3b"]
