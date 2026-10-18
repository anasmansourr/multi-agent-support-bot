import os
import pathlib
import re

import httpx
import pytest

from src import config
from src.llm import providers
from src.llm.providers import MissingAPIKeyError, classify_llm_error, get_groq_llm, redact

ROOT = pathlib.Path(config.ROOT_DIR)


def _status_error(cls, code):
    req = httpx.Request("POST", "https://api.groq.com/x")
    return cls("boom", response=httpx.Response(code, request=req), body=None)


def test_missing_key_raises_clear_error(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError, match=r"\.env"):
        get_groq_llm()


def test_placeholder_key_counts_as_missing(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "your_key_here")
    with pytest.raises(MissingAPIKeyError):
        get_groq_llm()


def test_llm_is_built_from_env_without_leaking_the_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_FAKEKEY123456")
    monkeypatch.setenv("GROQ_MODEL", "some/model")
    llm = get_groq_llm()
    assert llm.model_name == "some/model" and llm.temperature < 0.01   # ChatGroq turns 0 into 1e-8 (Groq rejects an exact 0)
    assert "FAKEKEY123456" not in repr(llm) and "FAKEKEY123456" not in str(llm)


def test_default_model_used_when_env_not_set(monkeypatch):
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    assert providers.get_groq_model_name() == config.GROQ_DEFAULT_MODEL


def test_error_classification():
    import groq
    assert classify_llm_error(MissingAPIKeyError())[0] == "missing_key"
    assert classify_llm_error(_status_error(groq.RateLimitError, 429))[0] == "rate_limit"
    assert classify_llm_error(_status_error(groq.AuthenticationError, 401))[0] == "auth"
    assert classify_llm_error(_status_error(groq.NotFoundError, 404))[0] == "model_not_found"
    assert classify_llm_error(_status_error(groq.InternalServerError, 500))[0] == "network"
    assert classify_llm_error(groq.APIConnectionError(request=httpx.Request("POST", "https://x")))[0] == "network"
    assert classify_llm_error(ValueError("weird"))[0] == "other"


def test_user_messages_never_contain_keys_or_tracebacks():
    for exc in (MissingAPIKeyError(), ValueError("gsk_SECRET123")):
        assert "gsk_" not in classify_llm_error(exc)[1]
    assert redact("Authorization failed for gsk_abcDEF123") == "Authorization failed for gsk_***"


# ----------------------------------------------------------- security guards
def test_env_file_is_gitignored_and_example_has_only_a_placeholder():
    assert ".env" in [line.strip() for line in (ROOT / ".gitignore").read_text().splitlines()]
    example = (ROOT / ".env.example").read_text()
    assert "GROQ_API_KEY=your_key_here" in example and "gsk_" not in example


def test_no_real_looking_groq_key_anywhere_in_the_repo():
    pattern = re.compile(r"gsk_[A-Za-z0-9]{20,}")
    for folder in ("src", "tests", "docs"):
        for path in (ROOT / folder).rglob("*"):
            if path.suffix in {".py", ".md", ".txt", ".json", ".csv"}:
                assert not pattern.search(path.read_text(encoding="utf-8", errors="ignore")), path


# ----------------------------------------------------------- live (opt-in, uses quota)
live = pytest.mark.skipif(os.getenv("RUN_LIVE_TESTS") != "1", reason="set RUN_LIVE_TESTS=1 to call Groq")


@live
def test_live_plain_call():
    assert str(get_groq_llm().invoke("Reply with exactly one word: pong").content).strip()
