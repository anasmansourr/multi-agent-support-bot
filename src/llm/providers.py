"""LLM providers: Groq (primary) and Ollama (local fallback), plus the fallback logic.

Design rules
  * The Groq key comes from the environment (.env locally). Never hardcoded, never printed.
  * Every failure becomes a short, user-safe category via `classify_llm_error`
    (no stack traces and no secrets reach the UI).
  * `call_with_fallback(run)` is the ONE place that decides "try Groq, else Ollama".

Smoke tests:
    python -m src.llm.providers --list       # models my Groq key can use
    python -m src.llm.providers              # Groq: plain call + tool call
    python -m src.llm.providers --ollama     # local Ollama: server, model, plain call + tool call
    python -m src.llm.providers --fallback   # simulate a Groq outage -> Ollama must answer
"""
import os
import re
import time
from dataclasses import dataclass, field
from typing import Callable

from dotenv import load_dotenv

from src import config

load_dotenv(config.ROOT_DIR / ".env")          # no error if the file does not exist

_KEY_PATTERN = re.compile(r"gsk_[A-Za-z0-9]+")
USER_UNAVAILABLE_MESSAGE = ("I'm having trouble reaching the AI service right now. "
                            "Please try again in a minute.")


class MissingAPIKeyError(RuntimeError):
    """GROQ_API_KEY is not set (or still the placeholder)."""


class SimulatedOutageError(RuntimeError):
    """Raised on purpose by the demo 'simulate outage' switch."""


class AllProvidersFailed(RuntimeError):
    """Every provider failed. `.attempts` lists (provider, category, message) for the sidebar/logs."""

    def __init__(self, attempts):
        super().__init__(USER_UNAVAILABLE_MESSAGE)
        self.attempts = attempts
        self.user_message = USER_UNAVAILABLE_MESSAGE


def redact(text):
    """Remove anything that looks like a Groq key from text before it is shown or logged."""
    return _KEY_PATTERN.sub("gsk_***", str(text))


# ----------------------------------------------------------------------- Groq
def get_groq_api_key():
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key or key == "your_key_here":
        raise MissingAPIKeyError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and paste your key from "
            "console.groq.com (API Keys).")
    return key


def get_groq_model_name():
    return os.getenv("GROQ_MODEL", "").strip() or config.GROQ_DEFAULT_MODEL


def get_groq_llm(model=None, temperature=config.LLM_TEMPERATURE):
    """Build the LangChain chat model for Groq. Does not call the network."""
    from langchain_groq import ChatGroq
    return ChatGroq(model=model or get_groq_model_name(), temperature=temperature,
                    api_key=get_groq_api_key(), timeout=config.LLM_TIMEOUT_SECONDS,
                    max_retries=config.LLM_MAX_RETRIES)


# --------------------------------------------------------------------- Ollama
def get_ollama_model_name():
    return os.getenv("OLLAMA_MODEL", "").strip() or config.OLLAMA_DEFAULT_MODEL


def get_ollama_base_url():
    return os.getenv("OLLAMA_BASE_URL", "").strip() or config.OLLAMA_DEFAULT_BASE_URL


def get_ollama_llm(model=None, temperature=config.LLM_TEMPERATURE):
    """Build the LangChain chat model for a LOCAL Ollama server. Does not call the network."""
    from langchain_ollama import ChatOllama
    return ChatOllama(model=model or get_ollama_model_name(), base_url=get_ollama_base_url(),
                      temperature=temperature,
                      client_kwargs={"timeout": config.OLLAMA_TIMEOUT_SECONDS})


def ollama_installed_models():
    """Names of models installed in the local Ollama. Raises if the server is not running."""
    import httpx
    resp = httpx.get(get_ollama_base_url().rstrip("/") + "/api/tags", timeout=3)
    resp.raise_for_status()
    return sorted(m["name"] for m in resp.json().get("models", []))


# ------------------------------------------------------------------ error handling
def classify_llm_error(exc):
    """exception -> (category, user_safe_message). Categories:
    missing_key | auth | rate_limit | model_not_found | network | simulated_outage | other."""
    if isinstance(exc, MissingAPIKeyError):
        return "missing_key", "The Groq API key is not configured."
    if isinstance(exc, SimulatedOutageError):
        return "simulated_outage", "Simulated outage (demo switch)."
    try:
        import groq
    except ImportError:                                   # pragma: no cover
        groq = None
    if groq is not None:
        if isinstance(exc, groq.AuthenticationError):
            return "auth", "The Groq API key was rejected (invalid or revoked)."
        if isinstance(exc, groq.RateLimitError):
            return "rate_limit", "Groq rate limit reached."
        if isinstance(exc, groq.NotFoundError):
            return "model_not_found", "The configured Groq model was not found for this key."
        if isinstance(exc, (groq.APIConnectionError, groq.APITimeoutError)):
            return "network", "Could not reach Groq (network problem or timeout)."
        if isinstance(exc, groq.APIStatusError) and getattr(exc, "status_code", 0) >= 500:
            return "network", "Groq is temporarily unavailable."
    try:
        import ollama
        if isinstance(exc, ollama.ResponseError):
            if getattr(exc, "status_code", 0) == 404:
                return "model_not_found", "The Ollama model is not installed (run: ollama pull <model>)."
            return "other", "The local Ollama model returned an error."
    except ImportError:                                   # pragma: no cover
        pass
    try:
        import httpx
        if isinstance(exc, (httpx.ConnectError, httpx.TimeoutException)):
            return "network", "Could not reach the model server (is Ollama running?)."
    except ImportError:                                   # pragma: no cover
        pass
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return "network", "Could not reach the model server (is Ollama running?)."
    return "other", "The language-model request failed."


# ------------------------------------------------------------------- fallback logic
@dataclass
class LLMProvider:
    name: str
    model_name: Callable[[], str]
    factory: Callable[[], object]


@dataclass
class FallbackResult:
    value: object
    provider: str                                  # which provider produced the answer
    model: str
    attempts: list = field(default_factory=list)   # failures BEFORE the one that worked

    @property
    def used_fallback(self):
        return bool(self.attempts)


def default_providers():
    return [LLMProvider("groq", get_groq_model_name, get_groq_llm),
            LLMProvider("ollama", get_ollama_model_name, get_ollama_llm)]


def call_with_fallback(run, providers=None, simulate_outage=()):
    """Call `run(llm)` with the first provider that works.

    run             function taking a LangChain chat model and returning anything
                    (e.g. lambda llm: agent_executor_for(llm).invoke(...))
    simulate_outage provider names to fail ON PURPOSE (demo / tests), e.g. {"groq"}

    Returns FallbackResult. Raises AllProvidersFailed (with a user-safe message) if none work.
    Failures are recorded as categories only; raw error text is redacted."""
    attempts = []
    for provider in (providers if providers is not None else default_providers()):
        try:
            if provider.name in simulate_outage:
                raise SimulatedOutageError(provider.name)
            value = run(provider.factory())
            return FallbackResult(value, provider.name, provider.model_name(), attempts)
        except Exception as exc:  # noqa: BLE001  - any provider failure -> try the next one
            category, message = classify_llm_error(exc)
            attempts.append({"provider": provider.name, "category": category, "message": message,
                             "error_type": type(exc).__name__, "detail": redact(exc)[:200]})
    raise AllProvidersFailed(attempts)


# ------------------------------------------------------------------ smoke tests
def list_models():
    from groq import Groq
    return sorted(m.id for m in Groq(api_key=get_groq_api_key()).models.list().data)


def _smoke(llm, label):
    """Plain call + real tool call against one chat model."""
    t0 = time.time()
    reply = llm.invoke("Reply with exactly one word: pong")
    print(f"\n1) Plain call OK ({time.time() - t0:.1f}s): {str(reply.content).strip()[:80]!r}")

    from langchain_core.tools import tool

    @tool
    def get_secret_number() -> int:
        """Return the secret number. Always call this tool when asked for the secret number."""
        return 42

    t0 = time.time()
    msg = llm.bind_tools([get_secret_number]).invoke("What is the secret number? Use the tool.")
    if msg.tool_calls:
        print(f"2) Tool-calling OK ({time.time() - t0:.1f}s): asked to call {msg.tool_calls[0]['name']!r}")
        print(f"\nSUCCESS: {label} can be used for tool-calling agents.")
    else:
        print(f"2) {label} answered WITHOUT calling the tool - try another model.")
        raise SystemExit(1)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="list the models your Groq key can use")
    parser.add_argument("--ollama", action="store_true", help="test the local Ollama fallback")
    parser.add_argument("--fallback", action="store_true", help="simulate a Groq outage and use Ollama")
    args = parser.parse_args()

    try:
        if args.ollama:
            print(f"Ollama server: {get_ollama_base_url()} | model: {get_ollama_model_name()}")
            installed = ollama_installed_models()
            print(f"Installed models: {installed or 'none'}")
            wanted = get_ollama_model_name()
            if not any(n == wanted or n.split(":")[0] == wanted for n in installed):
                print(f"\nModel not installed. Run:  ollama pull {wanted}")
                raise SystemExit(1)
            _smoke(get_ollama_llm(), f"Ollama ({wanted})")
            return

        if args.fallback:
            print("Simulating a Groq outage...")
            result = call_with_fallback(lambda llm: llm.invoke("Reply with exactly one word: pong"),
                                        simulate_outage={"groq"})
            print(f"Answered by: {result.provider} ({result.model}) -> {str(result.value.content).strip()[:60]!r}")
            for a in result.attempts:
                print(f"  earlier failure: {a['provider']} [{a['category']}] {a['message']}")
            print("\nSUCCESS: the fallback chain works.")
            return

        get_groq_api_key()
        print("GROQ_API_KEY: found (value hidden)")
        if args.list:
            models = list_models()
            print(f"\nModels available to your key ({len(models)}):")
            for m in models:
                print("  ", m)
            print(f"\nCurrently configured model: {get_groq_model_name()}")
            print("Pick one that supports tool use and put GROQ_MODEL=<id> in .env if needed.")
            return
        print(f"Model: {get_groq_model_name()}")
        _smoke(get_groq_llm(), f"Groq ({get_groq_model_name()})")
    except SystemExit:
        raise
    except AllProvidersFailed as exc:
        print(f"\nFAILED: {exc.user_message}")
        for a in exc.attempts:
            print(f"  {a['provider']}: [{a['category']}] {a['message']}")
        raise SystemExit(1)
    except Exception as exc:  # noqa: BLE001
        category, message = classify_llm_error(exc)
        print(f"\nFAILED [{category}]: {message}")
        print(f"Details: {type(exc).__name__}: {redact(exc)[:300]}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
