"""LLM providers. Phase 8: Groq (primary). Phase 9 adds Ollama as the fallback in this same file.

Design rules
  * The API key comes from the environment (.env locally). It is never hardcoded or printed.
  * Every failure is turned into a short, user-safe category by `classify_llm_error`,
    so the rest of the app never shows stack traces or secrets.

Smoke test (needs a key in .env):
    python -m src.llm.providers --list    # which models can my key use?
    python -m src.llm.providers           # plain call + a real tool call
"""
import os
import re
import time

from dotenv import load_dotenv

from src import config

load_dotenv(config.ROOT_DIR / ".env")          # no error if the file does not exist

_KEY_PATTERN = re.compile(r"gsk_[A-Za-z0-9]+")


class MissingAPIKeyError(RuntimeError):
    """GROQ_API_KEY is not set (or still the placeholder)."""


def redact(text):
    """Remove anything that looks like a Groq key from text before it is shown or logged."""
    return _KEY_PATTERN.sub("gsk_***", str(text))


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


# ------------------------------------------------------------------ error handling
def classify_llm_error(exc):
    """exception -> (category, user_safe_message). Categories:
    missing_key | auth | rate_limit | model_not_found | network | other.
    Phase 9 uses the category to decide whether to fall back to Ollama."""
    if isinstance(exc, MissingAPIKeyError):
        return "missing_key", "The Groq API key is not configured."
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
    return "other", "The Groq request failed."


# ------------------------------------------------------------------ smoke test
def list_models():
    from groq import Groq
    return sorted(m.id for m in Groq(api_key=get_groq_api_key()).models.list().data)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="list the models your key can use")
    args = parser.parse_args()

    try:
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

        llm = get_groq_llm()
        print(f"Model: {get_groq_model_name()}")

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
        calls = msg.tool_calls
        if calls:
            print(f"2) Tool-calling OK ({time.time() - t0:.1f}s): the model asked to call {calls[0]['name']!r}")
            print("\nSUCCESS: this model can be used for tool-calling agents.")
        else:
            print("2) The model answered WITHOUT calling the tool. Try another model with GROQ_MODEL.")
            raise SystemExit(1)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        category, message = classify_llm_error(exc)
        print(f"\nFAILED [{category}]: {message}")
        print(f"Details: {type(exc).__name__}: {redact(exc)[:300]}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
