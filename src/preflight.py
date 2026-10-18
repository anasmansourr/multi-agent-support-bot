"""Pre-recording check: verifies every component of the demo and WARMS UP the slow ones.

    python -m src.preflight            # everything (uses a few tokens of Groq quota; warms up Ollama)
    python -m src.preflight --no-llm   # only local checks (classifier, docs index, files)
"""
import time
from dataclasses import dataclass

from src import config


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    hint: str = ""
    info_only: bool = False      # information, never counts as a failure


def check_classifier(path=None):
    path = path or config.CLASSIFIER_PATH
    if not path.exists():
        return Check("Intent classifier", False, f"{path.name} not found", "Train it: python -m src.classifier.train")
    try:
        from src.classifier.inference import IntentPredictor
        out = IntentPredictor(path).predict("I was charged twice")
        return Check("Intent classifier", True, f"loads; 'I was charged twice' -> {out['intent']} ({out['confidence']:.2f})")
    except Exception as exc:  # noqa: BLE001
        return Check("Intent classifier", False, type(exc).__name__, "Re-train: python -m src.classifier.train")


def check_docs_index(index_dir=None):
    index_dir = index_dir or config.VECTORSTORE_DIR
    if not (index_dir / "index.faiss").exists():
        return Check("Documentation index (RAG)", False, "index not built", "Build it: python -m src.rag.ingest")
    try:
        from src.rag.retriever import get_retriever
        docs = get_retriever().invoke("how long do I have to request a refund")
        top = docs[0].metadata["source"] if docs else "nothing"
        return Check("Documentation index (RAG)", bool(docs), f"refund question -> {top}", "Rebuild: python -m src.rag.ingest")
    except Exception as exc:  # noqa: BLE001
        return Check("Documentation index (RAG)", False, type(exc).__name__, "Rebuild: python -m src.rag.ingest")


def check_groq():
    from src.llm.providers import classify_llm_error, get_groq_llm, get_groq_model_name, redact
    try:
        t0 = time.time()
        get_groq_llm().invoke("Reply with one word: ok")
        return Check("Groq (primary LLM)", True, f"{get_groq_model_name()} answered in {time.time() - t0:.1f}s")
    except Exception as exc:  # noqa: BLE001
        category, message = classify_llm_error(exc)
        hints = {"missing_key": "Put GROQ_API_KEY in .env (copy .env.example)", "auth": "Create a new key at console.groq.com",
                 "rate_limit": "Wait a minute; the Ollama fallback still works", "model_not_found": "Pick a model: python -m src.llm.providers --list"}
        return Check("Groq (primary LLM)", False, f"[{category}] {message} {redact(type(exc).__name__)}", hints.get(category, "See the details above"))


def check_ollama():
    from src.llm.providers import (get_ollama_base_url, get_ollama_llm, get_ollama_model_name, ollama_installed_models)
    wanted = get_ollama_model_name()
    try:
        installed = ollama_installed_models()
    except Exception:  # noqa: BLE001
        return Check("Ollama (fallback LLM)", False, f"server not reachable at {get_ollama_base_url()}",
                     "Start the Ollama app (Start menu), then run this again")
    if not any(n == wanted or n.split(":")[0] == wanted for n in installed):
        return Check("Ollama (fallback LLM)", False, f"model {wanted} not installed", f"Run: ollama pull {wanted}")
    try:
        t0 = time.time()
        get_ollama_llm().invoke("Reply with one word: ok")        # loads the model into memory = the warm-up
        return Check("Ollama (fallback LLM)", True, f"{wanted} loaded and answered in {time.time() - t0:.1f}s (kept warm for {config.OLLAMA_KEEP_ALIVE})")
    except Exception as exc:  # noqa: BLE001
        return Check("Ollama (fallback LLM)", False, type(exc).__name__, "Try: ollama run " + wanted)


def check_feature_file():
    path = config.FEATURE_REQUESTS_PATH
    n = sum(1 for _ in open(path, encoding="utf-8")) if path.exists() else 0
    return Check("Feature-request tickets", True, f"{n} existing" + ("" if n == 0 else f" (next ticket will be FR-{n + 1:04d})"),
                "For a clean demo (first ticket FR-0001) delete data\\feature_requests.jsonl" if n else "", info_only=True)


def check_app_dependencies():
    try:
        import streamlit
        return Check("Streamlit", True, f"version {streamlit.__version__}")
    except ImportError:
        return Check("Streamlit", False, "not installed", "pip install -r requirements.txt")


def run_all(llm=True):
    checks = [check_app_dependencies(), check_classifier(), check_docs_index(), check_feature_file()]
    if llm:
        print("Checking Groq and warming up Ollama (the first Ollama call can take about a minute)...")
        checks += [check_groq(), check_ollama()]
    return checks


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args()
    checks = run_all(llm=not args.no_llm)
    print()
    for c in checks:
        mark = "INFO" if c.info_only else ("OK  " if c.ok else "FAIL")
        print(f"[{mark}] {c.name}: {c.detail}")
        if c.hint and (not c.ok or c.info_only):
            print(f"        -> {c.hint}")
    failed = [c for c in checks if not c.ok and not c.info_only]
    print("\nREADY TO RECORD" if not failed else f"\nNOT READY: {len(failed)} problem(s) above")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
