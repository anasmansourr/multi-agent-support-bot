"""Phase 1 check: confirms the environment is ready. Run: python verify_setup.py"""
import importlib
import sys

print(f"Python {sys.version.split()[0]}")

packages = [
    "torch", "numpy", "pandas", "sklearn", "matplotlib", "datasets",
    "langchain", "langchain_community", "langchain_groq", "langchain_ollama",
    "sentence_transformers", "faiss", "streamlit", "dotenv", "pytest",
]
failed = []
for name in packages:
    try:
        mod = importlib.import_module(name)
        print(f"  OK   {name:24s} {getattr(mod, '__version__', '')}")
    except Exception as exc:  # noqa: BLE001
        failed.append(name)
        print(f"  FAIL {name:24s} {type(exc).__name__}: {exc}")

try:
    from langchain.agents import AgentExecutor, create_tool_calling_agent  # noqa: F401
    from langchain_core.runnables import RunnableBranch, RunnableLambda    # noqa: F401
    print("  OK   LangChain classic agent imports")
except Exception as exc:  # noqa: BLE001
    failed.append("langchain-agents")
    print(f"  FAIL LangChain classic agent imports: {exc}")

from src import config
print(f"Project root: {config.ROOT_DIR}")
print(f"Intents: {config.INTENTS}")

print("\nSETUP OK" if not failed else f"\nSETUP INCOMPLETE: {failed}")
sys.exit(1 if failed else 0)
