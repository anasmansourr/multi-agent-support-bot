import json

from src import config
from src.preflight import (check_app_dependencies, check_classifier, check_docs_index, check_feature_file, check_groq,
                           check_ollama, run_all)


def test_missing_classifier_gives_a_fix(tmp_path):
    c = check_classifier(tmp_path / "nope.pt")
    assert not c.ok and "python -m src.classifier.train" in c.hint


def test_missing_index_gives_a_fix_and_does_not_build_it(tmp_path):
    c = check_docs_index(tmp_path)
    assert not c.ok and "python -m src.rag.ingest" in c.hint and not (tmp_path / "index.faiss").exists()


def test_missing_groq_key_is_explained(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    c = check_groq()
    assert not c.ok and ".env" in c.hint and "gsk_" not in c.detail


def test_unreachable_ollama_is_explained(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:9")
    c = check_ollama()
    assert not c.ok and "Ollama app" in c.hint


def test_feature_file_is_informational_only(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "FEATURE_REQUESTS_PATH", tmp_path / "fr.jsonl")
    assert check_feature_file().detail == "0 existing"
    (tmp_path / "fr.jsonl").write_text(json.dumps({"a": 1}) + "\n" + json.dumps({"a": 2}) + "\n")
    c = check_feature_file()
    assert c.info_only and "FR-0003" in c.detail and "delete" in c.hint


def test_streamlit_check_and_local_only_run(monkeypatch, tmp_path):
    assert check_app_dependencies().ok
    monkeypatch.setattr(config, "CLASSIFIER_PATH", tmp_path / "missing.pt")
    names = [c.name for c in run_all(llm=False)]
    assert "Groq (primary LLM)" not in names and "Intent classifier" in names
