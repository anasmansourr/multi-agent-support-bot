import re

import numpy as np
import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage

from src import config
from src.agents import tools
from src.llm.providers import LLMProvider
from src.agents.agents import run_agent
from src.rag import retriever as rag
from src.rag.ingest import build_index, load_documents, split_documents
from src.rag.retriever import SupportDocsRetriever, cosine_from_l2, load_index
from tests.fakes import ScriptedChatModel, tool_call_msg

STOP = {"the", "and", "for", "you", "your", "are", "can", "how", "what", "does", "this", "that", "with", "from", "have", "has"}


class FakeEmbeddings(Embeddings):
    """Deterministic bag-of-words hashing (no download). Shared words => higher cosine, like a real model."""
    DIM = 512

    def embed_query(self, text):
        v = np.zeros(self.DIM)
        for w in re.findall(r"[a-z0-9]+", text.lower()):
            if len(w) > 2 and w not in STOP:
                v[hash(w) % self.DIM] += 1.0
        return v.tolist()

    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]


@pytest.fixture(scope="module")
def index_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("faiss")
    build_index(FakeEmbeddings(), index_dir=path)
    return path


@pytest.fixture
def docs_retriever(index_dir):
    store = load_index(FakeEmbeddings(), index_dir=index_dir)
    return SupportDocsRetriever(store=store, k=3, min_score=0.2)


# ------------------------------------------------------------------ the knowledge base itself
def test_knowledge_base_is_small_and_coherent():
    docs = load_documents()
    assert 10 <= len(docs) <= 30 and len(docs) == 16
    for d in docs:
        words = len(d.page_content.split())
        assert d.page_content.startswith("# ") and 30 <= words <= 300, (d.metadata["source"], words)
    assert len({d.metadata["source"] for d in docs}) == len(docs)


def test_chunking_respects_the_configured_size_and_keeps_the_source():
    chunks = split_documents(load_documents())
    assert len(chunks) > len(load_documents())                   # long docs were split
    assert all(len(c.page_content) <= config.CHUNK_SIZE for c in chunks)
    assert all(c.metadata["source"].endswith(".md") for c in chunks)
    assert (config.CHUNK_SIZE, config.CHUNK_OVERLAP) == (500, 50)


def test_kb_facts_match_the_mock_tools():
    """The documents and the mock tools must not contradict each other."""
    text = {d.metadata["source"]: d.page_content for d in load_documents()}
    assert "30 days" in text["refund_policy.md"] and tools.REFUND_WINDOW_DAYS == 30
    assert "$49" in text["plans_and_pricing.md"] and "$29" in text["plans_and_pricing.md"]
    assert "Dark mode" in tools.MOCK_ROADMAP[0]["feature"] and "dark mode (Q1 2027)" in text["feature_requests_policy.md"]
    for name, c in tools.MOCK_CUSTOMERS.items():                 # per-seat price x seats == last payment
        price = {"Starter": 29, "Pro": 49, "Business": 79}[c["plan"].split()[0]]
        seats = int(c["plan"].split("(")[1].split()[0])
        assert c["last_payment_amount"] == price * seats, name


# ------------------------------------------------------------------ retrieval
def test_retrieval_finds_the_right_document(docs_retriever):
    for question, source in [("how long do I have to request a refund", "refund_policy.md"),
                             ("webhook retries after failed delivery", "api_and_webhooks.md"),
                             ("what are the support hours", "support_and_contact.md"),
                             ("maximum upload file size", "troubleshooting_performance_and_uploads.md")]:
        assert docs_retriever.invoke(question)[0].metadata["source"] == source, question


def test_retriever_returns_at_most_k_scored_chunks(docs_retriever):
    docs = docs_retriever.invoke("refund payment billing invoice")
    assert 1 <= len(docs) <= 3
    assert all(0.2 <= d.metadata["score"] <= 1.0 for d in docs)
    assert [d.metadata["score"] for d in docs] == sorted((d.metadata["score"] for d in docs), reverse=True)


def test_off_topic_question_retrieves_nothing(docs_retriever):
    assert docs_retriever.invoke("quantum entanglement sourdough bread football") == []


def test_cosine_conversion():
    assert cosine_from_l2(0.0) == 1.0 and cosine_from_l2(2.0) == 0.0 and cosine_from_l2(1.0) == 0.5


def test_index_is_saved_and_reloadable(index_dir):
    assert (index_dir / "index.faiss").exists() and (index_dir / "index.pkl").exists()
    store = load_index(FakeEmbeddings(), index_dir=index_dir)
    assert store.similarity_search("refund", k=1)


# ------------------------------------------------------------------ the search_docs tool
@pytest.fixture
def patched_rag(monkeypatch, docs_retriever):
    monkeypatch.setattr(rag, "get_retriever", lambda *a, **k: docs_retriever)


def test_search_docs_tool_returns_excerpts_with_sources(patched_rag):
    out = tools.search_docs_tool.invoke({"query": "how long do I have to request a refund"})
    assert out["found"] and out["results"][0]["source"] == "refund_policy.md"
    assert "30 days" in out["results"][0]["excerpt"]


def test_search_docs_tool_says_so_when_nothing_is_relevant(patched_rag):
    out = tools.search_docs_tool.invoke({"query": "quantum entanglement sourdough bread"})
    assert out["found"] is False and "does not provide enough information" in out["message"]


def test_search_docs_tool_failure_is_safe(monkeypatch):
    def broken(*a, **k):
        raise OSError("model download failed at /secret/path")
    monkeypatch.setattr(rag, "get_retriever", broken)
    out = tools.search_docs_tool.invoke({"query": "refund"})
    assert "temporarily unavailable" in out and "/secret/path" not in out


def test_empty_query_is_rejected(patched_rag):
    assert "something to search for" in tools.search_docs_tool.invoke({"query": "  "})


# ------------------------------------------------------------------ an agent really uses RAG
def test_technical_agent_answers_from_retrieved_documentation(patched_rag):
    model = ScriptedChatModel(script=[
        tool_call_msg("search_docs", {"query": "webhook retries after failed delivery"}),
        AIMessage(content="Failed webhook deliveries are retried 3 times (after 1, 5 and 30 minutes).")])
    r = run_agent("technical", "how many times are webhooks retried?",
                  providers=[LLMProvider("groq", lambda: "fake", lambda: model)])
    assert r.tool_calls[0]["tool"] == "search_docs"
    assert r.tool_calls[0]["result"]["results"][0]["source"] == "api_and_webhooks.md"
    second_prompt = " ".join(str(m.content) for m in model.seen[1])
    assert "1, 5, 30 minutes" in second_prompt or "after 1, 5 and 30 minutes" in second_prompt   # chunk reached the LLM
    assert "3 times" in r.output
