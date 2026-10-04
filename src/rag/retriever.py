"""RAG retrieval: question -> the most relevant documentation chunks.

  SupportDocsRetriever  a LangChain retriever (works with .invoke(query)); returns up to k chunks
                        whose cosine similarity is >= min_score, each with metadata["score"] and ["source"].
  get_retriever()       loads the saved FAISS index once (builds it if missing) and caches it.

Try it (shows raw scores, nothing filtered):   python -m src.rag.retriever "how long do I have to ask for a refund"
Calibrate the relevance threshold:             python -m src.rag.retriever --calibrate
"""
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src import config
from src.rag.ingest import build_index, get_embeddings


def cosine_from_l2(squared_distance):
    """FAISS returns squared L2 distance between unit vectors: d^2 = 2 - 2*cos  =>  cos = 1 - d^2/2."""
    return float(1.0 - squared_distance / 2.0)


class SupportDocsRetriever(BaseRetriever):
    store: Any
    k: int = config.RAG_TOP_K
    min_score: float = config.RAG_MIN_SCORE

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None) -> list[Document]:
        found = []
        for doc, dist in self.store.similarity_search_with_score(query, k=self.k):
            score = cosine_from_l2(dist)
            if score >= self.min_score:
                found.append(Document(page_content=doc.page_content, metadata={**doc.metadata, "score": score}))
        return found


def load_index(embeddings=None, index_dir=None):
    """Load the saved index; build it first if it does not exist yet.
    allow_dangerous_deserialization is needed because FAISS metadata is stored with pickle - safe here
    because we load only the file WE created from our own documents, never a downloaded one."""
    index_dir = index_dir or config.VECTORSTORE_DIR
    embeddings = embeddings or get_embeddings()
    if not (index_dir / "index.faiss").exists():
        print("Vector index not found - building it now (one time)...")
        return build_index(embeddings, index_dir=index_dir)
    return FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True, normalize_L2=True)


_CACHE = {}


def get_retriever(k=None, min_score=None):
    """Cached: loading the embedding model takes seconds, so do it once per process."""
    if "store" not in _CACHE:
        _CACHE["store"] = load_index()
    return SupportDocsRetriever(store=_CACHE["store"], k=k or config.RAG_TOP_K,
                                min_score=config.RAG_MIN_SCORE if min_score is None else min_score)


def reset_cache():
    _CACHE.clear()


# ---------------------------------------------------------------- CLI
ANSWERABLE = [  # (question, file that should answer it)
    ("How long do I have to request a refund?", "refund_policy.md"),
    ("How do I reset my password?", "account_and_security.md"),
    ("What is the maximum file upload size?", "troubleshooting_performance_and_uploads.md"),
    ("How many times are webhooks retried?", "api_and_webhooks.md"),
    ("Do you give a discount to non-profits?", "plans_and_pricing.md"),
    ("What are your support hours?", "support_and_contact.md"),
    ("Is there a cancellation fee?", "subscription_and_cancellation.md"),
    ("Which data regions can I choose?", "account_and_security.md"),
]
UNANSWERABLE = [  # clearly outside the documentation
    "What is the capital of France?",
    "How do I bake sourdough bread?",
    "Who won the football world cup?",
    "Explain quantum entanglement",
]


def calibrate():
    store = get_retriever(k=1, min_score=-1.0).store
    def best(q):
        doc, dist = store.similarity_search_with_score(q, k=1)[0]
        return cosine_from_l2(dist), doc.metadata["source"]
    print("ANSWERABLE questions (should match the expected file):")
    good = []
    for q, expected in ANSWERABLE:
        score, src = best(q)
        good.append(score)
        print(f"  {score:.2f}  {'OK ' if src == expected else 'MISS'} {src:45s} {q}")
    print("\nOUT-OF-SCOPE questions (should score low):")
    bad = []
    for q in UNANSWERABLE:
        score, src = best(q)
        bad.append(score)
        print(f"  {score:.2f}       {src:45s} {q}")
    lo, hi = min(good), max(bad)
    print(f"\nLowest answerable score: {lo:.2f} | highest out-of-scope score: {hi:.2f} | current RAG_MIN_SCORE: {config.RAG_MIN_SCORE}")
    if lo > hi:
        print(f"Separable. A threshold around {(lo + hi) / 2:.2f} keeps all answerable questions and blocks out-of-scope ones.")
    else:
        print("Scores overlap - keep the threshold low and rely on the agent prompt to say 'the documentation does not provide enough information'.")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?")
    parser.add_argument("--calibrate", action="store_true")
    args = parser.parse_args()
    if args.calibrate:
        return calibrate()
    if not args.query:
        parser.error("give a query or --calibrate")
    store = get_retriever(min_score=-1.0).store
    print(f"Top {config.RAG_TOP_K} chunks for {args.query!r}  (threshold {config.RAG_MIN_SCORE}):")
    for doc, dist in store.similarity_search_with_score(args.query, k=config.RAG_TOP_K):
        score = cosine_from_l2(dist)
        print(f"\n[{score:.2f}] {'PASS' if score >= config.RAG_MIN_SCORE else 'below threshold'}  {doc.metadata['source']}\n{doc.page_content}")


if __name__ == "__main__":
    main()
