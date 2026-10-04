"""RAG ingestion: documents -> chunks -> embeddings -> FAISS index (saved to disk).

    Documents (data/knowledge_base/*.md)
      -> RecursiveCharacterTextSplitter (chunk_size=500, chunk_overlap=50)
      -> HuggingFace sentence-transformers embeddings (all-MiniLM-L6-v2, runs locally, free)
      -> FAISS vector store (data/vectorstore/)

Run after changing the documents:   python -m src.rag.ingest
"""
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config


def get_embeddings():
    """The embedding model. Imported lazily because loading it is slow (and downloads on first use)."""
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL, encode_kwargs={"normalize_embeddings": True})


def load_documents(kb_dir=None):
    """One Document per .md file; metadata keeps the file name so answers can cite their source."""
    kb_dir = kb_dir or config.KNOWLEDGE_BASE_DIR
    docs = []
    for path in sorted(kb_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            docs.append(Document(page_content=text, metadata={"source": path.name}))
    return docs


def split_documents(docs, chunk_size=None, chunk_overlap=None):
    """Chunking: small overlapping pieces so one retrieved chunk is focused and fits in the LLM prompt."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size or config.CHUNK_SIZE,
                                              chunk_overlap=config.CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap)
    return splitter.split_documents(docs)


def build_index(embeddings=None, kb_dir=None, index_dir=None):
    """Build the FAISS index from the knowledge base and save it. normalize_L2=True makes distances
    comparable as cosine similarity (see retriever.py)."""
    chunks = split_documents(load_documents(kb_dir))
    if not chunks:
        raise ValueError(f"No documents found in {kb_dir or config.KNOWLEDGE_BASE_DIR}")
    store = FAISS.from_documents(chunks, embeddings or get_embeddings(), normalize_L2=True)
    index_dir = index_dir or config.VECTORSTORE_DIR
    index_dir.mkdir(parents=True, exist_ok=True)
    store.save_local(str(index_dir))
    return store


def main():
    docs = load_documents()
    chunks = split_documents(docs)
    print(f"Loaded {len(docs)} documents -> {len(chunks)} chunks "
          f"(chunk_size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP})")
    print(f"Embedding with {config.EMBEDDING_MODEL} (first run downloads the model)...")
    build_index()
    print(f"Saved FAISS index -> {config.VECTORSTORE_DIR}")


if __name__ == "__main__":
    main()
