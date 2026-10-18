"""Central configuration. Every other module imports its settings from here,
so nothing (paths, thresholds, hyperparameters) is hardcoded in several places."""
from pathlib import Path

# ---- Paths -----------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
KNOWLEDGE_BASE_DIR = DATA_DIR / "knowledge_base"
MODELS_DIR = ROOT_DIR / "models"
CLASSIFIER_PATH = MODELS_DIR / "classifier.pt"

# ---- Intents (index order = class id used by the model) ---------------------
INTENTS = ["billing", "technical", "feature_request", "general"]

# ---- Preprocessing ----------------------------------------------------------
MAX_LEN = 40
MAX_VOCAB_SIZE = 15000
PAD_TOKEN, UNK_TOKEN = "<pad>", "<unk>"   # indices 0 and 1

# ---- Model hyperparameters ---------------------------------------------------
EMBED_DIM = 100
HIDDEN_DIM = 128
N_LAYERS = 1
DROPOUT = 0.3
BIDIRECTIONAL = True

# ---- Training ---------------------------------------------------------------
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
SEED = 42

# ---- Routing ----------------------------------------------------------------
CONFIDENCE_THRESHOLD = 0.60   # below this, route to the general agent

# ---- Dataset building (Phase 3) -----------------------------------------------
LABEL2ID = {name: i for i, name in enumerate(INTENTS)}
ID2LABEL = {i: name for name, i in LABEL2ID.items()}
REAL_CAP_PER_INTENT = 300     # max REAL rows kept per intent (keeps synthetic SaaS data from being drowned)
VAL_FRAC = 0.15
TEST_FRAC = 0.15
MIN_FREQ = 2                  # a word must appear >= 2 times in TRAIN to get its own id

# ---- Phase 5 training extras ---------------------------------------------------
EPOCHS = 15                   # upper bound; early stopping usually ends sooner
PATIENCE = 5                  # stop if val macro-F1 has not improved for this many epochs
GRAD_CLIP = 5.0
HISTORY_PATH = MODELS_DIR / "training_history.json"
CURVES_PATH = MODELS_DIR / "training_curves.png"
WORD_DROPOUT = 0.3          # share of words randomly replaced by <unk> while training; chosen on VALIDATION (0.0 -> 79.5%, 0.15 -> 86.1%, 0.3 -> 88.9%)

# ---- LLM settings (Phase 8) -----------------------------------------------------
# Default Groq model. Override without touching code: put GROQ_MODEL=<model id> in .env.
# Verify what your key can use with:  python -m src.llm.providers --list
GROQ_DEFAULT_MODEL = "openai/gpt-oss-20b"
LLM_TEMPERATURE = 0.0         # 0 = most deterministic; good for tool-calling agents
LLM_TIMEOUT_SECONDS = 30
LLM_MAX_RETRIES = 1           # keep low: on a rate limit we want to fail over fast, not wait

# ---- Ollama fallback (Phase 9) --------------------------------------------------
# Override with OLLAMA_MODEL / OLLAMA_BASE_URL in .env. The model must support tool calling.
OLLAMA_DEFAULT_MODEL = "llama3.2:3b"          # ~2 GB download, runs on an ordinary laptop
OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434"
OLLAMA_TIMEOUT_SECONDS = 120                  # local CPU inference is slower than Groq

# ---- Tools (Phase 10) -----------------------------------------------------------
FEATURE_REQUESTS_PATH = DATA_DIR / "feature_requests.jsonl"   # written by log_feature_request at runtime

# ---- Agents (Phase 11) ---------------------------------------------------------
PRODUCT_NAME = "FlowDesk"                 # fictional project-management SaaS used for the demo
DEFAULT_CUSTOMER_ID = "demo_customer"     # the "signed-in" customer in the demo (mock data)
AGENT_MAX_ITERATIONS = 5                  # safety cap on tool-calling loops

# ---- RAG (Phase 13) ------------------------------------------------------------
VECTORSTORE_DIR = DATA_DIR / "vectorstore"           # FAISS index, rebuilt by `python -m src.rag.ingest`
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # local and free (downloads ~90 MB once)
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
RAG_TOP_K = 3
RAG_MIN_SCORE = 0.20          # calibrated on real scores: off-topic questions <= 0.14, real questions >= 0.21 (python -m src.rag.retriever --calibrate)

# ---- Conversation memory (Phase 14) ---------------------------------------------
MEMORY_MAX_TURNS = 6          # most recent user+assistant turns sent to the agents
MEMORY_MAX_CHARS = 1500       # per stored message (saves tokens on rate-limited free tiers)
# True: a LOW-confidence message in the middle of a conversation stays with the current agent (so "yes please"
# keeps its flow). False: strict rule everywhere - low confidence always goes to the general agent.
FOLLOWUP_STICKY = True

# ---- Robustness (Phase 15) -------------------------------------------------------
# If fewer than this share of a message's words are in the classifier's vocabulary, the classifier's confidence
# cannot be trusted (it was 0.87 for the gibberish "asdf qwerty") -> treat as unknown input -> general agent.
MIN_KNOWN_WORD_RATIO = 0.34
MAX_INPUT_CHARS = 1000
OLLAMA_KEEP_ALIVE = "30m"     # keep the local model loaded between requests (avoids a ~40 s reload mid-demo)
