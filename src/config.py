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
WORD_DROPOUT = 0.15         # share of words randomly replaced by <unk> while training (regularisation)
