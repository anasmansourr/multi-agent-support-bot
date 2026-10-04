# Multi-Agent Support Chatbot (Bi-LSTM intent classifier, LangChain, RAG, tools, memory, Streamlit)

A customer-support assistant for **FlowDesk**, a *fictional* project-management SaaS. A **custom Bi-LSTM classifier trained from
scratch in PyTorch** decides the intent of every message; a LangChain router sends it to one of four specialised tool-calling
agents (billing, technical, feature request, general); agents use tools and a local RAG document search; Groq answers, with a
local Ollama model as automatic fallback; Streamlit shows the chat and exactly how each message was routed.

> **The LLM never decides the intent.** All account, status and roadmap data is **mock data**, FlowDesk and its documents are fictional,
> and the synthetic training sentences were written with an AI assistant (they are labelled `synthetic`).

```
User message -> Bi-LSTM classifier -> intent + confidence -> LangChain router (RunnableBranch)
             -> specialised agent (AgentExecutor) -> tools and/or RAG (FAISS) -> LLM (Groq, else Ollama) -> Streamlit
```

## Results (from real runs)
| | Accuracy | Macro-F1 |
|---|---|---|
| Test set (407 sentences) | **89.9%** | **0.898** |
| Hard test set (60 hand-written, partly ambiguous) | 78.3% | 0.783 |

181 automated tests pass; 26/26 real-stack integration checks pass. Details, confusion matrices and analysis: `docs/writeup.md`.

## Quick start (Windows PowerShell)
```powershell
git clone <repo-url>; cd multi-agent-support-bot
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
python verify_setup.py
```
1. **Groq key** (free): create one at console.groq.com, then `Copy-Item .env.example .env` and put the key in `.env`. Never commit or share `.env`.
   Check which model your key can use: `python -m src.llm.providers --list` (override with `GROQ_MODEL=` in `.env`).
2. **Ollama** (local fallback; no account needed): install from ollama.com, then `ollama pull llama3.2:3b`.
3. Check everything and warm it up: `python -m src.preflight`
4. Run the app: `streamlit run app.py`

## Rebuild everything from scratch (needs internet once, for the datasets and the embedding model)
```powershell
python -m src.classifier.data_acquisition       # download + map Banking77 / CLINC150
python -m src.classifier.synthetic_data         # synthetic sentences + hand-written hard test
python -m src.classifier.dataset                # leak-free split + vocabulary
python -m src.classifier.train                  # trains and saves models/classifier.pt (best validation macro-F1)
python -m src.classifier.evaluate --split val   # use val while improving; test / hard_test only for the final numbers
python -m src.rag.ingest                        # chunk, embed and index data/knowledge_base/
python -m src.rag.retriever --calibrate         # retrieval quality + relevance threshold
```

## Useful commands
| Command | What it does |
|---|---|
| `python -m pytest tests -q` | all automated tests (no network or API key needed) |
| `python -m src.demo_check [--routing-only]` | runs every demo scenario on the real stack, PASS/FAIL |
| `python -m src.bot` | terminal chat (`/memory`, `/outage`, `/reset`) |
| `python -m src.routing.router --route-only "msg"` | show routing for a message without calling an LLM |
| `python -m src.classifier.inference "msg"` | classifier only: intent + confidence |
| `python -m src.make_report` | builds `docs/report.html` (print it to PDF) |

## Project layout
```
app.py                 Streamlit UI
src/classifier/        preprocessing, model (Bi-LSTM), dataset, train, evaluate, inference, synthetic data, label mapping
src/routing/router.py  decide() rules + RunnableLambda / RunnableBranch chain
src/agents/            prompts.py, tools.py (mock data), agents.py (create_tool_calling_agent + AgentExecutor)
src/rag/               ingest.py (chunks, embeddings, FAISS), retriever.py
src/llm/providers.py   Groq + Ollama + call_with_fallback + error classification
src/memory/history.py  shared bounded conversation memory (summarizer hook)
src/bot.py             Router + memory;  src/ui_helpers.py, src/preflight.py, src/demo_check.py, src/make_report.py
data/knowledge_base/   16 short fictional FlowDesk documents (RAG)
docs/                  writeup.md, demo_script.md, test_matrix.md, labeling_policy.md, classifier_results.md
tests/                 pytest suite (including headless Streamlit tests)
```

## Configuration
Everything tunable lives in `src/config.py`: intents, model sizes, `CONFIDENCE_THRESHOLD` (0.60, also a UI slider),
RAG chunking and threshold, memory size, model names. Secrets only in `.env` (see `.env.example`).

## Security
No key is hardcoded; `.env` is gitignored; a test fails if anything that looks like a Groq key appears in the repo; keys are redacted from
error output; the UI shows friendly messages instead of stack traces. **Never include `.env` in a submission ZIP.**

## Known limitations
Short, context-dependent messages are weak for the classifier (it classifies each message alone) and its confidence is over-confident when wrong;
`feature_request` has no real training data; the 3B local fallback model follows instructions less reliably than Groq's model;
the hard test set is small. See `docs/writeup.md` sections 12 and 23.

## Data attribution
Banking77 (PolyAI) and CLINC150 (`clinc_oos`) are public datasets; see their dataset cards for authors and licences.
