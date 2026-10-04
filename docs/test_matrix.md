# Test matrix: requirement -> evidence

Run all automated tests: `python -m pytest tests -q`.  Real-stack check: `python -m src.demo_check`.

| # | Required test | Automated (pytest) | Real-stack / manual evidence |
|---|---|---|---|
| 1 | Classifier prediction | `tests/test_inference.py`, `tests/test_model.py`, `tests/test_train.py` | `models/eval_test.txt`, `eval_hard_test.txt`; `python -m src.classifier.inference "<text>"` |
| 2 | Intent confidence | `tests/test_inference.py` (probabilities sum to 1, 0 for odd input), `test_routing.py` (threshold boundary) | sidebar shows confidence; `--route-only` prints 3 decimals |
| 3 | Routing | `tests/test_routing.py`, `tests/test_demo_check.py` | `python -m src.demo_check --routing-only` |
| 4 | Agent selection | `tests/test_routing.py` (the LLM receives the selected agent's prompt and tools only) | sidebar "Selected agent" |
| 5 | Tool invocation | `tests/test_tools.py`, `tests/test_agents.py` (tool really executes; file really written) | `python -m src.demo_check` (live) |
| 6 | RAG retrieval | `tests/test_rag.py` | `python -m src.rag.retriever --calibrate` (hit@3 = 14/14 on the author's run) |
| 7 | Conversation memory | `tests/test_memory.py` (incl. memory survives a handoff) | `python -m src.bot`, `/memory` |
| 8 | Low-confidence fallback | `tests/test_routing.py`, `tests/test_memory.py` | `demo_check` + sidebar for "uploading a photo gives me a red banner saying failed" |
| 9 | Groq failure -> Ollama | `tests/test_llm_fallback.py`, `tests/test_agents.py` | `demo_check` (live) and the sidebar "Simulate Groq outage" toggle |
| 10 | Streamlit end-to-end | `tests/test_ui.py` (headless, real `app.py`, fake bot) | `streamlit run app.py` + screenshots |

Not automatable offline: the quality of live Groq / Ollama answers. `demo_check` records them in
`models/integration_report.txt` for human review.
