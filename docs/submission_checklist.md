# Submission checklist

## A. Evidence files (should already exist on your laptop)
- [ ] `models/classifier.pt` (the trained classifier; the app loads it, never retrains)
- [ ] `models/eval_test.txt`, `models/eval_hard_test.txt`, `models/eval_log.txt`
- [ ] `models/confusion_matrix_test.png`, `models/confusion_matrix_hard_test.png`, `models/training_curves.png`
- [ ] `models/integration_report.txt` (from `python -m src.demo_check`, ideally 26/26)

## B. Documents
- [ ] Screenshots saved in `docs/screenshots/` (`01_...png`, `02_...png`; list in `docs/demo_script.md`). No API key or `.env` visible in any of them.
- [ ] Section 23 of `docs/writeup.md` rewritten in your own words
- [ ] `python -m src.make_report` -> open `docs/report.html` -> Print -> Save as PDF -> save as `docs/FlowDesk_report.pdf`
      (the report should show no red "missing" boxes)
- [ ] `README.md` reads correctly (clone/setup steps match what you actually run)

## C. Demo video (3-5 minutes)
- [ ] Recorded following `docs/demo_script.md`; covers all 10 required points
- [ ] Watched once: audio is clear, text is readable, **no API key / Groq console / `.env` on screen**
- [ ] Submitted the way your course asks (a ZIP this large with video may be refused; upload the video separately if needed)

## D. Clean state
- [ ] `Remove-Item data\feature_requests.jsonl -ErrorAction SilentlyContinue`
- [ ] `python -m pytest tests -q` -> all pass
- [ ] `python -m src.package --smoke` -> prints **READY TO SUBMIT** (it refuses if anything looks like a key)
- [ ] Open `submission\multi_agent_support_bot.zip` in File Explorer: there is **no `.env`**, no `venv`, no `__pycache__`
- [ ] Optional reproducibility: `pip freeze > requirements-lock.txt` (before packaging) to record the exact versions you used

## E. After submitting
- [ ] Consider deleting the Groq key at console.groq.com (API Keys) when grading is over
- [ ] Optional: to make the GitHub repo runnable too, commit the trained model and result files:
      `git add models/classifier.pt models/*.png models/eval_*.txt models/integration_report.txt models/training_history.json`

## Non-negotiables: where the evidence is
| Requirement | Evidence |
|---|---|
| Classifier built from scratch in PyTorch (Embedding + Bi-LSTM) | `src/classifier/model.py`, write-up sections 6-7 |
| LLM does not decide the intent | `src/routing/router.py` (`decide`, `RunnableBranch`); test: the LLM is called once, for the answer only |
| Vocabulary from training data only; no leakage | `src/classifier/dataset.py`; tests assert no shared groups/texts across splits |
| Accuracy >= 80% on the 4 intents | test set 89.9% (hard test 78.3%, reported honestly) |
| Confusion matrix and analysis | write-up sections 11-12, `models/confusion_matrix_*.png` |
| Saved checkpoint with model_state, vocab, label_map | `models/classifier.pt`; `src/classifier/inference.py` |
| 4 agents, own prompts and tools | `src/agents/prompts.py`, `tools.py`, `agents.py` |
| Real tool calls | `python -m src.demo_check`; sidebar "Tool calls" |
| RAG (MiniLM + FAISS + RecursiveCharacterTextSplitter) | `src/rag/`, write-up section 17 |
| Memory survives a handoff | `tests/test_memory.py`, demo_check "handoff" |
| Low-confidence threshold in one place | `src/config.py` `CONFIDENCE_THRESHOLD` |
| Groq -> Ollama fallback, friendly errors | `src/llm/providers.py`, demo_check, sidebar toggles |
| No LangGraph; classic LangChain | `create_tool_calling_agent`, `AgentExecutor`, `RunnableBranch` |
| No secrets in the submission | `src/package.py` scan + `.env.example` only |
