# Demo video script (target 4:30, limit 5:00)

Every message below was run on the real system during development. LLM wording varies a little between takes;
the routing, tool calls and sidebar values are what you point at.

## Before recording (10 minutes)

1. `git pull` then `python -m pytest tests -q`  (all green)
2. Clean ticket numbering:  `Remove-Item data\feature_requests.jsonl -ErrorAction SilentlyContinue`
3. `python -m src.preflight`  -> must print **READY TO RECORD**. It also warms up Ollama (first call is slow).
4. `streamlit run app.py`, open http://localhost:8501 **in a fresh browser window**:
   - close every other tab (a tab showing API keys must never be on screen) and hide the bookmarks bar
   - browser zoom 100-110%, sidebar open, window maximised
   - do NOT show the `.env` file or any terminal that printed a key
5. Do one throw-away message, then click **Reset conversation** (so the models are warm and the counters start at 0).
6. Start the recorder (Win+Alt+R with Xbox Game Bar, or OBS), microphone on, 1080p.

## Timeline

| Time | Do (click from sidebar > Demo messages, or type) | Say | Point at |
|---|---|---|---|
| 0:00-0:25 | Show the app. | "This is a multi-agent support assistant for a fictional SaaS, FlowDesk. A custom Bi-LSTM intent classifier I trained in PyTorch decides which of four agents handles each message; the LLM never decides the intent. Agents use tools and document search, with Groq as the main LLM and Ollama as a local fallback." | Title, sidebar |
| 0:25-1:05 | `I have a billing problem.` then `It says I owe 500` | "Intent billing, confidence 0.62, Billing Agent. The first message only asks what is wrong. The second says 'it', and memory resolves that. The agent really called check_invoice; here is the tool call and its result. The data is mock demo data." | Predicted intent, confidence bar vs threshold 0.60, Selected agent, **Tool calls > expand JSON** |
| 1:05-1:40 | `the invoice page will not load` | "Now the classifier says technical, 0.89, so the router hands over to the Technical Agent. The conversation memory is shared, so it still knows we were talking about the invoice. That is an agent handoff with context preserved." | Selected agent changed to Technical; caption under the reply; tool `check_service_status` |
| 1:40-2:10 | `please add a linux app` then `please add dark mode` | "Feature Request Agent. It checks the roadmap first. Linux is not there, so it logs a real ticket, FR-0001. Dark mode is already planned for Q1 2027, so no ticket is created; that rule is enforced in code, not just in the prompt." | Two tool calls in the first reply; one in the second |
| 2:10-2:50 | `What are your support hours?` then `Can I run FlowDesk on my own server?` | "General Agent with RAG: documents were chunked, embedded locally with MiniLM into FAISS, and the agent retrieves chunks and answers only from them. Here is the source file and score. For something the documents never mention, it says the documentation does not cover it instead of inventing an answer." | Tool call `search_docs` > source file + score; the 'documentation doesn't cover that' answer |
| 2:50-3:25 | `uploading a photo gives me a red banner saying failed` then `asdf qwerty` | "Low confidence: the classifier is unsure, 0.32 for billing, which is below the 0.60 threshold, so it is routed to the General Agent, which answers from the docs. And for gibberish, the classifier is falsely confident, so a vocabulary check overrides it." | Predicted intent vs Selected agent; the reason line; Known words 0% |
| 3:25-4:05 | Switch on **Simulate Groq outage**, send `It says I owe 500` | "Now I simulate a Groq outage or rate limit. The system falls back to Ollama, a small local model, and it still calls the invoice tool. The sidebar shows the fallback and why." | Warning banner; `LLM: ollama (llama3.2:3b)`; Fallback used |
| 4:05-4:25 | Also switch on **Also simulate Ollama outage**, send any message | "If both providers fail, the user gets a friendly message, never a stack trace or a key." | The friendly error; Errors counter |
| 4:25-4:50 | Show `models/confusion_matrix_test.png` or the usage counters. | "Test accuracy 89.9%, macro-F1 0.898. On my separate hand-written hard test set it is 78.3%, below the 80% bar, and I report that honestly. The weak spots are short, context-dependent messages and feature requests. With more time: real support data and a calibrated confidence." | Confusion matrix |

If you are over 5:00, drop the "asdf qwerty" message and the Ollama-outage step.

## If something goes wrong while recording
- Groq says rate limit: that IS the fallback demo; turn the outage switch on and carry on, or wait a minute.
- Ollama is slow on the first call: that is why preflight warms it up; if it still happens, cut the pause in editing.
- A reply words things differently: fine. Only re-record if the routing or the tool call is wrong.

## Screenshots to take for the write-up (do them during a rehearsal)
1. Billing answer with the check_invoice tool call expanded
2. After the handoff: Technical Agent selected, two turns visible
3. Feature request ticket and the dark-mode "no ticket" case
4. RAG tool call expanded (source file + score)
5. Low-confidence fallback (predicted billing 0.321 -> General Agent)
6. Groq outage -> Ollama fallback banner
7. Both outages -> friendly error
8. `models/confusion_matrix_test.png`, `models/confusion_matrix_hard_test.png`, `models/training_curves.png`
Never include `.env`, the Groq console or any API key page.

## Requirement checklist (the 10 demo points)
1 each intent: billing, technical, feature request, general | 2 intent + confidence | 3 correct agent | 4 real tool call |
5 RAG | 6 multi-turn | 7 handoff with context | 8 low-confidence + fallback | 9 error / rate-limit (simulated) | 10 Ollama fallback
