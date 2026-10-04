# Multi-Agent Support Chatbot with a Custom Bi-LSTM Intent Classifier, LangChain, RAG, Tools, Memory and Streamlit

*Generative AI course project. Domain: SaaS product support (a fictional product called FlowDesk).*

> **Honesty notes.** FlowDesk, its documentation, customers, invoices, service status and roadmap are **fictional / mock data**.
> The synthetic training sentences were written with an AI assistant (Claude); they are labelled `synthetic` in the data.
> Banking77 and CLINC150 are real public datasets (see their dataset cards for licence terms).
> All metrics below come from runs on the author's machine; nothing is estimated.

## 1. Project domain

A customer-support assistant for a fictional project-management SaaS, **FlowDesk**. Each message is classified by a neural network
that was trained from scratch, routed to one of four specialised agents, and answered with the help of tools and a
retrieval-augmented document search. The classifier, not the LLM, decides which agent runs.

## 2. Selected intents

| Intent | Agent | Handles |
|---|---|---|
| `billing` | Billing Agent | the customer's own invoices, charges, refunds, payment methods |
| `technical` | Technical Agent | errors, outages, login, API and webhook problems |
| `feature_request` | Feature Request Agent | ideas and requests for new features |
| `general` | General Agent | greetings, company, plans and pricing information, unclear messages |

A written labelling policy (`docs/labeling_policy.md`) decides borderline cases; for example public pricing and discount
questions are `general`, while "how much do **I** owe" is `billing`.

## 3. Dataset source

| Source | Type | Used for |
|---|---|---|
| Banking77 (`PolyAI/banking77`) | real, public | `billing` (8 labels), `technical` (1 label) |
| CLINC150 (`clinc_oos`, "plus") | real, public | `billing` (4), `technical` (3), `general` (9) |
| Synthetic SaaS sentences | **synthetic**, written with an AI assistant | all four intents; the only source of `feature_request` |
| Hard test set | **synthetic**, hand-written, messy phrasing, partly ambiguous | final stress test only; never trained on |

Final splits: **train 1,866, validation 404, test 407**, plus **60** hard-test sentences (15 per intent).
Public datasets contain no `feature_request` class, so that intent is 100% synthetic. Real sentences are banking and
voice-assistant phrased, not SaaS phrased (a documented domain shift).

## 4. Label collapsing / mapping

Every public label used is listed explicitly in `src/classifier/label_mapping.py`; every other label is dropped on purpose.
The script validates the names against the real dataset and fails on a typo.

| Public label | Intent |
|---|---|
| Banking77: `card_payment_fee_charged`, `card_payment_not_recognised`, `declined_card_payment`, `extra_charge_on_statement`, `pending_card_payment`, `refund_not_showing_up`, `request_refund`, `transaction_charged_twice` | billing |
| Banking77: `passcode_forgotten` | technical |
| CLINC150: `bill_balance`, `bill_due`, `pay_bill`, `interest_rate` | billing |
| CLINC150: `account_blocked`, `reset_settings`, `sync_device` | technical |
| CLINC150: `greeting`, `goodbye`, `thank_you`, `what_can_i_ask_you`, `who_made_you`, `are_you_a_bot`, `what_is_your_name`, `how_old_are_you`, `who_do_you_work_for` | general |

At most **300 real rows per intent** are kept, sampled evenly across the original labels, so the synthetic SaaS data is not drowned out.

## 5. Data preprocessing

1. Lowercase; delete apostrophes ("don't" becomes "dont"); other punctuation becomes a space; numbers become the token `<num>`.
2. Word-level tokenisation (a custom `Vocabulary` class; `torchtext` was avoided because it is deprecated).
3. Vocabulary built from the **training split only** (words seen at least twice): **996 words**, `<pad>` = 0, `<unk>` = 1.
4. Sequences padded or truncated to **40** tokens (very few training sentences are longer; 0.3% in the first data build); true lengths are kept for packed RNN input.
5. **Leakage control:** duplicate and conflicting-label sentences removed; any sentence equal to a hard-test sentence removed from training;
   the split is **by group**: all light variants (greeting or closing added) of one synthetic sentence stay in the same split,
   and tests assert that no group or text appears in two splits.
6. Training-time regularisation: **word dropout 0.3** (random words replaced by `<unk>`), applied only while training.

## 6. Model architecture

```
token ids (batch, 40)
  -> Embedding(996, 100, padding_idx=0)
  -> Bi-LSTM (hidden 128, 1 layer, packed sequences so padding is ignored)
  -> final forward + backward hidden states (256)
  -> Dropout(0.3)
  -> Linear(256, 4)  -> logits -> softmax -> intent + confidence
```
Implemented from scratch in PyTorch (no pretrained weights): **336,148 trainable parameters**. A GRU variant is one argument away.

## 7. Training configuration

Adam, learning rate 1e-3, batch size 64, `CrossEntropyLoss` with **inverse-frequency class weights** (feature_request weight 1.5 vs about 0.9 for the others),
gradient clipping 5.0, up to 15 epochs with early stopping (patience 5), seed 42.
The checkpoint with the best **validation macro-F1** is kept: epoch 13. The saved `classifier.pt` holds `model_state`, `vocab`, `label_map`
and the hyper-parameters needed to rebuild the model. The app never retrains.

## 8. Training results (iteration history, validation set only)

The first model reached 72.2% on validation and badly overfit. Error analysis showed the synthetic slice was weakest (59.1%) for reasons
in **my data**, not the network: contradictory labels on pricing questions, a learned shortcut ("starts with *do you / can you*" means general),
and too few distinct sentences. I wrote a labelling policy, relabelled 7 sentences, added about 40 new sentences per intent, and added word dropout.

| Step | Validation accuracy | Validation macro-F1 |
|---|---|---|
| First model (old data, no word dropout) | 72.2% | 0.719 |
| + labelling policy and more diverse data | 79.5% | 0.783 |
| + word dropout 0.15 | 86.1% | 0.863 |
| + word dropout 0.30 (chosen) | 88.9% | 0.889 |

The validation set changed between the first two rows (the data pool changed). Three settings were compared on the same validation set, so 88.9% is slightly optimistic; the test set below is the honest number.

{{FIGURE:models/training_curves.png|Training curves (train vs validation)}}

## 9. Accuracy

| Split | n | Accuracy | Course target (>= 80%) |
|---|---|---|---|
| **Test** | 407 | **89.9%** | met |
| **Hard test** (hand-written, partly ambiguous) | 60 | **78.3%** (47/60) | not met; reported as measured |

Test accuracy by source: Banking77 100.0% (n=42), CLINC150 94.8% (n=97), synthetic 86.6% (n=268).
The hard-test set is small (one sentence = 1.7 points; a rough 95% interval is about 67-88%) and was not used for any tuning.

## 10. Macro-F1

Test **0.898**; hard test **0.783**.

| Intent | Test precision | recall | F1 | support | Hard-test F1 |
|---|---|---|---|---|---|
| billing | 0.935 | 0.926 | 0.930 | 108 | 0.800 |
| technical | 0.884 | 0.846 | 0.865 | 117 | 0.733 |
| feature_request | 0.966 | 0.824 | 0.889 | 68 | 0.800 |
| general | 0.854 | 0.974 | 0.910 | 114 | 0.800 |

## 11. Confusion matrices (rows = true intent, columns = predicted)

**Test set**

| | billing | technical | feature_request | general |
|---|---|---|---|---|
| billing | 100 | 4 | 0 | 4 |
| technical | 5 | 99 | 2 | 11 |
| feature_request | 0 | 8 | 56 | 4 |
| general | 2 | 1 | 0 | 111 |

**Hard-test set**

| | billing | technical | feature_request | general |
|---|---|---|---|---|
| billing | 12 | 2 | 0 | 1 |
| technical | 2 | 11 | 0 | 2 |
| feature_request | 0 | 2 | 10 | 3 |
| general | 1 | 0 | 0 | 14 |

{{FIGURE:models/confusion_matrix_test.png|Confusion matrix, test set}}

{{FIGURE:models/confusion_matrix_hard_test.png|Confusion matrix, hard-test set}}

## 12. Confusion analysis

* **technical -> general (11 on test), feature_request -> technical (8), feature_request -> general (4).** These two intents have the least real data
  (feature_request has none), so they rely on synthetic phrasing. Short, vague bug reports and polite requests ("can you build a plugin
  marketplace") resemble general help questions.
* **Greeting prefixes cause confident mistakes.** "hello, what is your cancellation fee" is predicted general because "hello" is a strong general cue.
* **billing <-> general is a genuine policy boundary**: "what is your cancellation fee" is a policy question that could reasonably be either.
* **feature_request recall is the weak point** (test 82.4%, hard test 66.7%) while its precision is 97-100%: the model only recognises requests with familiar phrasing.
* **Ambiguous hard-test sentences** ("payment went through but my order doesn't show up in the app": billing or technical?) explain part of the hard-test errors.
* Errors cluster: 13 of the 15 most confident test mistakes come from just 5 source sentences (each has several variants in the same split),
  so the effective test sample is smaller than 407.
* The classifier is **over-confident when wrong** (mean confidence 0.82 on mistakes), so the 0.60 threshold catches gibberish and off-topic text better than it catches confident confusions.
  Very short, context-dependent messages ("what do I owe?", "yes please") are the weakest case because each message is classified alone.

## 13. System architecture

```
User message
   -> IntentPredictor (custom Bi-LSTM)          intent + confidence + known-word ratio
   -> decide()  [LangChain RunnableLambda]       one place for the low-confidence / unknown-word rules
   -> RunnableBranch                             one branch per agent (general = default)
   -> AgentExecutor (own prompt + own tools)     create_tool_calling_agent
   -> tools and/or RAG (FAISS)                   real function calls
   -> LLM via call_with_fallback                 Groq first, Ollama if Groq fails
   -> answer + routing details -> Streamlit UI   shared ConversationMemory feeds every agent
```
The LLM never classifies. Classifier, router, agents, tools, RAG, memory and UI are separate modules with one job each.

## 14. Agent descriptions

| Agent | Role | Tools |
|---|---|---|
| Billing | The customer's own money and account. Can only *check*; humans issue refunds. | `check_invoice`, `check_refund_eligibility` |
| Technical | Errors, outages, login, API. Asks for details, checks service status, answers from documentation only. | `check_service_status`, `search_docs` |
| Feature Request | Looks up the roadmap; reports status if shipped/planned; logs a ticket otherwise. | `check_roadmap`, `log_feature_request`, `search_docs` |
| General | Greetings, company, plans, policies; asks one clarifying question when unclear. | `search_docs` |

## 15. System prompts

These are generated from the code (`src/agents/prompts.py`), so they are exactly what the agents receive.

{{PROMPTS}}

## 16. Tools

All account, status and roadmap data is **mock data** (deterministic, hard-coded, documented in `src/agents/tools.py`). The one real side effect is
`log_feature_request`, which appends to `data/feature_requests.jsonl`. `search_docs` is real retrieval over real (fictional) documents.

| Tool | Purpose |
|---|---|
| `check_invoice(customer_id)` | invoice status, amount due, due date, last payment (mock) |
| `check_refund_eligibility(customer_id)` | 30-day refund window check (mock) |
| `check_service_status(service)` | API, dashboard, mobile app, email, webhooks status (mock) |
| `check_roadmap(keyword)` | feature status: shipped, in progress, planned, under review, not planned (mock) |
| `log_feature_request(title, ...)` | creates a ticket; **refuses duplicates of features already on the roadmap, enforced in code** |
| `search_docs(query)` | RAG over the knowledge base |

Tools never crash the app: expected problems return a message, unexpected ones become a safe text error for the model. Each agent can only call its own tools (tested).

## 17. RAG architecture

* 16 short fictional documents (about 1,700 words) in `data/knowledge_base/`.
* `RecursiveCharacterTextSplitter`, chunk size 500, overlap 50 -> **30 chunks**.
* Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (local, free). Vector store: **FAISS**; cosine similarity from normalised vectors.
* Retriever returns up to **3** chunks above a relevance threshold of **0.20** (calibrated: off-topic questions scored at most 0.14, real questions at least 0.32).
* Measured on 14 questions: the expected document was ranked first for **13/14** and in the top 3 for **14/14**.
* The product name is stripped from search queries in code (measured: including it pushed the right page out of the top 5).
* Anti-hallucination: agents are told to answer only from retrieved text and to say "the documentation does not provide enough information" otherwise.
  Verified on real runs: "Can I run FlowDesk on my own server?" -> "I'm not sure if FlowDesk can be self-hosted. The documentation doesn't cover that."

## 18. Conversation memory

One shared `ConversationMemory` (last 6 turns, each message clipped to 1,500 characters) is passed to whichever agent is selected, so context survives a handoff
(tested: a billing turn followed by a technical turn; the technical agent's prompt contains the billing turn). Each turn records which agent answered.
A summarisation hook is built in (`summarizer`) but not enabled; dropped turns can be condensed instead of forgotten. Failed turns are not stored.

## 19. Routing logic

1. The classifier returns intent, confidence and the share of the message's words it knows.
2. If the classifier fails or returns something invalid -> General agent (reason recorded).
3. If fewer than 34% of the words are known (confidence cannot be trusted, e.g. "asdf qwerty" scored 0.87) -> General.
4. If confidence < threshold -> General (see below).
5. Otherwise -> the agent for the predicted intent. A confident classification always wins, which is how handoffs happen.

## 20. Low-confidence fallback

The threshold is **0.60**, defined once in `src/config.py` and adjustable in the UI. Example: "uploading a photo gives me a red banner saying failed" is classified
`billing` with confidence **0.321**, so it is routed to the General agent, which answers from the documentation (or asks a clarifying question).
The sidebar shows predicted intent, confidence, selected agent and the reason. An optional setting (`FOLLOWUP_STICKY`) keeps the current agent for low-confidence
follow-ups mid-conversation; a first low-confidence message always goes to General. Validation check: the unknown-word guard flagged 0 of 404 real validation sentences.

## 21. LLM fallback and error handling

* Primary: **Groq** (`openai/gpt-oss-20b`, free tier). Fallback: **Ollama** running `llama3.2:3b` locally, kept loaded for 30 minutes.
* `call_with_fallback` tries Groq, then Ollama, on any provider failure (missing key, invalid key, rate limit, network, unknown model).
  Errors are mapped to short categories; the UI never shows stack traces or keys (keys are redacted from logs).
* If both fail, the user sees "I'm having trouble reaching the AI service right now. Please try again in a minute."
* Empty input is rejected, very long input is truncated, any unexpected bug returns a friendly message and is logged.
* Demo switches in the sidebar simulate a Groq outage and an Ollama outage.
* Known limitation: the 3B fallback model follows instructions less reliably than Groq's model (it sometimes mentions tool names or offers actions it cannot perform).

## 22. Screenshots of the working application

{{SCREENSHOTS}}

## 23. Reflection: what I would improve with more time

*(Draft: rewrite this in your own words.)*

* Collect **real SaaS support tickets**: the biggest limit is that real training data is bank and voice-assistant phrased and `feature_request` is entirely synthetic.
* Write a **second, fresh hard-test set** so the model can be measured again without reusing one I have already looked at.
* **Classify with conversation context**: short follow-ups such as "what do I owe?" or "yes please" are misrouted because each message is classified alone.
* **Calibrate the confidence** (for example temperature scaling): wrong predictions currently average 0.82 confidence.
* Compare **GRU vs LSTM over several random seeds**; the validation set is small, so single-run differences of 2-3 points are noise.
* Use a **stronger local fallback model** if the hardware allows; summarise old conversation turns with an LLM; connect the tools to a real billing system.

## Appendix A. Testing summary

* **176 automated tests pass** (2 opt-in tests skipped): data pipeline, model, training, evaluation, inference, tools, agents, routing, memory, RAG, fallback, UI.
  The Streamlit UI is tested headlessly with Streamlit's `AppTest` against the real `app.py`; the tests were mutation-checked (deliberately broken code made them fail).
* **26/26 integration checks pass** on the real stack (`python -m src.demo_check`): real tool calls, RAG, memory, a billing -> technical handoff,
  Groq outage -> Ollama with the tool still called, both providers down, empty input, low confidence, unknown input.
* Requirement-to-test mapping: `docs/test_matrix.md`.
