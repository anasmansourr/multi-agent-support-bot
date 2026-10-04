# Questions you may be asked (and short, honest answers)

**Why not just ask the LLM to pick the agent?** The course requires a trained classifier to route. It is also faster, free,
testable and independent of API limits; the LLM only writes language.

**Why Bi-LSTM and not BERT?** The assignment requires an embedding + LSTM/GRU built from scratch. A Bi-LSTM reads the sentence in both
directions so words at the end can change the meaning of words at the start.

**What does the embedding layer do?** It is a learnable table: each word id maps to a 100-number vector that training adjusts so words used
in similar contexts end up with similar vectors.

**Why `pack_padded_sequence`?** Sentences are padded to 40 tokens. Without packing the backward LSTM would start by reading padding; packing makes it
stop at each sentence's real end. A test proves padding does not change the output.

**Why did the first model score only 72%?** It overfit, and my synthetic data had contradictory labels and repeated phrasing, so the model learned
shortcuts ("starts with 'do you'" means general). I fixed the data (labelling policy, more varied sentences) and added word dropout.

**What is word dropout?** During training, 30% of words are randomly replaced by `<unk>`, so the model cannot rely on one memorised word and learns to handle unknown words.

**How did you avoid data leakage?** Vocabulary from the training split only; duplicates removed; hard-test sentences removed from training;
the split is by *group* so variants of one sentence never sit in different splits; tests check all of this.

**Why is the hard-test score lower, and why report it?** It is small, written in messy natural phrasing and partly ambiguous. It was never used for tuning,
so it is the more honest number. Reporting it truthfully is better than hiding it.

**Why macro-F1 as well as accuracy?** It averages the four intents equally, so a weak small class (feature_request) cannot hide behind the big ones.

**What is a confusion matrix telling you here?** Which intents get mixed up: technical -> general and feature_request -> technical/general are the biggest.
Those classes have the least real data.

**What does RAG add?** Without it the technical agent invented menu paths. With it, answers come from retrieved text, and when nothing relevant
is found the agent says the documentation does not provide the answer.

**How does retrieval work?** Documents are cut into 500-character chunks, embedded with MiniLM, stored in FAISS; a question is embedded and the
3 nearest chunks (cosine similarity above 0.20) are given to the LLM.

**How does memory survive an agent handoff?** There is one shared memory for the whole conversation, not one per agent. Whichever agent runs
receives the last 6 turns.

**What happens when Groq fails?** `call_with_fallback` tries Groq, then local Ollama; if both fail the user gets a friendly message. Errors are
classified into categories; stack traces and keys are never shown.

**What are the known weaknesses?** Short context-dependent messages ("what do I owe?") because each message is classified alone; over-confident
wrong predictions; no real SaaS training data; small hard-test set; the 3B fallback model is less reliable. (All are in the write-up.)

**Is the data real?** Banking77 and CLINC150 are real public datasets. The synthetic sentences, FlowDesk, its documents and all account/status/roadmap
data are fictional, and the synthetic sentences were written with an AI assistant.
