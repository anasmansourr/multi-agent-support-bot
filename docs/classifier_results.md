# Classifier results (transcribed from the author's terminal runs)

Authoritative raw reports: `models/eval_test.txt`, `models/eval_hard_test.txt`, `models/eval_log.txt`.
Nothing below was invented; every number comes from an executed run.

## Setup
| Item | Value |
|---|---|
| Architecture | Embedding(996, 100) -> Bi-LSTM(128) -> Dropout 0.3 -> Linear(256, 4); 336,148 parameters |
| Training | Adam lr 1e-3, batch 64, class-weighted CrossEntropy, word dropout 0.3, grad clip 5.0, seed 42 |
| Model selection | best validation macro-F1 -> epoch 13 of 15 |
| Data | train 1,866 / val 404 / test 407 / hard test 60; vocabulary 996 words (built from train only) |
| Sources | Banking77 + CLINC150 (real, label-mapped, capped at 300 per intent) + synthetic SaaS sentences (written by the author's AI assistant, labelled `synthetic`) |

## Iteration history (all on the validation set, test untouched until the end)
| Step | Val accuracy | Val macro-F1 | Synthetic slice |
|---|---|---|---|
| First model (old data, no word dropout) | 72.2% | 0.719 | 59.1% |
| + labelling policy, relabelled 7 cores, ~40 new cores per intent (no word dropout) | 79.5% | 0.783 | 72.1% |
| + word dropout 0.15 | 86.1% | 0.863 | 81.4% |
| + word dropout 0.3 (chosen) | 88.9% | 0.889 | 85.4% |
Note: the validation set changed between row 1 and row 2 because the data pool changed.

## Final results
| Split | Accuracy | Macro-F1 |
|---|---|---|
| Test (n=407) | 89.9% | 0.898 |
| Hard test (n=60) | 78.3% | 0.783 |

Test, by source: banking77 100.0% (n=42), clinc150 94.8% (n=97), synthetic 86.6% (n=268).
Mean confidence: correct 0.93, wrong 0.82 (test); correct 0.92, wrong 0.81 (hard test).

### Test: per class
| Intent | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| billing | 0.935 | 0.926 | 0.930 | 108 |
| technical | 0.884 | 0.846 | 0.865 | 117 |
| feature_request | 0.966 | 0.824 | 0.889 | 68 |
| general | 0.854 | 0.974 | 0.910 | 114 |

### Test: confusion matrix (rows = true, columns = predicted)
| | billing | technical | feature_request | general |
|---|---|---|---|---|
| billing | 100 | 4 | 0 | 4 |
| technical | 5 | 99 | 2 | 11 |
| feature_request | 0 | 8 | 56 | 4 |
| general | 2 | 1 | 0 | 111 |

### Hard test: per class
| Intent | Precision | Recall | F1 |
|---|---|---|---|
| billing | 0.800 | 0.800 | 0.800 |
| technical | 0.733 | 0.733 | 0.733 |
| feature_request | 1.000 | 0.667 | 0.800 |
| general | 0.700 | 0.933 | 0.800 |

### Hard test: confusion matrix
| | billing | technical | feature_request | general |
|---|---|---|---|---|
| billing | 12 | 2 | 0 | 1 |
| technical | 2 | 11 | 0 | 2 |
| feature_request | 0 | 2 | 10 | 3 |
| general | 1 | 0 | 0 | 14 |

## Confusion analysis
1. **technical -> general (11 on test), feature_request -> technical (8), feature_request -> general (4).** Technical and
   feature_request are the two classes with the least real data (feature_request has none), so they rely on the
   author's synthetic phrasing. Short, vague bug reports ("emails from your system land in spam") and polite requests
   ("can you build a plugin marketplace") look like general chit-chat or help questions.
2. **Greeting prefixes cause confident mistakes.** "hello, ..." / "hey, ..." / "good morning, ..." are strong `general`
   cues in the training data, so e.g. "hello, what is your cancellation fee" is predicted `general`.
3. **billing <-> general is a genuine policy boundary.** "what is your cancellation fee" is a policy question; our
   labelling policy (docs/labeling_policy.md) calls it billing, but a reasonable person could say general.
4. **feature_request recall is the weak point** (test 82.4%, hard test 66.7%) while its precision is ~97-100%: the model is
   conservative and only calls something a request when it sees familiar request phrasing ("any chance of adding...",
   "wish the app had..." were misses).
5. **Ambiguous by design in the hard test:** "payment went through but my order doesn't show up in the app"
   (billing vs technical), "export the dashboard to excel not just csv" (request vs bug), "when will you support arabic"
   (request vs general). Misses on these are partly label ambiguity, not only model error.
6. **Real-data errors are rare** (banking77 100%, clinc150 94.8%), e.g. "i think my account is block" -> billing.

## Limitations (state these in the write-up)
- Errors are **clustered by core sentence**: 13 of the 15 most confident test mistakes come from only 5 core sentences
  (each core has several greeting/closing variants that sit in the same split). The effective test sample is smaller than
  407, so the 89.9% figure is less certain than n=407 suggests.
- The hard test has 60 sentences: one sentence = 1.7 points; a rough 95% interval around 78.3% is about 67-88%, so it
  is not statistically distinguishable from 80%. It is reported as measured and was NOT used for tuning.
- Confidence is over-confident when wrong (mean 0.82 / 0.81): the 0.60 threshold will catch gibberish and
  out-of-domain text better than it catches confident confusions (only 1 of 13 hard-test mistakes had confidence < 0.60).
- Word dropout (0.3) and the number of epochs were chosen by comparing three settings on the same validation set.
- Synthetic data is written by an AI assistant, not collected from real customers; real data is bank/voice-assistant
  domain, not SaaS.

## What I would improve with more time
Collect real SaaS support tickets; many more real-style feature requests; write a second, fresh hard-test set to
measure again without reuse; calibrate confidence (temperature scaling); compare GRU vs LSTM over several seeds.
