"""Loading a saved classifier and turning text into probabilities.

Phase 6 uses `load_checkpoint` + `predict_probs` for evaluation.
Phase 7 will add the friendly `predict(text) -> {"intent", "confidence"}` API on top.
"""
from dataclasses import dataclass

import numpy as np
import torch

from src import config
from src.classifier.model import IntentClassifier
from src.classifier.preprocessing import Vocabulary, tokenize


@dataclass
class LoadedClassifier:
    model: IntentClassifier
    vocab: Vocabulary
    label_map: dict          # {0: "billing", ...}
    max_len: int
    meta: dict               # best_epoch, val_metrics


def load_checkpoint(path=config.CLASSIFIER_PATH):
    """Rebuild the exact architecture from the saved hparams, load the weights, set eval mode."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)    # plain data only -> safe load
    model = IntentClassifier(**ckpt["hparams"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()                                                      # turns dropout off
    return LoadedClassifier(model=model, vocab=Vocabulary.from_dict(ckpt["vocab"]),
                            label_map={int(k): v for k, v in ckpt["label_map"].items()},
                            max_len=ckpt["max_len"],
                            meta={"best_epoch": ckpt["best_epoch"], "val_metrics": ckpt["val_metrics"]})


def predict_probs(clf, texts, batch_size=256):
    """texts -> numpy array (N, num_classes) of softmax probabilities."""
    texts = list(texts)
    if not texts:
        return np.zeros((0, len(clf.label_map)))
    encoded = [clf.vocab.encode(tokenize(t), clf.max_len) for t in texts]
    ids = torch.tensor([e[0] for e in encoded], dtype=torch.long)
    lengths = torch.tensor([e[1] for e in encoded], dtype=torch.long)
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            logits = clf.model(ids[i:i + batch_size], lengths[i:i + batch_size])
            out.append(torch.softmax(logits, dim=1).numpy())          # softmax -> probabilities (confidence)
    return np.concatenate(out)


# =============================================================== Phase 7: public API
class IntentPredictor:
    """Load the trained classifier ONCE, then call .predict(text) as often as you like.

        predictor = IntentPredictor()
        predictor.predict("I was charged twice")
        -> {"intent": "billing", "confidence": 0.96, "scores": {"billing": 0.96, ...}}

    This is the only thing the router needs. It never trains and never calls an LLM.
    """

    def __init__(self, checkpoint_path=config.CLASSIFIER_PATH):
        try:
            self._clf = load_checkpoint(checkpoint_path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Classifier checkpoint not found: {checkpoint_path}\n"
                "Train it first:  python -m src.classifier.train") from exc

    def predict(self, text):
        """text -> {"intent", "confidence", "known_ratio", "scores"}. Never raises on odd input:
        empty / non-string text gives confidence 0.0 so the router's low-confidence rule
        sends it to the general agent."""
        names = [self._clf.label_map[i] for i in range(len(self._clf.label_map))]
        tokens = tokenize(text) if isinstance(text, str) else []
        if not tokens:
            return {"intent": "general", "confidence": 0.0, "known_ratio": 0.0, "scores": {n: 0.0 for n in names}}
        probs = predict_probs(self._clf, [text])[0]
        best = int(probs.argmax())
        known = sum(t in self._clf.vocab.stoi for t in tokens)      # words the classifier has actually seen
        return {"intent": names[best],
                "confidence": round(float(probs[best]), 4),
                "known_ratio": round(known / len(tokens), 3),
                "scores": {n: round(float(p), 4) for n, p in zip(names, probs)}}


def main():
    """python -m src.classifier.inference "i was charged twice"     (one message)
       python -m src.classifier.inference                            (interactive; empty line quits)"""
    import sys
    predictor = IntentPredictor()

    def show(text):
        r = predictor.predict(text)
        scores = "  ".join(f"{k}={v:.2f}" for k, v in r["scores"].items())
        print(f"  intent={r['intent']}  confidence={r['confidence']:.2f}   [{scores}]")

    if len(sys.argv) > 1:
        show(" ".join(sys.argv[1:]))
        return
    print("Type a message (empty line to quit).")
    while True:
        text = input("> ").strip()
        if not text:
            break
        show(text)


if __name__ == "__main__":
    main()
