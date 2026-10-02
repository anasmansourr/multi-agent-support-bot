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
