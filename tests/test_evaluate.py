import numpy as np

from src import config
from src.classifier.dataset import prepare_pool, split_by_group
from src.classifier.evaluate import errors_frame, evaluate_df, plot_confusion
from src.classifier.inference import load_checkpoint, predict_probs
from src.classifier.preprocessing import Vocabulary, tokenize
from src.classifier.synthetic_data import build_hard_test, build_synthetic
from src.classifier.train import train
from tests.test_dataset import fake_real


def test_evaluate_df_on_known_predictions():
    import pandas as pd
    df = pd.DataFrame({"text": list("abcd"), "label": ["billing", "billing", "technical", "general"],
                       "source": ["x", "x", "y", "y"]})
    probs = np.eye(4)[[0, 1, 1, 3]]                    # predicts billing, technical, technical, general
    m = evaluate_df(df, probs)
    assert m["accuracy"] == 0.75
    assert m["confusion"][0] == [1, 1, 0, 0]           # one billing sentence was predicted 'technical'
    assert m["per_source_accuracy"] == {"x": 0.5, "y": 1.0}
    assert m["per_class"]["technical"]["recall"] == 1.0 and m["per_class"]["technical"]["precision"] == 0.5
    assert len(errors_frame(df, m)) == 1


def test_checkpoint_roundtrip_and_probabilities(tmp_path):
    pool, _ = prepare_pool(fake_real(), build_synthetic(), build_hard_test(), cap=100)
    splits = split_by_group(pool)
    vocab = Vocabulary.build([tokenize(t) for t in splits["train"]["text"]])
    ckpt = tmp_path / "c.pt"
    train(splits["train"], splits["val"], vocab, epochs=3, checkpoint_path=ckpt, verbose=False)

    clf = load_checkpoint(ckpt)
    assert not clf.model.training                                          # eval mode
    probs = predict_probs(clf, splits["val"]["text"])
    assert probs.shape == (len(splits["val"]), len(config.INTENTS))
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)                  # valid probabilities
    m = evaluate_df(splits["val"], probs)
    assert 0.0 <= m["accuracy"] <= 1.0
    plot_confusion(m["confusion"], "t", tmp_path / "cm.png")
    assert (tmp_path / "cm.png").exists()
