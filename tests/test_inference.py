import pytest

from src import config
from src.classifier.dataset import prepare_pool, split_by_group
from src.classifier.inference import IntentPredictor, load_checkpoint, predict_probs
from src.classifier.preprocessing import Vocabulary, tokenize
from src.classifier.synthetic_data import build_hard_test, build_synthetic
from src.classifier.train import train
from tests.test_dataset import fake_real


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    path = tmp_path_factory.mktemp("ckpt") / "classifier.pt"
    pool, _ = prepare_pool(fake_real(), build_synthetic(), build_hard_test(), cap=100)
    splits = split_by_group(pool)
    vocab = Vocabulary.build([tokenize(t) for t in splits["train"]["text"]])
    train(splits["train"], splits["val"], vocab, epochs=4, checkpoint_path=path, verbose=False)
    return path


def test_predict_returns_required_keys_and_valid_values(checkpoint):
    out = IntentPredictor(checkpoint).predict("i was charged twice for my subscription")
    assert {"intent", "confidence"} <= set(out)
    assert out["intent"] in config.INTENTS
    assert 0.0 <= out["confidence"] <= 1.0
    assert abs(sum(out["scores"].values()) - 1.0) < 1e-3
    assert out["confidence"] == max(out["scores"].values())


def test_prediction_is_deterministic(checkpoint):
    p = IntentPredictor(checkpoint)
    assert p.predict("the app crashes on startup") == p.predict("the app crashes on startup")


def test_predictor_matches_the_batch_evaluation_path(checkpoint):
    clf = load_checkpoint(checkpoint)
    text = "please add a dark mode"
    probs = predict_probs(clf, [text])[0]
    out = IntentPredictor(checkpoint).predict(text)
    assert out["intent"] == config.ID2LABEL[int(probs.argmax())]


@pytest.mark.parametrize("bad", ["", "   ", "!!!", None, 123])
def test_odd_input_does_not_crash_and_gets_zero_confidence(checkpoint, bad):
    out = IntentPredictor(checkpoint).predict(bad)
    assert out["intent"] == "general" and out["confidence"] == 0.0


def test_unknown_words_and_very_long_text_are_handled(checkpoint):
    p = IntentPredictor(checkpoint)
    assert p.predict("zzzz qqqq xxxx")["intent"] in config.INTENTS
    assert p.predict("refund " * 500)["intent"] in config.INTENTS


def test_missing_checkpoint_gives_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="python -m src.classifier.train"):
        IntentPredictor(tmp_path / "nope.pt")
