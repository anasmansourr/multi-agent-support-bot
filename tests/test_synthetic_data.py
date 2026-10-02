import re

from src import config
from src.classifier.synthetic_data import build_hard_test, build_synthetic


def _norm(t):
    return re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()


def test_all_intents_present_with_enough_rows():
    df = build_synthetic()
    counts = df["label"].value_counts()
    assert set(counts.index) == set(config.INTENTS)
    assert counts.min() >= 200


def test_everything_is_marked_synthetic():
    assert set(build_synthetic()["source"]) == {"synthetic"}
    assert set(build_hard_test()["source"]) == {"hard_test"}


def test_hard_test_not_leaked_into_training_pool():
    train = {_norm(t) for t in build_synthetic()["text"]}
    hard = {_norm(t) for t in build_hard_test()["text"]}
    assert not (train & hard)


def test_no_text_has_two_labels():
    df = build_synthetic()
    df["k"] = df["text"].map(_norm)
    assert (df.groupby("k")["label"].nunique() == 1).all()


def test_hard_test_covers_all_intents():
    assert build_hard_test()["label"].value_counts().min() >= 12


def test_generation_is_deterministic():
    assert build_synthetic().equals(build_synthetic())
