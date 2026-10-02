import pandas as pd
import pytest

from src import config
from src.classifier.dataset import IntentDataset, assert_no_leakage, cap_balanced, prepare_pool, split_by_group
from src.classifier.preprocessing import Vocabulary, tokenize
from src.classifier.synthetic_data import build_hard_test, build_synthetic


def fake_real():
    """Stand-in for Banking77/CLINC rows (the real files are built on the student's laptop)."""
    rows = []
    for intent in config.INTENTS:
        for lab in range(3):
            for i in range(120):
                rows.append({"text": f"{intent} real sample {lab} number {i} words", "label": intent,
                             "source": "fake_real", "original_label": f"{intent}_{lab}"})
    return pd.DataFrame(rows)


def test_cap_balanced_respects_cap_and_spreads_over_labels():
    capped = cap_balanced(fake_real().query("label == 'billing'"), 90)
    assert len(capped) == 90
    assert capped["original_label"].value_counts().nunique() == 1      # 30 / 30 / 30


def build():
    hard = build_hard_test()
    pool, rep = prepare_pool(fake_real(), build_synthetic(), hard, cap=100)
    return pool, hard, rep


def test_pool_has_no_hard_test_sentences():
    pool, hard, _ = build()
    from src.classifier.preprocessing import clean_text
    assert not set(pool["clean"]) & set(hard["text"].map(clean_text))


def test_split_is_leak_free_and_stratified():
    pool, hard, _ = build()
    splits = split_by_group(pool)
    assert_no_leakage(splits, hard)                                     # raises if any leak
    for intent in config.INTENTS:
        n = {s: (d["label"] == intent).sum() for s, d in splits.items()}
        total = sum(n.values())
        assert 0.10 < n["val"] / total < 0.22 and 0.10 < n["test"] / total < 0.22


def test_split_is_deterministic():
    pool, _, _ = build()
    a, b = split_by_group(pool), split_by_group(pool)
    assert a["train"]["text"].equals(b["train"]["text"])


def test_vocab_built_from_train_only_never_contains_test_only_words():
    pool, _, _ = build()
    splits = split_by_group(pool)
    vocab = Vocabulary.build([tokenize(t) for t in splits["train"]["text"]], min_freq=1)
    train_words = {w for t in splits["train"]["text"] for w in tokenize(t)}
    assert set(vocab.stoi) - {config.PAD_TOKEN, config.UNK_TOKEN} == train_words


def test_torch_dataset_shapes():
    torch = pytest.importorskip("torch")
    pool, _, _ = build()
    train = split_by_group(pool)["train"]
    vocab = Vocabulary.build([tokenize(t) for t in train["text"]])
    ds = IntentDataset(train, vocab)
    ids, length, label = ds[0]
    assert ids.shape == (config.MAX_LEN,) and ids.dtype == torch.long
    assert 1 <= length <= config.MAX_LEN and 0 <= label < len(config.INTENTS)
