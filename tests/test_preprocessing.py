from src import config
from src.classifier.preprocessing import Vocabulary, clean_text, tokenize


def test_clean_text_lowercases_and_strips_punctuation():
    assert clean_text("Hello,  WORLD!!") == "hello world"


def test_apostrophes_are_deleted_not_split():
    assert tokenize("Don't") == tokenize("dont") == ["dont"]


def test_numbers_become_num_token():
    assert tokenize("I owe $500, order 4821") == ["i", "owe", "<num>", "order", "<num>"]


def test_vocab_reserved_ids_and_unknown_words():
    vocab = Vocabulary.build([["refund", "refund", "please", "please"]], min_freq=2)
    assert vocab.stoi[config.PAD_TOKEN] == 0 and vocab.stoi[config.UNK_TOKEN] == 1
    ids, _ = vocab.encode(["refund", "zebra"], max_len=4)
    assert ids[1] == vocab.unk_id                    # unseen word -> <unk>


def test_vocab_uses_min_freq_and_max_size():
    toks = [["a", "a", "b", "b", "c"]]
    assert "c" not in Vocabulary.build(toks, min_freq=2).stoi
    assert len(Vocabulary.build(toks, max_size=3, min_freq=1)) == 3


def test_encode_pads_and_truncates():
    vocab = Vocabulary.build([["a", "b"] * 3], min_freq=1)
    ids, n = vocab.encode(["a", "b"], max_len=5)
    assert len(ids) == 5 and n == 2 and ids[2:] == [0, 0, 0]
    ids, n = vocab.encode(["a"] * 9, max_len=5)
    assert len(ids) == 5 and n == 5


def test_empty_input_is_safe():
    vocab = Vocabulary.build([["a"] * 3], min_freq=1)
    ids, n = vocab.encode([], max_len=4)
    assert n == 1 and ids[0] == vocab.unk_id


def test_vocab_roundtrip():
    vocab = Vocabulary.build([["a", "b"] * 3], min_freq=1)
    assert Vocabulary.from_dict(vocab.to_dict()).stoi == vocab.stoi
