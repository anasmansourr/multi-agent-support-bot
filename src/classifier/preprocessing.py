"""Text -> tokens -> ids.  No machine learning here, just deterministic text handling.

  clean_text("Don't PAY $50!!")  -> "dont pay 50"
  tokenize("Don't PAY $50!!")    -> ["dont", "pay", "<num>"]
  Vocabulary.build(train_tokens) -> word <-> id table (built from TRAINING data only)
  vocab.encode(tokens, 40)       -> (ids padded/truncated to 40, true length)
"""
import re
from collections import Counter

from src import config

NUM_TOKEN = "<num>"
_APOSTROPHES = re.compile(r"['’`]")
_NON_ALNUM = re.compile(r"[^a-z0-9\s]")


def clean_text(text):
    """Lowercase, delete apostrophes (don't -> dont), turn other punctuation into spaces,
    collapse whitespace. Apostrophes are deleted, not split, so 'dont' (typed without the
    apostrophe, very common in chat) and "don't" end up as the same word."""
    text = str(text).lower()
    text = _APOSTROPHES.sub("", text)
    text = _NON_ALNUM.sub(" ", text)
    return " ".join(text.split())


def tokenize(text):
    """Word-level tokenizer. Pure numbers become <num> so '500' and '49' share one id."""
    return [NUM_TOKEN if tok.isdigit() else tok for tok in clean_text(text).split()]


class Vocabulary:
    """word <-> integer id. Index 0 = <pad>, index 1 = <unk>."""

    def __init__(self, stoi):
        self.stoi = dict(stoi)
        self.itos = {i: w for w, i in self.stoi.items()}

    @classmethod
    def build(cls, token_lists, max_size=config.MAX_VOCAB_SIZE, min_freq=config.MIN_FREQ):
        """Build from TRAINING token lists only (the caller's responsibility; see dataset.py)."""
        counts = Counter(tok for toks in token_lists for tok in toks)
        specials = {config.PAD_TOKEN, config.UNK_TOKEN}
        words = sorted((w for w, c in counts.items() if c >= min_freq and w not in specials),
                       key=lambda w: (-counts[w], w))[: max_size - 2]
        stoi = {config.PAD_TOKEN: 0, config.UNK_TOKEN: 1}
        stoi.update({w: i for i, w in enumerate(words, start=2)})
        return cls(stoi)

    @property
    def pad_id(self):
        return self.stoi[config.PAD_TOKEN]

    @property
    def unk_id(self):
        return self.stoi[config.UNK_TOKEN]

    def __len__(self):
        return len(self.stoi)

    def encode(self, tokens, max_len=config.MAX_LEN):
        """tokens -> (ids of exactly max_len, true length). Unknown words -> <unk>.
        Longer inputs are truncated; shorter are right-padded with <pad>.
        An empty input becomes a single <unk> so the model never sees a zero-length sequence."""
        ids = [self.stoi.get(t, self.unk_id) for t in tokens][:max_len] or [self.unk_id]
        length = len(ids)
        return ids + [self.pad_id] * (max_len - length), length

    def to_dict(self):
        return dict(self.stoi)

    @classmethod
    def from_dict(cls, stoi):
        return cls(stoi)
