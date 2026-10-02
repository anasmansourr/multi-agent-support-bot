"""Phase 3: build leak-free train / val / test splits + vocabulary, and the PyTorch Dataset.

Run from the project root:   python -m src.classifier.dataset

Pipeline (the order matters - this is what prevents data leakage):
  1. load REAL mapped data, cap it per intent        (balance)
  2. load SYNTHETIC data
  3. clean text, drop duplicates / conflicting labels  (so no sentence has two labels)
  4. remove anything that equals a HARD TEST sentence  (keeps the final exam unseen)
  5. split BY GROUP into train/val/test                (variants of one sentence stay together)
  6. build the vocabulary from TRAIN ONLY
  7. save CSVs + vocab and print a report
"""
import json
import random

import numpy as np
import pandas as pd

from src import config
from src.classifier.preprocessing import Vocabulary, clean_text, tokenize

COLUMNS = ["text", "label", "source", "original_label", "group"]


# ------------------------------------------------------------------ steps 1-4
def cap_balanced(df, cap, seed=config.SEED):
    """Keep at most `cap` rows, taking an as-equal-as-possible share from each original label."""
    if len(df) <= cap:
        return df
    by_label = {k: g.sample(frac=1, random_state=seed) for k, g in df.groupby("original_label")}
    keys = sorted(by_label, key=lambda k: len(by_label[k]))
    remaining, take = cap, {}
    for i, k in enumerate(keys):                       # smallest groups first; leftovers go to bigger ones
        take[k] = min(len(by_label[k]), remaining // (len(keys) - i))
        remaining -= take[k]
    return pd.concat([by_label[k].head(take[k]) for k in keys])


def prepare_pool(real, synthetic, hard, cap=config.REAL_CAP_PER_INTENT, seed=config.SEED):
    """Combine real + synthetic into one clean pool. Returns (pool, report_dict)."""
    real = real.reset_index(drop=True).copy()
    real["group"] = [f"real_{i}" for i in range(len(real))]          # every real row is its own group
    real = pd.concat([cap_balanced(g, cap, seed) for _, g in real.groupby("label")])

    pool = pd.concat([real[COLUMNS], synthetic[COLUMNS]], ignore_index=True)
    pool["clean"] = pool["text"].map(clean_text)
    n0 = len(pool)

    pool = pool[pool["clean"] != ""]
    conflicting = pool.groupby("clean")["label"].transform("nunique") > 1       # same text, 2 labels
    pool = pool[~conflicting].drop_duplicates(subset="clean")
    n1 = len(pool)

    hard_clean = set(hard["text"].map(clean_text))
    pool = pool[~pool["clean"].isin(hard_clean)]
    report = {"start": n0, "after_dedupe": n1, "after_hard_removal": len(pool)}
    return pool.reset_index(drop=True), report


# --------------------------------------------------------------------- step 5
def split_by_group(pool, val_frac=config.VAL_FRAC, test_frac=config.TEST_FRAC, seed=config.SEED):
    """Stratified (per intent) split where a whole `group` goes to exactly one split."""
    rng = random.Random(seed)
    assignment = {}
    for _, sub in pool.groupby("label"):
        sizes = sub.groupby("group").size()
        groups = sorted(sizes.index)
        rng.shuffle(groups)
        target = {"test": test_frac * len(sub), "val": val_frac * len(sub)}
        counts = {"test": 0, "val": 0, "train": 0}
        for g in groups:
            split = "test" if counts["test"] < target["test"] else "val" if counts["val"] < target["val"] else "train"
            assignment[g] = split
            counts[split] += sizes[g]
    split_col = pool["group"].map(assignment)
    return {s: pool[split_col == s].reset_index(drop=True) for s in ("train", "val", "test")}


def assert_no_leakage(splits, hard):
    """Hard guarantees. Raises AssertionError if any rule is broken."""
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not set(splits[a]["group"]) & set(splits[b]["group"]), f"group leak {a}/{b}"
            assert not set(splits[a]["clean"]) & set(splits[b]["clean"]), f"text leak {a}/{b}"
    hard_clean = set(hard["text"].map(clean_text))
    for name, df in splits.items():
        assert not set(df["clean"]) & hard_clean, f"hard-test sentence found in {name}"


# --------------------------------------------------------------------- PyTorch
class IntentDataset:
    """Wraps a DataFrame as a torch Dataset: item = (ids[max_len], length, label_id)."""

    def __init__(self, df, vocab, max_len=config.MAX_LEN):
        import torch
        encoded = [vocab.encode(tokenize(t), max_len) for t in df["text"]]
        self.ids = torch.tensor([e[0] for e in encoded], dtype=torch.long)
        self.lengths = torch.tensor([e[1] for e in encoded], dtype=torch.long)
        self.labels = torch.tensor([config.LABEL2ID[l] for l in df["label"]], dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.ids[i], self.lengths[i], self.labels[i]


# ------------------------------------------------------------------------ main
def oov_rate(df, vocab):
    toks = [t for text in df["text"] for t in tokenize(text)]
    return float(np.mean([t not in vocab.stoi for t in toks])) if toks else 0.0


def main():
    real = pd.read_csv(config.PROCESSED_DIR / "real_mapped.csv")
    synthetic = pd.read_csv(config.SYNTHETIC_DIR / "synthetic.csv")
    hard = pd.read_csv(config.SYNTHETIC_DIR / "hard_test.csv")

    pool, rep = prepare_pool(real, synthetic, hard)
    print(f"Pool: {rep['start']} rows -> {rep['after_dedupe']} after de-dup/conflict removal "
          f"-> {rep['after_hard_removal']} after removing hard-test sentences")

    splits = split_by_group(pool)
    assert_no_leakage(splits, hard)
    print("Leakage checks passed (no shared groups, no shared texts, hard test untouched).")

    vocab = Vocabulary.build([tokenize(t) for t in splits["train"]["text"]])   # TRAIN ONLY
    print(f"\nVocabulary size: {len(vocab)} (incl. <pad>=0, <unk>=1)")

    for name, df in splits.items():
        df.drop(columns="clean").to_csv(config.PROCESSED_DIR / f"{name}.csv", index=False)
    hard.to_csv(config.PROCESSED_DIR / "hard_test.csv", index=False)
    (config.PROCESSED_DIR / "vocab.json").write_text(json.dumps(vocab.to_dict(), indent=0), encoding="utf-8")

    print("\nRows per split x intent:")
    print(pd.DataFrame({n: d["label"].value_counts() for n, d in splits.items()}).loc[config.INTENTS]
          .assign(total=lambda x: x.sum(axis=1)).to_string())
    print("\nTRAIN rows per intent x source:")
    print(pd.crosstab(splits["train"]["label"], splits["train"]["source"]).to_string())

    print("\nOut-of-vocabulary rate (share of words mapped to <unk>):")
    for name, df in {**splits, "hard_test": hard}.items():
        print(f"  {name:10s} {oov_rate(df, vocab):.1%}")

    lens = np.array([len(tokenize(t)) for t in splits["train"]["text"]])
    print(f"\nTrain sentence length (words): mean={lens.mean():.1f}  95th pct={np.percentile(lens, 95):.0f}  "
          f"max={lens.max()}  truncated at {config.MAX_LEN}: {(lens > config.MAX_LEN).mean():.1%}")

    ex = splits["train"]["text"].iloc[0]
    ids, n = vocab.encode(tokenize(ex), config.MAX_LEN)
    print(f"\nExample: {ex!r}\n  tokens: {tokenize(ex)}\n  ids   : {ids[:n]} ... (length {n}, padded to {config.MAX_LEN})")
    print(f"\nSaved train/val/test/hard_test CSVs + vocab.json -> {config.PROCESSED_DIR}")


if __name__ == "__main__":
    main()
