"""Phase 2: download Banking77 + CLINC150, map their labels to our 4 intents,
and save the REAL (non-synthetic) part of the dataset.

Run from the project root:
    python -m src.classifier.data_acquisition --list-labels   # inspect label names
    python -m src.classifier.data_acquisition                 # build real_mapped.csv

Output: data/processed/real_mapped.csv  with columns
    text, label, source, original_label
Nothing is split yet. Splitting happens once, later, for the whole blended dataset
(so we control data leakage in one place).
"""
import argparse

import pandas as pd

from src import config
from src.classifier.label_mapping import BANKING77_TO_INTENT, CLINC150_TO_INTENT


# --------------------------------------------------------------- pure logic
def apply_mapping(rows, mapping, source):
    """rows: iterable of (text, original_label_name). Returns a DataFrame with only
    the rows whose label is in `mapping` (matched case-insensitively)."""
    mapping = {k.lower(): v for k, v in mapping.items()}
    bad_targets = set(mapping.values()) - set(config.INTENTS)
    if bad_targets:
        raise ValueError(f"[{source}] mapping targets unknown intents: {bad_targets}")

    out = [
        {"text": text, "label": mapping[name.lower()], "source": source, "original_label": name}
        for text, name in rows
        if name.lower() in mapping
    ]
    return pd.DataFrame(out, columns=["text", "label", "source", "original_label"])


def validate_mapping_names(mapping, available_names, source):
    """Fail loudly if the mapping uses a label that does not exist in the dataset."""
    available = {n.lower() for n in available_names}
    unknown = sorted(k for k in mapping if k.lower() not in available)
    if unknown:
        raise ValueError(
            f"[{source}] these mapped labels do not exist in the dataset: {unknown}\n"
            f"Available labels: {sorted(available)}"
        )


def clean_duplicates(df):
    """Drop exact duplicates, and drop texts that appear with CONFLICTING labels."""
    df = df.copy()
    df["_key"] = df["text"].str.lower().str.strip()
    n_labels = df.groupby("_key")["label"].transform("nunique")
    df = df[n_labels == 1]
    df = df.drop_duplicates(subset=["_key"]).drop(columns="_key")
    return df.reset_index(drop=True)


# ------------------------------------------------------------ dataset loading
def _extract_rows(ds_dict, label_cols=("label", "intent")):
    """Pool ALL splits of a Hugging Face DatasetDict into (text, label_name) pairs.
    We pool because we create our own train/val/test split later."""
    rows, names = [], set()
    for split in ds_dict.values():
        cols = split.column_names
        if "label_text" in cols:                       # some mirrors store the name directly
            label_names = split["label_text"]
        else:
            col = next(c for c in label_cols if c in cols)
            feature = split.features[col]
            label_names = [feature.int2str(i) for i in split[col]]
        rows.extend(zip(split["text"], label_names))
        names.update(label_names)
    return rows, sorted(names)


def _try_load(candidates):
    from datasets import load_dataset
    errors = []
    for args, kwargs in candidates:
        try:
            return load_dataset(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{args} {kwargs}: {type(exc).__name__}: {str(exc)[:150]}")
    raise RuntimeError("Could not load dataset. Tried:\n  " + "\n  ".join(errors))


def load_banking77():
    ds = _try_load([
        (("PolyAI/banking77",), {}),
        (("PolyAI/banking77",), {"trust_remote_code": True}),
        (("mteb/banking77",), {}),
    ])
    return _extract_rows(ds)


def load_clinc150():
    ds = _try_load([
        (("clinc_oos", "plus"), {}),
        (("clinc/clinc_oos", "plus"), {}),
    ])
    return _extract_rows(ds)


# ------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-labels", action="store_true",
                        help="only print/save the label names of both datasets")
    args = parser.parse_args()

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading Banking77 ...")
    b_rows, b_names = load_banking77()
    print("Downloading CLINC150 ...")
    c_rows, c_names = load_clinc150()
    print(f"Banking77: {len(b_rows)} rows, {len(b_names)} labels")
    print(f"CLINC150 : {len(c_rows)} rows, {len(c_names)} labels")

    (config.RAW_DIR / "banking77_labels.txt").write_text("\n".join(b_names), encoding="utf-8")
    (config.RAW_DIR / "clinc150_labels.txt").write_text("\n".join(c_names), encoding="utf-8")
    if args.list_labels:
        print("\nBanking77 labels:\n ", ", ".join(b_names))
        print("\nCLINC150 labels:\n ", ", ".join(c_names))
        return

    validate_mapping_names(BANKING77_TO_INTENT, b_names, "banking77")
    validate_mapping_names(CLINC150_TO_INTENT, c_names, "clinc150")

    df = pd.concat([
        apply_mapping(b_rows, BANKING77_TO_INTENT, "banking77"),
        apply_mapping(c_rows, CLINC150_TO_INTENT, "clinc150"),
    ], ignore_index=True)

    before = len(df)
    df = clean_duplicates(df)
    print(f"\nKept {before} mapped rows; {len(df)} after removing duplicates/conflicts.")

    out_path = config.PROCESSED_DIR / "real_mapped.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}\n")

    print("Rows per intent x source (REAL data only):")
    print(pd.crosstab(df["label"], df["source"], margins=True))
    print("\nRows per original label:")
    print(df.groupby(["label", "original_label"]).size())
    print("\nRandom sample:")
    print(df.sample(min(12, len(df)), random_state=config.SEED)[["text", "label"]].to_string(index=False))


if __name__ == "__main__":
    main()
