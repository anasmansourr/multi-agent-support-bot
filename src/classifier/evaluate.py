"""Phase 6: honest evaluation of the saved classifier.

    python -m src.classifier.evaluate --split val         # use this while diagnosing / improving
    python -m src.classifier.evaluate --split test        # FINAL number - run once the model is frozen
    python -m src.classifier.evaluate --split hard_test   # hand-written, messy, partly ambiguous

Reports: accuracy, macro-F1, per-class precision/recall/F1, confusion matrix, accuracy per data
source, the most confident mistakes, and writes a confusion-matrix PNG + an errors CSV.
Every run on `test` or `hard_test` is appended to models/eval_log.txt (an audit trail).
"""
import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from src import config
from src.classifier.inference import load_checkpoint, predict_probs

TARGET_ACC = 0.80


def evaluate_df(df, probs):
    """Pure function: DataFrame (with text,label,source) + probabilities -> metrics dict."""
    labels = list(config.INTENTS)
    y_true = df["label"].map(config.LABEL2ID).to_numpy()
    y_pred = probs.argmax(1)
    conf = probs.max(1)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=range(len(labels)), zero_division=0)
    per_source = {src: float((y_pred[m] == y_true[m]).mean()) for src, m in
                  ((src, (df["source"] == src).to_numpy()) for src in df["source"].unique())}
    return {
        "n": int(len(df)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class": {labels[i]: {"precision": float(p[i]), "recall": float(r[i]),
                                  "f1": float(f[i]), "support": int(s[i])} for i in range(len(labels))},
        "confusion": confusion_matrix(y_true, y_pred, labels=range(len(labels))).tolist(),
        "per_source_accuracy": per_source,
        "mean_conf_correct": float(conf[y_pred == y_true].mean()) if (y_pred == y_true).any() else 0.0,
        "mean_conf_wrong": float(conf[y_pred != y_true].mean()) if (y_pred != y_true).any() else 0.0,
        "y_pred": y_pred, "conf": conf,
    }


def plot_confusion(cm, title, path):
    """Heatmap: colour = share of the TRUE class (row), number = count. One-hue blue ramp."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
    cmap = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"])
    cm = np.array(cm)
    share = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6.4, 5.4), facecolor=SURFACE)
    ax.imshow(share, cmap=cmap, vmin=0, vmax=1)
    names = list(config.INTENTS)
    ax.set_xticks(range(len(names)), names, color=MUTED, rotation=20)
    ax.set_yticks(range(len(names)), names, color=MUTED)
    ax.set_xlabel("predicted intent", color=MUTED)
    ax.set_ylabel("true intent", color=MUTED)
    ax.set_title(title, color=INK, loc="left", fontsize=12)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{cm[i, j]}\n{share[i, j]:.0%}", ha="center", va="center", fontsize=10,
                    color="white" if share[i, j] > 0.45 else INK)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def print_report(df, m, split):
    names = list(config.INTENTS)
    print(f"\n=== {split.upper()}  (n={m['n']}) ===")
    print(f"Accuracy : {m['accuracy']:.1%}   (course target >= {TARGET_ACC:.0%}: "
          f"{'MET' if m['accuracy'] >= TARGET_ACC else 'NOT MET'})")
    print(f"Macro-F1 : {m['macro_f1']:.3f}")
    print("\nPer-class:")
    print(pd.DataFrame(m["per_class"]).T.round(3).to_string())
    print("\nConfusion matrix (rows = TRUE, columns = PREDICTED):")
    print(pd.DataFrame(m["confusion"], index=names, columns=names).to_string())
    print("\nAccuracy by data source:")
    for src, acc in sorted(m["per_source_accuracy"].items()):
        print(f"  {src:10s} {acc:.1%}  (n={(df['source'] == src).sum()})")
    print(f"\nMean confidence when correct: {m['mean_conf_correct']:.2f} | when wrong: {m['mean_conf_wrong']:.2f}")

    cm = np.array(m["confusion"])
    pairs = [(cm[i, j], names[i], names[j]) for i in range(len(names)) for j in range(len(names)) if i != j and cm[i, j]]
    print("\nLargest confusions (true -> predicted):")
    for n, t, p in sorted(pairs, reverse=True)[:5]:
        print(f"  {t:16s} -> {p:16s} {n} sentences")


def errors_frame(df, m):
    out = df.copy()
    out["predicted"] = [config.ID2LABEL[i] for i in m["y_pred"]]
    out["confidence"] = m["conf"].round(3)
    return out[out["predicted"] != out["label"]].sort_values("confidence", ascending=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test", "hard_test"], default="val")
    parser.add_argument("--checkpoint", default=str(config.CLASSIFIER_PATH))
    args = parser.parse_args()

    clf = load_checkpoint(args.checkpoint)
    df = pd.read_csv(config.PROCESSED_DIR / f"{args.split}.csv")
    probs = predict_probs(clf, df["text"])
    m = evaluate_df(df, probs)
    print(f"Checkpoint: best epoch {clf.meta['best_epoch']}")
    print_report(df, m, args.split)

    errs = errors_frame(df, m)
    print(f"\nMost confident mistakes ({len(errs)} total, showing up to 15):")
    for _, r in errs.head(15).iterrows():
        print(f"  [{r['confidence']:.2f}] {r['text'][:70]!r}\n         true={r['label']}  predicted={r['predicted']}  source={r['source']}")

    errs.to_csv(config.MODELS_DIR / f"errors_{args.split}.csv", index=False)
    png = config.MODELS_DIR / f"confusion_matrix_{args.split}.png"
    plot_confusion(m["confusion"], f"Confusion matrix - {args.split}", png)
    print(f"\nSaved {png.name} and errors_{args.split}.csv in {config.MODELS_DIR}")

    if args.split in ("test", "hard_test"):
        with open(config.MODELS_DIR / "eval_log.txt", "a", encoding="utf-8") as f:
            f.write(f"{datetime.now():%Y-%m-%d %H:%M} split={args.split} acc={m['accuracy']:.4f} "
                    f"macro_f1={m['macro_f1']:.4f} best_epoch={clf.meta['best_epoch']}\n")
        print("Logged to eval_log.txt. Do NOT tune the model on this split.")


if __name__ == "__main__":
    main()
