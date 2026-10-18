"""Phase 5: training loop with per-epoch validation and best-checkpoint saving.

Run from the project root:   python -m src.classifier.train
Options:                     --epochs 15 --rnn-type lstm|gru --lr 0.001 --batch-size 64

What happens every epoch:
  1. TRAIN pass : forward -> loss -> backward -> clip gradients -> Adam step
  2. VALIDATE   : forward only (model.eval(), no gradients) on the validation set
  3. If validation macro-F1 is the best so far -> overwrite models/classifier.pt

The TEST set and the HARD TEST set are never touched here.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from src import config
from src.classifier.dataset import IntentDataset
from src.classifier.model import IntentClassifier, count_parameters
from src.classifier.preprocessing import Vocabulary


def compute_class_weights(labels, num_classes=len(config.INTENTS)):
    """Inverse-frequency weights: rare classes (feature_request) get a larger weight in the
    loss, so mistakes on them cost more. weight_c = N / (num_classes * count_c)."""
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    return torch.tensor(len(labels) / (num_classes * np.maximum(counts, 1)), dtype=torch.float)


def apply_word_dropout(ids, p, unk_id, pad_id=0):
    """Randomly replace a fraction `p` of REAL words with <unk> (never the padding).
    Regularisation: the model cannot rely on one memorised word, and it practises on
    unknown words, which it meets at test time (OOV rate was 13-18%)."""
    if p <= 0:
        return ids
    drop = (torch.rand(ids.shape, device=ids.device) < p) & (ids != pad_id)
    return torch.where(drop, torch.full_like(ids, unk_id), ids)


def run_epoch(model, loader, loss_fn, device, optimizer=None, word_dropout=0.0, unk_id=1):
    """One pass over `loader`. Trains if an optimizer is given, otherwise only evaluates."""
    training = optimizer is not None
    model.train(training)
    total_loss, preds, golds = 0.0, [], []
    with torch.set_grad_enabled(training):
        for ids, lengths, labels in loader:
            ids, labels = ids.to(device), labels.to(device)
            if training:
                ids = apply_word_dropout(ids, word_dropout, unk_id)    # training only, never at validation
            logits = model(ids, lengths)
            loss = loss_fn(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)  # guards against exploding gradients in RNNs
                optimizer.step()
            total_loss += loss.item() * len(labels)
            preds.extend(logits.argmax(1).cpu().tolist())
            golds.extend(labels.cpu().tolist())
    return {"loss": total_loss / len(golds), "acc": accuracy_score(golds, preds),
            "macro_f1": f1_score(golds, preds, average="macro", zero_division=0)}


def save_checkpoint(path, model, vocab, epoch, val_metrics):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "vocab": vocab.to_dict(),
        "label_map": dict(config.ID2LABEL),
        "hparams": model.hparams,              # lets inference rebuild the identical architecture
        "max_len": config.MAX_LEN,
        "best_epoch": epoch,
        "val_metrics": val_metrics,
    }, path)


def train(train_df, val_df, vocab, epochs=config.EPOCHS, lr=config.LEARNING_RATE,
          batch_size=config.BATCH_SIZE, rnn_type="lstm", patience=config.PATIENCE,
          checkpoint_path=config.CLASSIFIER_PATH, seed=config.SEED, verbose=True,
          word_dropout=config.WORD_DROPOUT):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, val_ds = IntentDataset(train_df, vocab), IntentDataset(val_df, vocab)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = IntentClassifier(vocab_size=len(vocab), rnn_type=rnn_type).to(device)
    weights = compute_class_weights(train_ds.labels.numpy()).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if verbose:
        print(f"Device: {device} | model: {rnn_type.upper()} | params: {count_parameters(model):,} "
              f"| word dropout: {word_dropout} | seed: {seed}")
        print("Class weights:", {config.ID2LABEL[i]: round(w, 2) for i, w in enumerate(weights.tolist())})
        print(f"\n{'epoch':>5} | {'train loss':>10} {'acc':>6} | {'val loss':>8} {'acc':>6} {'macroF1':>7}")

    history, best_f1, best_epoch, stale = [], -1.0, 0, 0
    for epoch in range(1, epochs + 1):
        tr = run_epoch(model, train_loader, loss_fn, device, optimizer,
                       word_dropout=word_dropout, unk_id=vocab.unk_id)
        va = run_epoch(model, val_loader, loss_fn, device)
        history.append({"epoch": epoch, **{f"train_{k}": v for k, v in tr.items()},
                        **{f"val_{k}": v for k, v in va.items()}})
        improved = va["macro_f1"] > best_f1
        if improved:
            best_f1, best_epoch, stale = va["macro_f1"], epoch, 0
            save_checkpoint(checkpoint_path, model, vocab, epoch, va)
        else:
            stale += 1
        if verbose:
            print(f"{epoch:>5} | {tr['loss']:>10.4f} {tr['acc']:>6.1%} | {va['loss']:>8.4f} "
                  f"{va['acc']:>6.1%} {va['macro_f1']:>7.3f}{'  <- best, saved' if improved else ''}")
        if stale >= patience:
            if verbose:
                print(f"Early stopping: no improvement for {patience} epochs.")
            break
    return history, best_epoch, best_f1


def plot_curves(history, best_epoch, path):
    """Two small panels (loss, accuracy) - train vs validation. Same colours in both panels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    TRAIN, VAL, INK, MUTED, GRID, SURFACE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
    ep = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor=SURFACE)
    for ax, (key, title) in zip(axes, [("loss", "Loss"), ("acc", "Accuracy")]):
        ax.set_facecolor(SURFACE)
        ax.plot(ep, [h[f"train_{key}"] for h in history], color=TRAIN, lw=2, marker="o", ms=4, label="train")
        ax.plot(ep, [h[f"val_{key}"] for h in history], color=VAL, lw=2, marker="o", ms=4, label="validation")
        ax.axvline(best_epoch, color=MUTED, lw=1, ls=":")
        ax.set_title(title, color=INK, loc="left", fontsize=12)
        ax.set_xlabel("epoch", color=MUTED)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.tick_params(colors=MUTED)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.set_xticks(ep if len(ep) <= 15 else ep[:: max(1, len(ep) // 10)])
        if key == "acc":
            ax.set_ylim(0, 1.02)
        ax.legend(frameon=False, labelcolor=INK)
    axes[0].annotate(f"best epoch ({best_epoch})", (best_epoch, axes[0].get_ylim()[1]), color=MUTED,
                     fontsize=9, xytext=(4, -12), textcoords="offset points")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--rnn-type", choices=["lstm", "gru"], default="lstm")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--word-dropout", type=float, default=config.WORD_DROPOUT)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--out", default=str(config.CLASSIFIER_PATH), help="checkpoint file to write")
    args = parser.parse_args()

    vocab = Vocabulary.from_dict(json.loads((config.PROCESSED_DIR / "vocab.json").read_text(encoding="utf-8")))
    train_df = pd.read_csv(config.PROCESSED_DIR / "train.csv")
    val_df = pd.read_csv(config.PROCESSED_DIR / "val.csv")
    print(f"Train rows: {len(train_df)} | Validation rows: {len(val_df)} | Vocab: {len(vocab)}\n")

    history, best_epoch, best_f1 = train(train_df, val_df, vocab, epochs=args.epochs, lr=args.lr,
                                         batch_size=args.batch_size, rnn_type=args.rnn_type,
                                         word_dropout=args.word_dropout, seed=args.seed,
                                         checkpoint_path=Path(args.out))
    config.HISTORY_PATH.write_text(json.dumps(history, indent=2), encoding="utf-8")
    plot_curves(history, best_epoch, config.CURVES_PATH)
    print(f"\nBest epoch: {best_epoch} (val macro-F1 {best_f1:.3f})")
    print(f"Saved checkpoint -> {args.out}")
    print(f"Saved history    -> {config.HISTORY_PATH}")
    print(f"Saved curves     -> {config.CURVES_PATH}")
    print("\nNote: these are VALIDATION numbers. Honest test results come in Phase 6.")


if __name__ == "__main__":
    main()
