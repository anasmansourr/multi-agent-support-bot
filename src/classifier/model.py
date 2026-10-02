"""The custom intent classifier, built from scratch in PyTorch (no pretrained weights).

    token ids  (batch, 40)
        -> Embedding            (batch, 40, 100)   each word id -> a learnable 100-number vector
        -> Bi-LSTM / Bi-GRU     reads the sentence left->right AND right->left
        -> final hidden states  (batch, 256)       forward (128) + backward (128), concatenated
        -> Dropout(0.3)
        -> Linear               (batch, 4)         one score ("logit") per intent
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from src import config


class IntentClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=config.EMBED_DIM, hidden_dim=config.HIDDEN_DIM,
                 num_classes=len(config.INTENTS), n_layers=config.N_LAYERS,
                 bidirectional=config.BIDIRECTIONAL, dropout=config.DROPOUT,
                 rnn_type="lstm", pad_idx=0):
        super().__init__()
        if rnn_type not in ("lstm", "gru"):
            raise ValueError("rnn_type must be 'lstm' or 'gru'")
        # Saved inside classifier.pt so inference can rebuild the EXACT same architecture.
        self.hparams = dict(vocab_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim,
                            num_classes=num_classes, n_layers=n_layers, bidirectional=bidirectional,
                            dropout=dropout, rnn_type=rnn_type, pad_idx=pad_idx)
        self.bidirectional = bidirectional

        # padding_idx keeps the <pad> vector at zero and never updates it.
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(embed_dim, hidden_dim, num_layers=n_layers, batch_first=True,
                           bidirectional=bidirectional,
                           dropout=dropout if n_layers > 1 else 0.0)   # inner dropout only exists between layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), num_classes)

    def forward(self, ids, lengths):
        """ids: (batch, max_len) long. lengths: (batch,) true lengths before padding.
        Returns logits (batch, num_classes). Softmax is applied later (at inference),
        because CrossEntropyLoss expects raw logits."""
        emb = self.embedding(ids)
        # Packing tells the RNN to stop at each sentence's real end, so padding never
        # influences the result (especially important for the backward direction).
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.rnn(packed)
        h_n = hidden[0] if isinstance(hidden, tuple) else hidden        # LSTM returns (h_n, c_n); GRU returns h_n
        # h_n: (n_layers * directions, batch, hidden). The last layer's states are at the end.
        features = torch.cat([h_n[-2], h_n[-1]], dim=1) if self.bidirectional else h_n[-1]
        return self.fc(self.dropout(features))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------- demo / sanity
def main():
    """Run:  python -m src.classifier.model
    1) prints the architecture, 2) shows tensor shapes, 3) tries to memorise 32 real
    training sentences. If the model + data are wired correctly the loss must fall to ~0."""
    import json

    import pandas as pd
    from torch.utils.data import DataLoader

    from src.classifier.dataset import IntentDataset
    from src.classifier.preprocessing import Vocabulary

    torch.manual_seed(config.SEED)
    vocab = Vocabulary.from_dict(json.loads((config.PROCESSED_DIR / "vocab.json").read_text(encoding="utf-8")))
    model = IntentClassifier(vocab_size=len(vocab))
    print(model)
    print(f"\nTrainable parameters: {count_parameters(model):,}")

    train = pd.read_csv(config.PROCESSED_DIR / "train.csv").sample(32, random_state=config.SEED)
    ds = IntentDataset(train, vocab)
    ids, lengths, labels = next(iter(DataLoader(ds, batch_size=32)))
    logits = model(ids, lengths)
    print(f"\nShapes: ids {tuple(ids.shape)} -> logits {tuple(logits.shape)}")

    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    print(f"\nSanity check: memorise 32 sentences (random guessing would give loss = ln(4) = 1.386)")
    for step in range(1, 151):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(ids, lengths), labels)
        loss.backward()
        opt.step()
        if step in (1, 25, 50, 100, 150):
            print(f"  step {step:3d}  loss {loss.item():.4f}")
    model.eval()
    with torch.no_grad():
        acc = (model(ids, lengths).argmax(1) == labels).float().mean().item()
    print(f"Accuracy on those 32 sentences after training on them: {acc:.0%}  (this is NOT a real result)")


if __name__ == "__main__":
    main()
