import numpy as np
import torch

from src import config
from src.classifier.dataset import prepare_pool, split_by_group
from src.classifier.preprocessing import Vocabulary, tokenize
from src.classifier.synthetic_data import build_hard_test, build_synthetic
from src.classifier.train import compute_class_weights, plot_curves, train
from tests.test_dataset import fake_real


def test_class_weights_favour_rare_classes():
    w = compute_class_weights(np.array([0] * 80 + [1] * 10 + [2] * 5 + [3] * 5))
    assert w[2] > w[1] > w[0] and w[2] == w[3]


def test_train_saves_a_loadable_checkpoint_and_learns(tmp_path):
    pool, _ = prepare_pool(fake_real(), build_synthetic(), build_hard_test(), cap=100)
    splits = split_by_group(pool)
    vocab = Vocabulary.build([tokenize(t) for t in splits["train"]["text"]])
    ckpt = tmp_path / "classifier.pt"

    history, best_epoch, best_f1 = train(splits["train"], splits["val"], vocab, epochs=6,
                                         checkpoint_path=ckpt, verbose=False)

    assert ckpt.exists() and 1 <= best_epoch <= 6
    assert history[-1]["train_loss"] < history[0]["train_loss"]       # it is actually learning
    assert best_f1 > 0.5                                              # far above the 0.25 of random guessing

    saved = torch.load(ckpt, weights_only=True)                       # plain data only: safe to load
    assert {"model_state", "vocab", "label_map", "hparams"} <= set(saved)
    assert saved["label_map"] == config.ID2LABEL
    assert saved["vocab"]["<pad>"] == 0 and saved["vocab"]["<unk>"] == 1

    plot_curves(history, best_epoch, tmp_path / "curves.png")
    assert (tmp_path / "curves.png").stat().st_size > 0
