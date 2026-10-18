import pytest
import torch
import torch.nn as nn

from src import config
from src.classifier.model import IntentClassifier, count_parameters

V = 50


def batch(n=6, max_len=config.MAX_LEN):
    g = torch.Generator().manual_seed(0)
    lengths = torch.randint(1, max_len + 1, (n,), generator=g)
    ids = torch.zeros(n, max_len, dtype=torch.long)
    for i, L in enumerate(lengths):
        ids[i, :L] = torch.randint(2, V, (int(L),), generator=g)
    return ids, lengths


@pytest.mark.parametrize("rnn_type", ["lstm", "gru"])
@pytest.mark.parametrize("bidirectional", [True, False])
def test_output_shape(rnn_type, bidirectional):
    model = IntentClassifier(V, rnn_type=rnn_type, bidirectional=bidirectional)
    ids, lengths = batch()
    assert model(ids, lengths).shape == (6, len(config.INTENTS))


def test_padding_does_not_change_the_prediction():
    """The same sentence padded to 10 or to 40 positions must give identical logits."""
    torch.manual_seed(0)
    model = IntentClassifier(V).eval()
    sentence = torch.tensor([[5, 9, 3, 7]])
    short = torch.cat([sentence, torch.zeros(1, 6, dtype=torch.long)], dim=1)
    long = torch.cat([sentence, torch.zeros(1, 36, dtype=torch.long)], dim=1)
    lengths = torch.tensor([4])
    with torch.no_grad():
        assert torch.allclose(model(short, lengths), model(long, lengths), atol=1e-5)


def test_pad_embedding_is_zero_and_not_trained():
    model = IntentClassifier(V)
    assert torch.all(model.embedding.weight[0] == 0)
    ids, lengths = batch()
    nn.CrossEntropyLoss()(model(ids, lengths), torch.zeros(6, dtype=torch.long)).backward()
    assert torch.all(model.embedding.weight.grad[0] == 0)


def test_can_overfit_a_tiny_batch():
    """If wiring is right, the model must be able to memorise 8 sentences."""
    torch.manual_seed(0)
    model = IntentClassifier(V)
    ids, lengths = batch(8)
    labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()
    first = None
    for _ in range(120):
        opt.zero_grad()
        loss = loss_fn(model(ids, lengths), labels)
        first = first if first is not None else loss.item()
        loss.backward()
        opt.step()
    assert first > 1.0 and loss.item() < 0.1


def test_hparams_can_rebuild_the_same_architecture():
    a = IntentClassifier(V, rnn_type="gru", hidden_dim=32)
    b = IntentClassifier(**a.hparams)
    b.load_state_dict(a.state_dict())                  # would raise if shapes differed
    assert count_parameters(a) == count_parameters(b)


def test_invalid_rnn_type():
    with pytest.raises(ValueError):
        IntentClassifier(V, rnn_type="transformer")
