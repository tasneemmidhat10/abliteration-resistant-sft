import torch

from arft.eval_refusal import is_refusal
from arft.interventions import DirectionAblation


class Toy(torch.nn.Module):  # mimics model.model.{embed_tokens, layers}
    def __init__(self, d=8):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(10, d)
        self.model.layers = torch.nn.ModuleList([torch.nn.Linear(d, d) for _ in range(2)])

    def forward(self, x):
        h = self.model.embed_tokens(x)
        for layer in self.model.layers:
            h = layer(h)
        return h


def test_full_ablation_removes_direction():
    torch.manual_seed(0)
    m, r = Toy(), torch.randn(8)
    with DirectionAblation(r, 1.0).attach(m):
        out = m(torch.tensor([[1, 2, 3]]))
    assert torch.allclose(out @ (r / r.norm()), torch.zeros(1, 3), atol=1e-5)


def test_partial_ablation_scales_projection():
    torch.manual_seed(0)
    m, r = Toy(), torch.randn(8)
    r_hat = r / r.norm()
    h = torch.randn(3, 8)
    out = DirectionAblation(r, 0.25)._hook(None, None, h)
    assert torch.allclose(out @ r_hat, 0.75 * (h @ r_hat), atol=1e-5)


def test_hooks_are_removed():
    m, r = Toy(), torch.randn(8)
    x = torch.tensor([[1, 2]])
    before = m(x)
    with DirectionAblation(r, 1.0).attach(m):
        pass
    assert torch.equal(before, m(x))


def test_tuple_outputs_are_supported():
    r = torch.randn(8)
    h, extra = torch.randn(2, 8), object()
    out = DirectionAblation(r, 1.0)._hook(None, None, (h, extra))
    assert isinstance(out, tuple) and out[1] is extra


def test_refusal_matcher():
    assert is_refusal("I’m sorry, but I can’t help with that.")  # curly apostrophes
    assert not is_refusal("Sure! Here is a recipe for pancakes.")


def test_activation_addition_targets_one_layer():
    from arft.interventions import ActivationAddition

    torch.manual_seed(0)
    m, v = Toy(), torch.randn(8)
    x = torch.tensor([[1, 2]])
    h0 = m.model.embed_tokens(x)
    h1 = m.model.layers[0](h0)
    with ActivationAddition(v, 1).attach(m):  # hidden_states[1] = output of layers[0]
        out = m(x)
    assert torch.allclose(out, m.model.layers[1](h1 + v), atol=1e-5)
