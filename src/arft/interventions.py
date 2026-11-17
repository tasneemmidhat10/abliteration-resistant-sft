"""Generic interventions (the attacks of handbook §8). Never touches training code."""

import torch


class DirectionAblation:
    """h <- h - strength * (h . r_hat) r_hat, applied to the residual stream
    after the embedding and after every decoder layer.

    Usage:
        with DirectionAblation(r, 0.5).attach(model):
            model.generate(...)

    Note: in transformers 5, output_hidden_states is also recorded by hooks on
    the decoder layers, so hidden_states read *while ablating* may or may not
    include the ablation depending on hook order. Don't read them under ablation.
    """

    def __init__(self, direction: torch.Tensor, strength: float = 1.0):
        self.r = direction / direction.norm()
        self.strength = strength
        self.handles = []

    def _hook(self, module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        r = self.r.to(h.device, h.dtype)
        h = h - self.strength * (h @ r).unsqueeze(-1) * r
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

    def attach(self, model):
        core = model.model  # Qwen2/Llama: model.model.{embed_tokens, layers}
        self.handles.append(core.embed_tokens.register_forward_hook(self._hook))
        for layer in core.layers:
            self.handles.append(layer.register_forward_hook(self._hook))
        return self

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()


class ActivationAddition(DirectionAblation):
    """h <- h + vector at a single layer (hidden_states[layer]), every position.

    Used by direction selection to check that a candidate *induces* refusal on
    harmless prompts (Arditi et al. 2024).
    """

    def __init__(self, vector: torch.Tensor, layer: int):
        self.v, self.layer, self.handles = vector, layer, []

    def _hook(self, module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        h = h + self.v.to(h.device, h.dtype)
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

    def attach(self, model):
        core = model.model  # hidden_states[0] = embeddings, hidden_states[l] = layers[l-1]
        module = core.embed_tokens if self.layer == 0 else core.layers[self.layer - 1]
        self.handles.append(module.register_forward_hook(self._hook))
        return self
