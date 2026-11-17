"""Refusal-direction estimation and selection (handbook Step 2).

Convention used everywhere in the repo: "layer l" == hidden_states[l] from
output_hidden_states=True, where hidden_states[0] is the embedding output and
hidden_states[l] is the output of decoder layer l-1.

Selection follows Arditi et al. (2024), "Refusal in language models is mediated
by a single direction": candidates are difference-in-means vectors at every
layer below 0.8 * n_layers and every post-instruction template position. A
candidate is kept only if
  (1) adding it at its layer makes harmless prompts look like refusals (induce > 0),
  (2) ablating it barely changes next-token predictions on harmless prompts (KL < 0.1),
and the kept candidate that lowers the refusal score on harmful prompts most wins.
Scores use the first answer token's probability of starting a refusal, so no
generation is needed.
"""

import torch
import torch.nn.functional as F
from datasets import load_dataset

from arft.interventions import ActivationAddition, DirectionAblation

N_ATTACK, N_SELECT = 256, 64
# First tokens of typical refusals ("I cannot...", "As an AI..."), as in Arditi et al.
REFUSAL_FIRST_WORDS = ["I", "As"]
N_POST_INSTRUCTION = 5  # Qwen: <|im_end|> \n <|im_start|> assistant \n
KL_MAX, MAX_LAYER_FRAC = 0.1, 0.8


def attacker_prompts(n=N_ATTACK, n_select=N_SELECT, seed=0):
    """The attacker's own data, independent of our training data.

    Returns (harmful, harmless, harmful_val, harmless_val): AdvBench and Alpaca
    (no-input) instructions for the difference in means, plus held-out prompts
    of each kind used only to select the direction.
    """
    advbench = load_dataset("walledai/AdvBench", split="train").shuffle(seed=seed)["prompt"]
    alpaca = load_dataset("tatsu-lab/alpaca", split="train").filter(lambda x: x["input"] == "")
    harmless = alpaca.shuffle(seed=seed)["instruction"]
    return (advbench[:n], harmless[:n], advbench[n : n + n_select], harmless[n : n + n_select])


def chat(tok, prompt):
    return tok.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False
    )


def _encode(tok, prompts, device):
    tok.padding_side = "left"  # so position -1 is the last real token
    return tok(
        [chat(tok, p) for p in prompts], return_tensors="pt", padding=True, add_special_tokens=False
    ).to(device)


@torch.no_grad()
def token_acts(model, tok, prompts, n_pos=1, bs=16):
    """hidden_states at the last n_pos prompt positions: [n_prompts, n_pos, n_layers+1, d]."""
    acts = []
    for i in range(0, len(prompts), bs):
        enc = _encode(tok, prompts[i : i + bs], model.device)
        hs = model(**enc, output_hidden_states=True).hidden_states
        assert len(hs) == model.config.num_hidden_layers + 1, "hidden_states convention changed"
        acts.append(torch.stack([h[:, -n_pos:] for h in hs], dim=2).float().cpu())
    return torch.cat(acts)


def mean_diffs(model, tok, harmful, harmless, n_pos=1):
    """Raw difference-in-means vectors: [n_pos, n_layers+1, d] (position -n_pos .. -1)."""
    return token_acts(model, tok, harmful, n_pos).mean(0) - token_acts(model, tok, harmless, n_pos).mean(0)


def refusal_directions(model, tok, harmful, harmless):
    """Unit difference-in-means direction at the last token, every layer: [n_layers+1, d]."""
    diff = mean_diffs(model, tok, harmful, harmless)[-1]
    return diff / diff.norm(dim=-1, keepdim=True)


def refusal_token_ids(tok):
    ids = [tok.encode(w, add_special_tokens=False) for w in REFUSAL_FIRST_WORDS]
    assert all(len(i) == 1 for i in ids), f"refusal words are not single tokens: {ids}"
    return [i[0] for i in ids]


@torch.no_grad()
def next_token_logprobs(model, tok, prompts, bs=32):
    out = []
    for i in range(0, len(prompts), bs):
        enc = _encode(tok, prompts[i : i + bs], model.device)
        out.append(F.log_softmax(model(**enc).logits[:, -1].float(), dim=-1).cpu())
    return torch.cat(out)


def refusal_score(logprobs, refusal_ids):
    """Mean log-odds that the answer starts with a refusal token."""
    p = logprobs[:, refusal_ids].exp().sum(-1).clamp(1e-8, 1 - 1e-8)
    return (p.log() - (1 - p).log()).mean().item()


def kl(base_logprobs, new_logprobs):
    """Mean KL(base || new) of next-token distributions."""
    return (base_logprobs.exp() * (base_logprobs - new_logprobs)).sum(-1).mean().item()


def select_direction(model, tok, harmful, harmless, harmful_val, harmless_val, log=print):
    """Arditi et al. selection. Returns (unit direction, info dict with the candidate table)."""
    n_layers = model.config.num_hidden_layers
    ref_ids = refusal_token_ids(tok)
    diffs = mean_diffs(model, tok, harmful, harmless, N_POST_INSTRUCTION)

    base_harmful = refusal_score(next_token_logprobs(model, tok, harmful_val), ref_ids)
    base_harmless_lp = next_token_logprobs(model, tok, harmless_val)
    base_harmless = refusal_score(base_harmless_lp, ref_ids)
    log(f"  baseline refusal score: harmful {base_harmful:.2f}, harmless {base_harmless:.2f}")

    table = []
    for layer in range(1, int(MAX_LAYER_FRAC * n_layers)):
        for p in range(N_POST_INSTRUCTION):
            v = diffs[p, layer]
            with DirectionAblation(v, 1.0).attach(model):
                bypass = refusal_score(next_token_logprobs(model, tok, harmful_val), ref_ids)
                kl_harmless = kl(base_harmless_lp, next_token_logprobs(model, tok, harmless_val))
            with ActivationAddition(v, layer).attach(model):
                induce = refusal_score(next_token_logprobs(model, tok, harmless_val), ref_ids)
            table.append({"layer": layer, "pos": p - N_POST_INSTRUCTION, "bypass": round(bypass, 3),
                          "induce": round(induce, 3), "kl": round(kl_harmless, 4)})  # fmt: skip

    ok = [c for c in table if c["induce"] > 0 and c["kl"] < KL_MAX]
    if not ok:
        raise RuntimeError("no candidate direction passes the induce/KL filters")
    best = min(ok, key=lambda c: c["bypass"])
    log(f"  {len(ok)}/{len(table)} candidates pass the filters; best: {best}")
    v = diffs[best["pos"] + N_POST_INSTRUCTION, best["layer"]]
    info = {"method": "arditi", "best": best, "baseline_refusal_score":
            {"harmful": round(base_harmful, 3), "harmless": round(base_harmless, 3)},
            "filters": {"kl_max": KL_MAX, "max_layer_frac": MAX_LAYER_FRAC, "induce_min": 0.0},
            "candidates": table}  # fmt: skip
    return v / v.norm(), info
