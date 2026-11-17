"""Attack one model and evaluate it at every strength (handbook Steps 2 and 4).

    python -m arft.sweep --name base --model Qwen/Qwen2.5-1.5B-Instruct
    python -m arft.sweep --name smoke --model Qwen/Qwen2.5-0.5B-Instruct --smoke

The attacker re-estimates the refusal direction on the model being attacked,
with its own data (AdvBench / Alpaca), then sweeps alpha. Direction selection:
  --select arditi (default)  Arditi et al. 2024: layer x post-instruction position
                             candidates, filtered by induce > 0 and harmless KL < 0.1
  --select crude             handbook POC: last-token direction of the layer with the
                             lowest generated refusal rate at alpha=1 (damages capability)

Resumable: rerunning the same command skips finished rows (the chosen direction
is saved in results/directions/<name>.pt). With --budget-min it stops cleanly
before a row that would overrun, e.g. on the 1-hour A40devel partition:
    sbatch -p A40devel -t 1:00:00 scripts/slurm/sweep.sh base <model> --budget-min 50

Writes results/metrics/<name>.json (committed) and the refusal-eval generations
to results/generations/<name>/ (gitignored: they contain harmful completions).
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from arft.directions import attacker_prompts, refusal_directions, select_direction
from arft.eval_capability import capability_scores
from arft.eval_refusal import load_harmful_test, refusal_eval, refusal_rate
from arft.interventions import DirectionAblation

# Ablation at all 29 hook points compounds, so most of the change happens below 0.25.
ALPHAS = [0.0, 0.02, 0.05, 0.1, 0.15, 0.25, 0.5, 1.0]
T0 = time.time()
ROOT = Path(__file__).resolve().parents[2]


def git_commit():
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
        dirty = subprocess.call(["git", "diff", "--quiet", "HEAD"], cwd=ROOT) != 0
        return out + ("-dirty" if dirty else "")
    except FileNotFoundError:  # Bender compute nodes have no git binary: read .git directly
        head = (ROOT / ".git" / "HEAD").read_text().strip()
        if not head.startswith("ref: "):
            return head + " (dirty state unknown)"
        ref = head[5:]
        ref_file = ROOT / ".git" / ref
        if ref_file.exists():
            sha = ref_file.read_text().strip()
        else:
            packed = (ROOT / ".git" / "packed-refs").read_text().splitlines()
            sha = next(line.split()[0] for line in packed if line.endswith(" " + ref))
        return sha + " (dirty state unknown)"


def select_layer(model, tok, directions, select_prompts, layers):
    """Attacker's layer choice: lowest refusal at alpha=1 on held-out AdvBench."""
    table = []
    for layer in layers:
        with DirectionAblation(directions[layer], 1.0).attach(model):
            rr = refusal_rate(model, tok, select_prompts)
        table.append({"layer": layer, "refusal_at_alpha1": rr})
        print(f"  layer {layer:>2}: refusal at alpha=1 = {rr:.3f}", flush=True)
    best = min(table, key=lambda t: t["refusal_at_alpha1"])  # ties -> earliest layer
    return best["layer"], table


def evaluate(model, tok, test_prompts, gen_dir, tag, smoke):
    t0 = time.time()
    refusal, gens = refusal_eval(model, tok, test_prompts)
    (gen_dir / f"{tag}.jsonl").write_text("".join(json.dumps(g) + "\n" for g in gens))
    caps = capability_scores(model, tok, smoke=smoke, dump_prefix=str(gen_dir / tag))
    items = {k: caps.pop(k) for k in list(caps) if k.endswith("_items")}
    items["refusal_items"] = {i: float(g["refused"]) for i, g in enumerate(gens)}
    return {"refusal": refusal, **caps, "seconds": round(time.time() - t0), "items": items}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="results/metrics/<name>.json")
    ap.add_argument("--model", required=True, help="HF id or local path")
    ap.add_argument("--smoke", action="store_true", help="tiny sizes, to test the pipeline")
    ap.add_argument("--budget-min", type=float, default=0,
                    help="stop before a row that would exceed this many minutes (0 = no limit)")
    ap.add_argument("--select", choices=["arditi", "crude"], default="arditi")
    args = ap.parse_args()

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model = model.cuda().eval()
    n_layers = model.config.num_hidden_layers

    harmful, harmless, harmful_val, harmless_val = attacker_prompts()
    test_prompts = load_harmful_test()
    layers = list(range(n_layers // 4, 3 * n_layers // 4, 2))
    if args.smoke:
        harmful, harmless = harmful[:32], harmless[:32]
        harmful_val, harmless_val = harmful_val[:16], harmless_val[:16]
        test_prompts, layers = test_prompts[:16], layers[:2]

    gen_dir = ROOT / "results" / "generations" / args.name
    gen_dir.mkdir(parents=True, exist_ok=True)
    out_path = ROOT / "results" / "metrics" / f"{args.name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dir_path = ROOT / "results" / "directions" / f"{args.name}.pt"
    dir_path.parent.mkdir(parents=True, exist_ok=True)

    def save():
        out_path.write_text(json.dumps(result, indent=2) + "\n")

    if out_path.exists() and dir_path.exists():  # resume an interrupted sweep
        result = json.loads(out_path.read_text())
        assert result["model_path"] == args.model and result["smoke"] == args.smoke
        r, best_layer = torch.load(dir_path), result["attack_layer"]
        print(f"== resuming: layer {best_layer}, {len(result['rows'])} rows done", flush=True)
    else:
        print(f"== selecting refusal direction ({args.select})", flush=True)
        if args.select == "arditi":
            r, selection = select_direction(
                model, tok, harmful, harmless, harmful_val, harmless_val,
                log=lambda m: print(m, flush=True))  # fmt: skip
            best_layer = selection["best"]["layer"]
        else:
            directions = refusal_directions(model, tok, harmful, harmless)
            best_layer, layer_table = select_layer(model, tok, directions, harmful_val, layers)
            r, selection = directions[best_layer], {"method": "crude", "candidates": layer_table}
        torch.save(r, dir_path)
        print(f"== attack layer {best_layer}", flush=True)
        result = {
            "model": args.name,
            "model_path": args.model,
            "git_commit": git_commit(),
            "smoke": args.smoke,
            "attack_layer": best_layer,
            "selection": selection,
            "n_attacker_prompts": {"harmful": len(harmful), "harmless": len(harmless),
                                   "harmful_val": len(harmful_val), "harmless_val": len(harmless_val)},
            "n_refusal_test": len(test_prompts),
            "rows": [],
        }
        save()

    # Negative control: ablate a random unit direction at full strength. If capability
    # drops as much as with r, the drop is generic, not specific to refusal.
    rand = torch.randn(r.shape, generator=torch.Generator().manual_seed(0))
    plan = [("refusal", a, r) for a in ALPHAS] + [("random", 1.0, rand)]
    done = {(row["direction"], row["alpha"]) for row in result["rows"]}

    for kind, alpha, direction in plan:
        if (kind, alpha) in done:
            continue
        elapsed = (time.time() - T0) / 60
        last = max((row["seconds"] for row in result["rows"]), default=0) / 60
        if args.budget_min and elapsed + 1.2 * last > args.budget_min:
            print(f"== time budget reached after {elapsed:.0f} min; resubmit to continue", flush=True)
            return
        with DirectionAblation(direction, alpha).attach(model):
            row = {"alpha": alpha, "direction": kind, "layer": best_layer,
                   **evaluate(model, tok, test_prompts, gen_dir, f"{kind}_a{alpha}", args.smoke)}  # fmt: skip
        print({k: v for k, v in row.items() if k != "items"}, flush=True)
        result["rows"].append(row)
        save()
    print(f"== all rows done: {out_path}")


if __name__ == "__main__":
    main()
