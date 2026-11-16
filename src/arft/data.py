"""Step 1 - build the frozen train/val splits (handbook §6, Step 1).

Run once (on a compute node, it downloads ~3 GB):
    python -m arft.data            # or: sbatch scripts/slurm/build_data.sh

Outputs (in data/):
    train.jsonl, val.jsonl   full splits (gitignored, rebuilt by this script)
    stats.json               counts per stage/split/source + length statistics
    samples.jsonl            20 train examples per stratum, for reading by eye
    eval_indices.json        frozen random subsets for the evaluation step

Differences from the handbook's starter code, and why:
  * Tülu sources are matched by exact name. The substring "personahub_math"
    also matches the Algebra subset (paper-only), which would be mixed in silently.
  * Exact-duplicate prompts are dropped (after whitespace/case normalisation),
    both within and across sources.
  * Training prompts that share any 13-word n-gram with an evaluation or attacker
    prompt (GSM8K test, HumanEval, WildGuardMix-test, AdvBench) are dropped.
  * Refusal and general prompts are length-matched (same count per word-length
    bin), because in WildGuardMix harmful prompts are ~2x longer than harmless
    ones and the policy head could otherwise learn length instead of harm.
    This caps both strata at ~1.4k instead of 2k. Disable with --no-length-match.
    The Persona (coding/math) vs WildGuardMix style gap cannot be matched.
  * The val split is stratified by stratum (refusal / general / coding / math).
  * eval_indices.json fixes a seeded random subset of GSM8K and WildGuardMix-test,
    because lm-eval's `limit=N` takes the first N items, not a random sample.
"""

import argparse
import json
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset

SEED = 0
VAL_FRAC = 0.1
NGRAM = 13
N_SAMPLES_PER_STRATUM = 20

# Target counts per stratum. "refusal" = harmful prompt + refusal (task_type
# "safety"); "general" = harmless WildGuardMix prompt + compliant answer.
N_TARGET = {"refusal": 2000, "general": 2000, "coding": 4000, "math": 4000}

# Exact Tülu `source` names, checked on 2026-11-16 with Counter(tulu["source"]).
TULU_SOURCES = {
    "coding": "ai2-adapt-dev/personahub_code_v2_34999",
    "math": "ai2-adapt-dev/personahub_math_v5_regen_149960",
}

# Prompt word-count bins used to length-match refusal vs general.
LENGTH_BINS = [0, 6, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 25, 30, 10**6]

# Frozen evaluation subsets (handbook §7, POC column).
EVAL_N = {"gsm8k": 250, "wildguardtest_harmful": 200}


# ---------------------------------------------------------------- helpers
def row(prompt, response, task_type, is_refusal, source):
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
        "task_type": task_type,
        "is_refusal_example": is_refusal,
        "source": source,
    }


def stratum(r):
    return "refusal" if r["is_refusal_example"] else r["task_type"]


def prompt_of(r):
    return r["messages"][0]["content"]


def normalise(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


def ngrams(text, n=NGRAM):
    words = re.findall(r"\w+", text.lower())
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def n_words(text):
    return len(text.split())


def describe(values):
    q = statistics.quantiles(values, n=10)
    return {
        "mean": round(statistics.mean(values), 1),
        "median": statistics.median(values),
        "p10": round(q[0], 1),
        "p90": round(q[-1], 1),
    }


# ---------------------------------------------------------------- sources
def wildguard_candidates():
    ds = load_dataset("allenai/wildguardmix", "wildguardtrain", split="train")
    ds = ds.filter(lambda x: x["response"] is not None and not x["adversarial"])
    out = {"refusal": [], "general": []}
    for x in ds:
        if x["prompt_harm_label"] == "harmful" and x["response_refusal_label"] == "refusal":
            out["refusal"].append(row(x["prompt"], x["response"], "safety", True, "wildguardmix"))
        elif x["prompt_harm_label"] == "unharmful" and x["response_refusal_label"] == "compliance":
            out["general"].append(row(x["prompt"], x["response"], "general", False, "wildguardmix"))
    return out


def tulu_candidates():
    ds = load_dataset("allenai/tulu-3-sft-mixture", split="train")
    present = set(ds.unique("source"))
    missing = set(TULU_SOURCES.values()) - present
    if missing:  # fail loudly instead of silently producing zero rows
        raise ValueError(f"Tülu sources not found (renamed upstream?): {missing}")
    out = {}
    for task, source in TULU_SOURCES.items():
        sub = ds.filter(
            lambda x: x["source"] == source
            and len(x["messages"]) == 2
            and x["messages"][0]["role"] == "user"
            and x["messages"][1]["role"] == "assistant"
        )
        out[task] = [
            row(x["messages"][0]["content"], x["messages"][1]["content"], task, False, source)
            for x in sub
        ]
    return out


def reference_prompts():
    """Evaluation + attacker prompts that must not leak into training."""
    refs = {
        "gsm8k_test": load_dataset("openai/gsm8k", "main", split="test")["question"],
        "humaneval": load_dataset("openai/openai_humaneval", split="test")["prompt"],
        "wildguardtest": [
            p for p in load_dataset("allenai/wildguardmix", "wildguardtest", split="test")["prompt"]
        ],
        "advbench": load_dataset("walledai/AdvBench", split="train")["prompt"],
    }
    return refs


# ---------------------------------------------------------------- filtering
def length_bin(r):
    n = n_words(prompt_of(r))
    return next(i for i, hi in enumerate(LENGTH_BINS[1:]) if n < hi)


def length_match(a, b):
    """Keep the same number of rows from a and b in every length bin.

    Both inputs are already shuffled, so taking the first k per bin is a
    random draw. Output order follows the input order.
    """
    bins_a, bins_b = Counter(map(length_bin, a)), Counter(map(length_bin, b))
    quota = {k: min(bins_a[k], bins_b[k]) for k in bins_a}

    def take(rows):
        used, out = Counter(), []
        for r in rows:
            k = length_bin(r)
            if used[k] < quota.get(k, 0):
                used[k] += 1
                out.append(r)
        return out

    return take(a), take(b)


def length_only_accuracy(pos, neg):
    """Best accuracy of a single prompt-word-count threshold separating pos/neg."""
    pts = sorted([(n_words(prompt_of(r)), 1) for r in pos] + [(n_words(prompt_of(r)), 0) for r in neg])
    n_pos, best = len(pos), 0.0
    pos_below = neg_below = 0
    for i, (n, y) in enumerate(pts):
        pos_below += y
        neg_below += 1 - y
        if i + 1 < len(pts) and pts[i + 1][0] == n:
            continue
        # predict "pos" above the threshold, or the reverse - take the better one
        acc = (neg_below + (n_pos - pos_below)) / len(pts)
        best = max(best, acc, 1 - acc)
    return round(best, 3)


def build_pools(candidates, refs, match_lengths=True):
    """Dedup + decontaminate every candidate pool, then subsample to N_TARGET."""
    ref_exact = {normalise(p) for ps in refs.values() for p in ps}
    ref_grams = {}
    for name, ps in refs.items():
        for p in ps:
            for g in ngrams(p):
                ref_grams.setdefault(g, name)

    seen = set()  # shared across pools -> also removes cross-source duplicates
    report, pools = {}, {}
    for key in ["refusal", "general", "coding", "math"]:  # fixed order = deterministic
        rows = candidates[key]
        stage = {"candidates": len(rows), "duplicates": 0, "contaminated": Counter()}
        kept = []
        for r in rows:
            p = prompt_of(r)
            norm = normalise(p)
            if norm in seen:
                stage["duplicates"] += 1
                continue
            seen.add(norm)
            if norm in ref_exact:
                stage["contaminated"]["exact"] += 1
                continue
            hit = next((ref_grams[g] for g in ngrams(p) if g in ref_grams), None)
            if hit:
                stage["contaminated"][hit] += 1
                continue
            kept.append(r)
        random.Random(SEED).shuffle(kept)
        stage["clean"] = len(kept)
        stage["contaminated"] = dict(stage["contaminated"])
        pools[key], report[key] = kept, stage

    report["length_only_acc_refusal_vs_general"] = {
        "before_matching": length_only_accuracy(pools["refusal"], pools["general"])
    }
    if match_lengths:
        pools["refusal"], pools["general"] = length_match(pools["refusal"], pools["general"])
        for key in ("refusal", "general"):
            report[key]["length_matched"] = len(pools[key])

    for key in pools:
        if key in ("refusal", "general") and match_lengths:
            n = min(N_TARGET[key], len(pools[key]))  # matching caps the count
        else:
            n = N_TARGET[key]
            if len(pools[key]) < n:
                raise ValueError(f"{key}: only {len(pools[key])} clean rows, need {n}")
        pools[key] = pools[key][:n]
        report[key]["selected"] = n
    report["_matched"] = match_lengths
    report["length_only_acc_refusal_vs_general"]["selected"] = length_only_accuracy(
        pools["refusal"], pools["general"]
    )
    return pools, report


def stratified_split(pools):
    train, val = [], []
    for key in sorted(pools):
        rows = pools[key]
        n_val = round(len(rows) * VAL_FRAC)
        val += rows[:n_val]
        train += rows[n_val:]
    rng = random.Random(SEED)
    rng.shuffle(train)
    rng.shuffle(val)
    return {"train": train, "val": val}


# ---------------------------------------------------------------- outputs
def eval_indices():
    rng = random.Random(SEED)
    gsm8k_n = len(load_dataset("openai/gsm8k", "main", split="test"))
    wgt = load_dataset("allenai/wildguardmix", "wildguardtest", split="test")
    harmful = [
        i for i, x in enumerate(wgt) if x["prompt_harm_label"] == "harmful" and not x["adversarial"]
    ]
    return {
        "_note": "Row indices into the HF test splits, seed 0. GSM8K: pass to "
        "lm_eval.simple_evaluate(samples={'gsm8k': ...}) instead of limit=. "
        "HumanEval: use all 164 problems.",
        "gsm8k": sorted(rng.sample(range(gsm8k_n), EVAL_N["gsm8k"])),
        "wildguardtest_harmful": sorted(rng.sample(harmful, EVAL_N["wildguardtest_harmful"])),
    }


def stats_for(splits, filtering):
    stats = {"seed": SEED, "length_matched": filtering.pop("_matched"), "filtering": filtering, "splits": {}, "lengths_words": {}}
    for name, rows in splits.items():
        stats["splits"][name] = {
            "total": len(rows),
            "by_stratum": dict(sorted(Counter(stratum(r) for r in rows).items())),
            "by_source": dict(sorted(Counter(r["source"] for r in rows).items())),
        }
    by = defaultdict(list)
    for r in splits["train"] + splits["val"]:
        by[stratum(r)].append(r)
    for key, rows in sorted(by.items()):
        stats["lengths_words"][key] = {
            "prompt": describe([n_words(prompt_of(r)) for r in rows]),
            "response": describe([n_words(r["messages"][1]["content"]) for r in rows]),
        }
    return stats


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(out_dir="data", match_lengths=True):
    out = Path(out_dir)
    out.mkdir(exist_ok=True)

    candidates = {**wildguard_candidates(), **tulu_candidates()}
    pools, filtering = build_pools(candidates, reference_prompts(), match_lengths)
    splits = stratified_split(pools)

    for name, rows in splits.items():
        write_jsonl(out / f"{name}.jsonl", rows)

    # WildGuardMix is gated (AI2 terms), and the repo is public: its rows go to a
    # local, gitignored file; only the openly licensed Tülu samples are committed.
    samples = defaultdict(list)
    for key in sorted(pools):
        rows = [r for r in splits["train"] if stratum(r) == key][:N_SAMPLES_PER_STRATUM]
        gated = any(r["source"] == "wildguardmix" for r in rows)
        samples["samples_wildguardmix.jsonl" if gated else "samples.jsonl"] += rows
    for name, rows in samples.items():
        write_jsonl(out / name, rows)

    stats = stats_for(splits, filtering)
    (out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    (out / "eval_indices.json").write_text(json.dumps(eval_indices(), indent=1) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--no-length-match", dest="match_lengths", action="store_false",
                    help="keep 2k/2k refusal/general without matching prompt lengths")
    args = ap.parse_args()
    main(args.out_dir, args.match_lengths)
