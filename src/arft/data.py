"""Step 1 - build the frozen train/val splits (handbook §6, Step 1).

Run once (on a compute node, it downloads ~3 GB):
    python -m arft.data            # or: sbatch scripts/slurm/build_data.sh

Outputs (in data/):
    train.jsonl, val.jsonl   full splits (gitignored, rebuilt by this script)
    stats.json               counts per stage/split/source + length statistics
    samples.jsonl            20 train examples per stratum, for reading by eye
    eval_indices.json        frozen random subsets for the evaluation step

Differences from the handbook's starter code, and why (sources: docs/DATA.md,
"Provenance and references"):
  * Tülu sources are matched by exact name. The substring "personahub_math"
    also matches the Algebra subset (paper-only), which would be mixed in silently.
  * Rows whose whole prompt is a refusal sentence ("I'm sorry, I can't assist with
    that.") are dropped: Tülu Persona code contains a few, answered by stub code.
  * Only whole-row duplicates are dropped (same prompt and same response, after
    whitespace/case normalisation). WildGuardMix often has several responses per
    prompt; those rows are all kept.
  * Training prompts that share any 13-word n-gram with an evaluation or attacker
    prompt (GSM8K test, HumanEval, WildGuardMix-test, AdvBench, Alpaca) are dropped.
  * The split is stratified (10% of each stratum goes to val) and grouped: rows
    with the same prompt always land in the same split, so val prompts are unseen.
  * Prompt length is not controlled. stats.json reports how well length alone
    separates refusal from general prompts (length_only_acc_refusal_vs_general).
  * eval_indices.json fixes a seeded random subset of GSM8K and WildGuardMix-test,
    because lm-eval's `limit=N` takes the first N items, not a random sample.
"""

import argparse
import json
import random
import re
import statistics
from collections import Counter, defaultdict
from itertools import groupby
from operator import itemgetter
from pathlib import Path

from datasets import load_dataset

SEED = 0
VAL_FRAC = 0.1
NGRAM = 13
N_SAMPLES_PER_STRATUM = 20

# Target rows per stratum. "refusal" = harmful prompt + refusal (task_type
# "safety"); "general" = harmless WildGuardMix prompt + compliant answer.
# Order matters: pools are cleaned in this order and earlier ones win
# cross-pool duplicates.
N_TARGET = {"refusal": 2000, "general": 2000, "coding": 4000, "math": 4000}

# A prompt that is nothing but a refusal sentence is a data-generation error, not a
# task. Matched against the whole prompt, so ordinary prompts are never caught.
JUNK_PROMPT = re.compile(r"i'?m sorry, i can(?:'?t|not) assist with that(?: request)?\.?", re.I)

# Exact Tülu `source` names, checked on 2026-11-16 with Counter(tulu["source"]).
TULU_SOURCES = {
    "coding": "ai2-adapt-dev/personahub_code_v2_34999",
    "math": "ai2-adapt-dev/personahub_math_v5_regen_149960",
}

# Frozen evaluation subsets (handbook §7, POC column).
EVAL_N = {"gsm8k": 250, "wildguardtest_harmful": 200}


# ---------------------------------------------------------------- rows and text
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


def response_of(r):
    return r["messages"][1]["content"]


def n_words(text):
    return len(text.split())


def prompt_len(r):
    return n_words(prompt_of(r))


def normalise(text):
    """Lowercase and collapse whitespace: the key for duplicate checks."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def is_junk_prompt(text):
    return JUNK_PROMPT.fullmatch(text.strip()) is not None


def ngrams(text, n=NGRAM):
    """Set of n-word tuples, ignoring case and punctuation."""
    words = re.findall(r"\w+", text.lower())
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


# ---------------------------------------------------------------- sources
def wildguard_candidates():
    ds = load_dataset("allenai/wildguardmix", "wildguardtrain", split="train")
    ds = ds.filter(lambda x: x["response"] is not None and not x["adversarial"])
    out = {"refusal": [], "general": []}
    for x in ds:
        labels = (x["prompt_harm_label"], x["response_refusal_label"])
        if labels == ("harmful", "refusal"):
            out["refusal"].append(row(x["prompt"], x["response"], "safety", True, "wildguardmix"))
        elif labels == ("unharmful", "compliance"):
            out["general"].append(row(x["prompt"], x["response"], "general", False, "wildguardmix"))
    return out


def tulu_candidates():
    ds = load_dataset("allenai/tulu-3-sft-mixture", split="train")
    missing = set(TULU_SOURCES.values()) - set(ds.unique("source"))
    if missing:  # fail loudly instead of silently producing zero rows
        raise ValueError(f"Tülu sources not found (renamed upstream?): {missing}")

    def single_turn(x, source):
        roles = [m["role"] for m in x["messages"]]
        return x["source"] == source and roles == ["user", "assistant"]

    out = {}
    for task, source in TULU_SOURCES.items():
        sub = ds.filter(single_turn, fn_kwargs={"source": source})
        out[task] = [row(prompt_of(x), response_of(x), task, False, source) for x in sub]
    return out


def reference_prompts():
    """Evaluation + attacker prompts that must not leak into training."""
    return {
        "gsm8k_test": load_dataset("openai/gsm8k", "main", split="test")["question"],
        "humaneval": load_dataset("openai/openai_humaneval", split="test")["prompt"],
        "wildguardtest": load_dataset("allenai/wildguardmix", "wildguardtest", split="test")["prompt"],
        "advbench": load_dataset("walledai/AdvBench", split="train")["prompt"],
        # Harmless side of the attacker's difference in means. The filter must match
        # directions.attacker_prompts(); the whole set is excluded, not only the slice
        # the attacker currently uses. Most are <13 words, so exact match does the work.
        "alpaca": load_dataset("tatsu-lab/alpaca", split="train")
        .filter(lambda x: x["input"] == "")["instruction"],
    }


# ---------------------------------------------------------------- cleaning
def contamination_checker(refs):
    """Return a function prompt -> name of the leaked reference set, or None.

    "exact" means the normalised prompt equals a reference prompt; otherwise the
    name is the first reference set that shares an n-gram with the prompt.
    """
    exact = {normalise(p) for prompts in refs.values() for p in prompts}
    gram_source = {}
    for name, prompts in refs.items():
        for p in prompts:
            for g in ngrams(p):
                gram_source.setdefault(g, name)

    def leaked_from(prompt):
        if normalise(prompt) in exact:
            return "exact"
        return next((gram_source[g] for g in ngrams(prompt) if g in gram_source), None)

    return leaked_from


def clean_pools(candidates, refs):
    """Drop junk prompts, whole-row duplicates and contaminated rows; shuffle each pool.

    Each pool gets its own RNG, so its order depends only on its own contents:
    changing one pool never reshuffles another.
    """
    leaked_from = contamination_checker(refs)
    seen = set()  # shared across pools -> also removes cross-source duplicates
    pools, report = {}, {}
    for key in N_TARGET:
        rows = candidates[key]
        stage = {"candidates": len(rows), "junk_prompts": 0, "whole_row_duplicates": 0,
                 "contaminated": Counter()}
        kept = []
        for r in rows:
            whole_row = (normalise(prompt_of(r)), normalise(response_of(r)))
            if is_junk_prompt(prompt_of(r)):
                stage["junk_prompts"] += 1
            elif whole_row in seen:
                stage["whole_row_duplicates"] += 1
            elif source := leaked_from(prompt_of(r)):
                stage["contaminated"][source] += 1
            else:
                kept.append(r)
            seen.add(whole_row)
        random.Random(SEED).shuffle(kept)
        stage["contaminated"] = dict(stage["contaminated"])
        stage["clean"] = len(kept)
        pools[key], report[key] = kept, stage
    return pools, report


def length_only_accuracy(pos, neg):
    """Best accuracy of a single prompt-word-count threshold separating pos/neg."""
    labelled = sorted([(prompt_len(r), 1) for r in pos] + [(prompt_len(r), 0) for r in neg])
    total, n_pos = len(labelled), len(pos)
    best, pos_below, neg_below = 0.0, 0, 0
    # Thresholds fall between distinct lengths, so consume equal lengths together.
    for _, group in groupby(labelled, key=itemgetter(0)):
        labels = [y for _, y in group]
        pos_below += sum(labels)
        neg_below += len(labels) - sum(labels)
        # predict "pos" above the threshold, or the reverse - take the better one
        acc = (neg_below + n_pos - pos_below) / total
        best = max(best, acc, 1 - acc)
    return round(best, 3)


# ---------------------------------------------------------------- selection and split
def build_pools(candidates, refs):
    """Clean every pool and cut it to N_TARGET (a random draw: pools are shuffled)."""
    pools, report = clean_pools(candidates, refs)
    length_acc = {"clean_pool": length_only_accuracy(pools["refusal"], pools["general"])}
    for key, target in N_TARGET.items():
        if len(pools[key]) < target:
            raise ValueError(f"{key}: only {len(pools[key])} clean rows, need {target}")
        pools[key] = pools[key][:target]
        report[key]["selected"] = target
    length_acc["selected"] = length_only_accuracy(pools["refusal"], pools["general"])
    report["length_only_acc_refusal_vs_general"] = length_acc
    return pools, report


def stratified_split(pools):
    """Put about VAL_FRAC of every (already shuffled) pool in val, by prompt group.

    Rows that share a normalised prompt go to the same split, so no val prompt is
    seen in training. Groups are taken in pool order; a group that would push val
    past its quota goes to train, so val is at most VAL_FRAC of each stratum.
    """
    train, val = [], []
    for key in sorted(pools):
        rows = pools[key]
        groups = defaultdict(list)  # keeps order of first appearance
        for r in rows:
            groups[normalise(prompt_of(r))].append(r)
        n_val, k_val = round(len(rows) * VAL_FRAC), 0
        for group in groups.values():
            if k_val + len(group) <= n_val:
                val += group
                k_val += len(group)
            else:
                train += group
    assert not prompt_overlap(train, val), "a prompt is in both train and val"
    rng = random.Random(SEED)
    rng.shuffle(train)
    rng.shuffle(val)
    return {"train": train, "val": val}


def prompt_overlap(a, b):
    """Number of normalised prompts that occur in both a and b."""
    return len({normalise(prompt_of(r)) for r in a} & {normalise(prompt_of(r)) for r in b})


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


def describe(values):
    deciles = statistics.quantiles(values, n=10)
    return {
        "mean": round(statistics.mean(values), 1),
        "median": statistics.median(values),
        "p10": round(deciles[0], 1),
        "p90": round(deciles[-1], 1),
    }


def stats_for(splits, filtering):
    by_stratum = defaultdict(list)
    for r in splits["train"] + splits["val"]:
        by_stratum[stratum(r)].append(r)
    return {
        "seed": SEED,
        "filtering": filtering,
        "split_prompt_overlap": prompt_overlap(splits["train"], splits["val"]),
        "splits": {
            name: {
                "total": len(rows),
                "by_stratum": dict(sorted(Counter(map(stratum, rows)).items())),
                "by_source": dict(sorted(Counter(r["source"] for r in rows).items())),
            }
            for name, rows in splits.items()
        },
        "lengths_words": {
            key: {
                "prompt": describe([prompt_len(r) for r in rows]),
                "response": describe([n_words(response_of(r)) for r in rows]),
            }
            for key, rows in sorted(by_stratum.items())
        },
    }


def samples_by_file(train):
    """First N_SAMPLES_PER_STRATUM train rows per stratum, grouped by output file.

    WildGuardMix is gated (AI2 terms) and the repo is public, so its rows go to a
    local, gitignored file; only the openly licensed Tülu samples are committed.
    """
    files = defaultdict(list)
    for key in sorted(N_TARGET):
        rows = [r for r in train if stratum(r) == key][:N_SAMPLES_PER_STRATUM]
        gated = any(r["source"] == "wildguardmix" for r in rows)
        files["samples_wildguardmix.jsonl" if gated else "samples.jsonl"] += rows
    return files


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path, obj, indent):
    path.write_text(json.dumps(obj, indent=indent) + "\n")


def main(out_dir="data"):
    out = Path(out_dir)
    out.mkdir(exist_ok=True)

    candidates = {**wildguard_candidates(), **tulu_candidates()}
    pools, filtering = build_pools(candidates, reference_prompts())
    splits = stratified_split(pools)

    for name, rows in splits.items():
        write_jsonl(out / f"{name}.jsonl", rows)
    for filename, rows in samples_by_file(splits["train"]).items():
        write_jsonl(out / filename, rows)

    stats = stats_for(splits, filtering)
    write_json(out / "stats.json", stats, indent=2)
    write_json(out / "eval_indices.json", eval_indices(), indent=1)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default="data")
    main(ap.parse_args().out_dir)
