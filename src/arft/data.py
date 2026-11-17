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
    both within and across sources. For duplicated general prompts, a copy whose
    response is not a refusal-style lecture is preferred over the first copy.
  * Training prompts that share any 13-word n-gram with an evaluation or attacker
    prompt (GSM8K test, HumanEval, WildGuardMix-test, AdvBench, Alpaca) are dropped.
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
import bisect
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
LENGTH_MATCHED = ("refusal", "general")

# WildGuardMix has several responses per prompt, and some "compliance" responses to
# harmless prompts open with a refusal or safety lecture. When a general prompt is
# duplicated, keep a copy whose response does not open like that, if one exists.
# Heuristic on the opening words; not applied to "refusal", where refusing is correct.
PREFER_PLAIN_RESPONSE = ("general",)
LECTURE_OPENING = re.compile(
    r"^\W*(i'?m sorry|i am sorry|i apologi[sz]e|i cannot|i can'?t|i can not"
    r"|i must (emphasi[sz]e|stress|clarify|point out|note)|as an ai|i'?m unable|i am unable"
    r"|i won'?t|i'?m here to help, but|i'?m not able|it'?s important to note that"
    r"|i understand (that|your)|while i (can'?t|cannot))",
    re.I,
)

# Exact Tülu `source` names, checked on 2026-11-16 with Counter(tulu["source"]).
TULU_SOURCES = {
    "coding": "ai2-adapt-dev/personahub_code_v2_34999",
    "math": "ai2-adapt-dev/personahub_math_v5_regen_149960",
}

# Lower edges of the prompt word-count bins used for length matching.
LENGTH_BINS = [0, 6, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 25, 30]

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
    """Lowercase and collapse whitespace: the key for exact-duplicate checks."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def opens_with_lecture(text):
    return bool(LECTURE_OPENING.match(text.strip()))


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


# ---------------------------------------------------------------- decontamination
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
    """Drop duplicate and contaminated rows, then shuffle each pool.

    A duplicate normally loses to the first copy. In PREFER_PLAIN_RESPONSE pools,
    a plain response replaces a kept lecture-style one, in the kept copy's slot,
    so pool sizes and the shuffle are unaffected.

    Each pool gets its own RNG, so its order depends only on its own contents:
    changing one pool never reshuffles another.
    """
    leaked_from = contamination_checker(refs)
    seen = set()  # shared across pools -> also removes cross-source duplicates
    pools, report = {}, {}
    for key in N_TARGET:
        rows = candidates[key]
        stage = {"candidates": len(rows), "duplicates": 0, "plain_copy_preferred": 0,
                 "contaminated": Counter()}
        kept, slot = [], {}  # slot: normalised prompt -> index in kept (this pool only)
        for r in rows:
            norm = normalise(prompt_of(r))
            if norm in seen:
                stage["duplicates"] += 1
                i = slot.get(norm)
                if (key in PREFER_PLAIN_RESPONSE and i is not None
                        and opens_with_lecture(response_of(kept[i]))
                        and not opens_with_lecture(response_of(r))):
                    kept[i] = r
                    stage["plain_copy_preferred"] += 1
            elif source := leaked_from(prompt_of(r)):
                stage["contaminated"][source] += 1
            else:
                slot[norm] = len(kept)
                kept.append(r)
            seen.add(norm)
        random.Random(SEED).shuffle(kept)
        stage["contaminated"] = dict(stage["contaminated"])
        stage["clean"] = len(kept)
        pools[key], report[key] = kept, stage
    return pools, report


# ---------------------------------------------------------------- length matching
def length_bin(r):
    return bisect.bisect_right(LENGTH_BINS, prompt_len(r)) - 1


def length_match(a, b):
    """Keep the same number of rows from a and b in every length bin.

    Both inputs are already shuffled, so taking the first k per bin is a
    random draw. Output order follows the input order.
    """
    bins_a, bins_b = Counter(map(length_bin, a)), Counter(map(length_bin, b))
    quota = bins_a & bins_b  # per-bin minimum; bins missing from either get 0

    def take(rows):
        used, out = Counter(), []
        for r in rows:
            k = length_bin(r)
            if used[k] < quota[k]:
                used[k] += 1
                out.append(r)
        return out

    return take(a), take(b)


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
def build_pools(candidates, refs, match_lengths=True):
    """Clean every pool, optionally length-match refusal/general, cut to N_TARGET."""
    pools, report = clean_pools(candidates, refs)
    length_acc = {"before_matching": length_only_accuracy(*(pools[k] for k in LENGTH_MATCHED))}
    report["length_only_acc_refusal_vs_general"] = length_acc

    if match_lengths:
        matched = length_match(*(pools[k] for k in LENGTH_MATCHED))
        for key, rows in zip(LENGTH_MATCHED, matched):
            pools[key] = rows
            report[key]["length_matched"] = len(rows)

    for key, target in N_TARGET.items():
        n = target
        if match_lengths and key in LENGTH_MATCHED:
            n = min(target, len(pools[key]))  # matching caps the count
        elif len(pools[key]) < target:
            raise ValueError(f"{key}: only {len(pools[key])} clean rows, need {target}")
        pools[key] = pools[key][:n]
        report[key]["selected"] = n

    length_acc["selected"] = length_only_accuracy(*(pools[k] for k in LENGTH_MATCHED))
    return pools, report


def stratified_split(pools):
    """Put the first VAL_FRAC of every (already shuffled) pool in val."""
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


def describe(values):
    deciles = statistics.quantiles(values, n=10)
    return {
        "mean": round(statistics.mean(values), 1),
        "median": statistics.median(values),
        "p10": round(deciles[0], 1),
        "p90": round(deciles[-1], 1),
    }


def stats_for(splits, filtering, match_lengths):
    by_stratum = defaultdict(list)
    for r in splits["train"] + splits["val"]:
        by_stratum[stratum(r)].append(r)
    return {
        "seed": SEED,
        "length_matched": match_lengths,
        "filtering": filtering,
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


def main(out_dir="data", match_lengths=True):
    out = Path(out_dir)
    out.mkdir(exist_ok=True)

    candidates = {**wildguard_candidates(), **tulu_candidates()}
    pools, filtering = build_pools(candidates, reference_prompts(), match_lengths)
    splits = stratified_split(pools)

    for name, rows in splits.items():
        write_jsonl(out / f"{name}.jsonl", rows)
    for filename, rows in samples_by_file(splits["train"]).items():
        write_jsonl(out / filename, rows)

    stats = stats_for(splits, filtering, match_lengths)
    write_json(out / "stats.json", stats, indent=2)
    write_json(out / "eval_indices.json", eval_indices(), indent=1)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--no-length-match", dest="match_lengths", action="store_false",
                    help="keep 2k/2k refusal/general without matching prompt lengths")
    args = ap.parse_args()
    main(args.out_dir, args.match_lengths)
