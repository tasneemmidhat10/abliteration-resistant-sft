from collections import Counter
from itertools import count

from arft.data import (
    clean_pools,
    is_junk_prompt,
    length_only_accuracy,
    ngrams,
    normalise,
    prompt_of,
    response_of,
    row,
    stratified_split,
)

_uid = count()


def mk(n_words, task="general", refusal=False, prompt=None):
    """A row with an n-word prompt; prompts are unique unless `prompt` is given."""
    if prompt is None:
        prompt = " ".join([f"p{next(_uid)}"] + [f"w{i}" for i in range(1, n_words)])
    return row(prompt, "r", task, refusal, "s")


def pools_of(**rows):
    return {"refusal": [], "general": [], "coding": [], "math": [], **rows}


def test_length_only_accuracy():
    assert length_only_accuracy([mk(20)] * 5, [mk(10)] * 5) == 1.0
    assert length_only_accuracy([mk(10)] * 5, [mk(10)] * 5) == 0.5


def test_normalise_and_ngrams():
    assert normalise("  Hello\n\tWORLD ") == "hello world"
    assert len(ngrams(" ".join(["x"] * 12))) == 0
    assert len(ngrams(" ".join(f"t{i}" for i in range(14)))) == 2


def test_junk_prompt():
    for junk in ["I'm sorry, I can't assist with that.", "I'm sorry, I cannot assist with that request.",
                 "  i'm sorry, i can't assist with that request. "]:
        assert is_junk_prompt(junk)
    for ok in ["As an aircraft mechanic, compute the fuel load for a 3-hour flight.",
               "I'm sorry, I can't assist with that. Write a function that reverses a list.",
               "Write a python function to sum a list of integers."]:
        assert not is_junk_prompt(ok)


def test_dedup_drops_only_whole_rows():
    candidates = pools_of(general=[
        row("Nail a talk", "Answer A", "general", False, "s"),
        row("nail a  TALK", "Answer B", "general", False, "s"),   # same prompt, other response: kept
        row("Nail a talk", "answer  a", "general", False, "s"),   # whole-row duplicate: dropped
    ], coding=[row("I'm sorry, I can't assist with that.", "def soln(): ...", "coding", False, "s")])
    pools, report = clean_pools(candidates, refs={})
    assert sorted(response_of(r) for r in pools["general"]) == ["Answer A", "Answer B"]
    assert report["general"]["whole_row_duplicates"] == 1
    assert pools["coding"] == [] and report["coding"]["junk_prompts"] == 1


def test_split_is_stratified():
    pools = {
        "refusal": [mk(10, "safety", True) for _ in range(50)],
        "coding": [mk(10, "coding") for _ in range(100)],
    }
    splits = stratified_split(pools)
    val = Counter(r["task_type"] for r in splits["val"])
    assert val == {"safety": 5, "coding": 10}
    assert len(splits["train"]) == 135


def test_split_keeps_prompt_groups_together():
    # 20 general rows: 3 copies each of two prompts, the rest unique; quota is 2 val rows
    general = [mk(5, prompt="dup a") for _ in range(3)] + [mk(5) for _ in range(14)]
    general += [mk(5, prompt="dup b") for _ in range(3)]
    splits = stratified_split({"general": general})
    train_prompts = {normalise(prompt_of(r)) for r in splits["train"]}
    val_prompts = {normalise(prompt_of(r)) for r in splits["val"]}
    assert not train_prompts & val_prompts
    assert len(splits["val"]) == 2 and len(splits["train"]) == 18
