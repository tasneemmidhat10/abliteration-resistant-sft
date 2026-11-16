from collections import Counter

from arft.data import (
    length_bin,
    length_match,
    length_only_accuracy,
    ngrams,
    normalise,
    row,
    stratified_split,
)


def mk(n_words, task="general", refusal=False):
    return row(" ".join(f"w{i}" for i in range(n_words)), "r", task, refusal, "s")


def test_length_match_equalises_bins():
    a = [mk(n) for n in [5, 9, 9, 12, 22, 22, 40]]
    b = [mk(n) for n in [5, 9, 12, 12, 22]]
    ma, mb = length_match(a, b)
    assert Counter(map(length_bin, ma)) == Counter(map(length_bin, mb))
    assert len(ma) == 4


def test_length_only_accuracy():
    assert length_only_accuracy([mk(20)] * 5, [mk(10)] * 5) == 1.0
    assert length_only_accuracy([mk(10)] * 5, [mk(10)] * 5) == 0.5


def test_normalise_and_ngrams():
    assert normalise("  Hello\n\tWORLD ") == "hello world"
    assert len(ngrams(" ".join(["x"] * 12))) == 0
    assert len(ngrams(" ".join(f"t{i}" for i in range(14)))) == 2


def test_split_is_stratified():
    pools = {
        "refusal": [mk(10, "safety", True) for _ in range(50)],
        "coding": [mk(10, "coding") for _ in range(100)],
    }
    splits = stratified_split(pools)
    val = Counter(r["task_type"] for r in splits["val"])
    assert val == {"safety": 5, "coding": 10}
    assert len(splits["train"]) == 135
