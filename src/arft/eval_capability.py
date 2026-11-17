"""Capability evaluation (handbook Step 5) through lm-eval's Python API,
so that ablation hooks attached to `model` stay active.

Pinned: lm-eval 0.4.13 (task names and metric keys differ between versions).
"""

import json
import os

from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM

from arft.eval_refusal import EVAL_INDICES

# HumanEval executes model-written code (in a subprocess with a timeout).
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

METRIC = {"gsm8k": "exact_match,flexible-extract", "humaneval_instruct": "pass@1,create_test"}


def task_samples(task, smoke=False):
    """Frozen random GSM8K subset instead of limit= (which takes the first N items)."""
    if task == "gsm8k":
        idx = json.loads(EVAL_INDICES.read_text())["gsm8k"]
    else:
        idx = list(range(164))  # all HumanEval problems
    return idx[:5] if smoke else idx


def per_item(samples, metric):
    """{doc_id: score} for one metric, e.g. 'exact_match,flexible-extract'."""
    name, filt = metric.split(",")
    return {s["doc_id"]: float(s[name]) for s in samples if s.get("filter") == filt}


def dump_outputs(samples, metric, path):
    """Raw model answers + extracted answer + score per item, for reading failures."""
    name, filt = metric.split(",")
    with open(path, "w") as f:
        for s in samples:
            if s.get("filter") != filt:
                continue
            raw = s["resps"][0][0] if s["resps"] and s["resps"][0] else ""
            f.write(json.dumps({"doc_id": s["doc_id"], "score": float(s[name]),
                                "extracted": s["filtered_resps"], "raw": raw}) + "\n")  # fmt: skip


def capability_scores(model, tok, smoke=False, batch_size=32, dump_prefix=None):
    """Returns {task: accuracy, task + '_items': {doc_id: 0/1}} (items for paired tests).

    With dump_prefix, raw answers go to f"{dump_prefix}_{task}.jsonl".
    """
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=batch_size)
    scores = {}
    for task, metric in METRIC.items():
        res = simple_evaluate(
            model=lm,
            tasks=[task],
            samples={task: task_samples(task, smoke)},
            apply_chat_template=True,
            fewshot_as_multiturn=(task == "gsm8k"),
            confirm_run_unsafe_code=True,
            log_samples=True,
        )
        task_res = res["results"][task]
        if metric not in task_res:
            raise KeyError(f"{task}: metric {metric!r} not found, available: {list(task_res)}")
        scores[task] = float(task_res[metric])
        items = per_item(res["samples"][task], metric)
        expected = len(task_samples(task, smoke))
        if len(items) != expected:
            raise ValueError(f"{task}: {len(items)} per-item scores, expected {expected}")
        scores[task + "_items"] = items
        if dump_prefix:
            dump_outputs(res["samples"][task], metric, f"{dump_prefix}_{task}.jsonl")
    return scores
