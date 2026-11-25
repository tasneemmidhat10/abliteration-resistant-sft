"""Collect the newest lm-eval results of each model into results/lm-eval/summary.md (+ summary.json).
Run from the repo root:  python evals/summarize_results.py
"""
import glob
import json
import os
from datetime import date
from transformers import AutoTokenizer

MODELS = {  # table name -> folder under results/lm-eval/ (made by evals/run_lm_eval.sh)
    'Base model (no SFT)': 'qwen3-0.6b-base-prepared',
    'Own loop, epoch 1': 'own-loop_epoch-1',
    'Own loop, epoch 3': 'own-loop_epoch-3',
    'llm-foundry fp32, epoch 1': 'foundry-fp32_checkpoint-169',
    'llm-foundry fp32, epoch 3': 'foundry-fp32_final',
    'llm-foundry pure bf16, epoch 3': 'foundry-bf16_final',
}
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B-Base')


def newest(pattern):
    return sorted(glob.glob(pattern))[-1]          # file names contain a timestamp, so the last one is the newest


def pct(value, stderr=None):
    """0.3752, 0.0208 -> '37.5 ± 2.1'"""
    return f'{100 * value:.1f}' + (f' ± {100 * stderr:.1f}' if stderr is not None else '')


rows, sources = {}, {}
for name, folder in MODELS.items():
    base = f'results/lm-eval/{folder}/*/'
    results_file = newest(base + 'results_*.json')
    res = json.load(open(results_file))['results']
    ifeval, humaneval = res['ifeval'], res['humaneval_instruct']
    # A reply as long as the 1280-token limit means the model never produced <|im_end|>.
    ifeval_samples = [json.loads(line) for line in open(newest(base + 'samples_ifeval_*.jsonl'))]
    lengths = [len(tokenizer(s['resps'][0][0])['input_ids']) for s in ifeval_samples]
    rows[name] = {
        'ifeval_prompt_strict': ifeval['prompt_level_strict_acc,none'],
        'ifeval_prompt_strict_stderr': ifeval['prompt_level_strict_acc_stderr,none'],
        'ifeval_inst_strict': ifeval['inst_level_strict_acc,none'],
        'ifeval_prompt_loose': ifeval['prompt_level_loose_acc,none'],
        'humaneval_pass@1': humaneval['pass@1,create_test'],
        'humaneval_pass@1_stderr': humaneval['pass@1_stderr,create_test'],
        'ifeval_no_stop': sum(n >= 1270 for n in lengths) / len(lengths),
    }
    sources[name] = os.path.relpath(results_file, 'results/lm-eval')

table = ['| Model | IFEval: all instructions followed | IFEval: single instructions followed '
         '| HumanEval: problems solved | IFEval replies that never end |',
         '|---|---|---|---|---|']
for name, r in rows.items():
    table.append(f"| {name} | {pct(r['ifeval_prompt_strict'], r['ifeval_prompt_strict_stderr'])} "
                 f"| {pct(r['ifeval_inst_strict'])} | {pct(r['humaneval_pass@1'], r['humaneval_pass@1_stderr'])} "
                 f"| {pct(r['ifeval_no_stop'])} |")

summary = f"""# Benchmark results: Qwen3-0.6B-Base after SFT

Generated on {date.today()} by `evals/summarize_results.py` from the newest lm-eval run of each model.
All numbers are percentages; ± is one standard error.

{chr(10).join(table)}

## What the columns mean

- **IFEval: all instructions followed** – share of the 541 IFEval prompts where the reply met every instruction
  (for example "no commas" *and* "at least 300 words"), checked strictly by code. The usual headline IFEval number.
- **IFEval: single instructions followed** – share of the individual instructions met, counted one by one.
  Always higher than the column before, because one missed instruction fails a whole prompt.
- **HumanEval: problems solved** – share of the 164 Python problems whose one answer passes all hidden unit tests
  (pass@1, greedy decoding).
- **IFEval replies that never end** – replies that run into the 1,280-token limit without the end-of-turn token
  `<|im_end|>`. A high value means the model has not learned when to stop.

## How to read differences

With one standard error of about 2 points on IFEval and about 4 points on HumanEval, differences smaller than
roughly twice that (4 and 8 points) can be noise. Each configuration was trained once (one seed).

## How the numbers were produced

- lm-evaluation-harness, `hf` backend, model in bf16, 0-shot, greedy decoding, batch size 32
  ([evals/run_lm_eval.sh](../../evals/run_lm_eval.sh)).
- Every prompt is wrapped in the model's chat template, which ends with the empty think block
  `<think>\\n\\n</think>\\n\\n`, the same format the models were trained on.
- IFEval stops at `<|im_end|>` or 1,280 new tokens; HumanEval at `<|im_end|>`, a new top-level
  `def`/`class`/`if`/`print`/comment, or 1,024 new tokens.
- Every prompt and reply is saved next to the scores (`samples_*.jsonl`), so any number can be checked by reading
  the outputs.

## Source files

{chr(10).join(f'- {name}: `{path}`' for name, path in sources.items())}
"""

with open('results/lm-eval/summary.md', 'w') as f:
    f.write(summary)
with open('results/lm-eval/summary.json', 'w') as f:
    json.dump(rows, f, indent=2)
print(summary)
