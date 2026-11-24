"""Collect the newest lm-eval results of each model into one table (+ results/lm-eval/summary.json).
Run from the repo root:  python evals/summarize_results.py
"""
import glob
import json
import os
from transformers import AutoTokenizer

MODELS = {  # table name -> folder under results/lm-eval/ (made by evals/run_lm_eval.sh)
    'base (no SFT)': 'qwen3-0.6b-base-prepared',
    'own loop, epoch 1': 'own-loop_epoch-1',
    'own loop, epoch 3': 'own-loop_epoch-3',
    'TRL fp32, epoch 1': 'foundry-fp32_checkpoint-169',
    'TRL fp32, epoch 3': 'foundry-fp32_final',
    'TRL pure bf16, epoch 3': 'foundry-bf16_final',
}
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B-Base')


def newest(pattern):
    return sorted(glob.glob(pattern))[-1]          # file names contain a timestamp, so the last one is the newest


rows = {}
for name, folder in MODELS.items():
    base = f'results/lm-eval/{folder}/*/'
    res = json.load(open(newest(base + 'results_*.json')))['results']
    # A reply as long as the 1280-token limit means the model never produced <|im_end|>.
    ifeval_samples = [json.loads(line) for line in open(newest(base + 'samples_ifeval_*.jsonl'))]
    lengths = [len(tokenizer(s['resps'][0][0])['input_ids']) for s in ifeval_samples]
    rows[name] = {
        'ifeval_prompt_strict': res['ifeval']['prompt_level_strict_acc,none'],
        'ifeval_inst_strict': res['ifeval']['inst_level_strict_acc,none'],
        'ifeval_prompt_loose': res['ifeval']['prompt_level_loose_acc,none'],
        'humaneval_pass@1': res['humaneval_instruct']['pass@1,create_test'],
        'ifeval_no_stop': sum(n >= 1270 for n in lengths) / len(lengths),
    }

print('| model | IFEval prompt strict | IFEval instr. strict | IFEval prompt loose | HumanEval pass@1 | IFEval replies without stop |')
print('|---|---|---|---|---|---|')
for name, r in rows.items():
    print(f"| {name} | {r['ifeval_prompt_strict']:.3f} | {r['ifeval_inst_strict']:.3f} | {r['ifeval_prompt_loose']:.3f} "
          f"| {r['humaneval_pass@1']:.3f} | {r['ifeval_no_stop']:.1%} |")
with open('results/lm-eval/summary.json', 'w') as f:
    json.dump(rows, f, indent=2)
