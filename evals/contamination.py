"""Does our SFT data overlap with the benchmarks we evaluate on (IFEval, HumanEval)?
Run from the repo root:  python evals/contamination.py      Output: printed report + results/contamination.json

Two checks over all 3,000 source rows (prompt and answer):
  1. 13-gram overlap: any 13 consecutive lowercased words shared with a benchmark item.
  2. HumanEval function names: `def <entry_point>(` from any of the 164 problems.
"""
import json
import re
from collections import defaultdict
from datasets import load_dataset

N = 13
SOURCES = ['data/tulu_coding.jsonl', 'data/tulu_instruction-following.jsonl', 'data/wildguard.jsonl']


def ngrams(text):
    words = re.findall(r'\w+', text.lower())          # lowercased words, punctuation and spacing ignored
    return {' '.join(words[i:i + N]) for i in range(len(words) - N + 1)}


ifeval = load_dataset('google/IFEval', split='train')
humaneval = load_dataset('openai/openai_humaneval', split='test')
benchmarks = {                                        # benchmark -> {13-gram: id of the item it comes from}
    'ifeval': {g: row['key'] for row in ifeval for g in ngrams(row['prompt'])},
    'humaneval': {g: row['task_id'] for row in humaneval
                  for g in ngrams(row['prompt'] + '\n' + row['canonical_solution'])},
}
entry_points = {row['entry_point']: row['task_id'] for row in humaneval}

rows = [json.loads(line) for f in SOURCES for line in open(f)]
report = {'ngram_hits': defaultdict(list), 'function_name_hits': []}
for row in rows:
    for part, text in (('prompt', row['messages'][0]['content']), ('answer', row['messages'][1]['content'])):
        grams = ngrams(text)
        for bench, index in benchmarks.items():
            shared = grams & index.keys()
            if shared:
                example = sorted(shared)[0]
                report['ngram_hits'][bench].append({'id': row['id'], 'subset': row['task_type'], 'part': part,
                                                    'benchmark_item': index[example], 'shared_ngram': example})
        for name, task_id in entry_points.items():
            if re.search(rf'\bdef {re.escape(name)}\s*\(', text):
                report['function_name_hits'].append({'id': row['id'], 'subset': row['task_type'], 'part': part,
                                                     'function': name, 'benchmark_item': task_id})

print(f'{len(rows)} training rows checked against IFEval ({len(ifeval)}) and HumanEval ({len(humaneval)})')
for bench in benchmarks:
    hits = report['ngram_hits'][bench]
    print(f'\n{bench}: {len({h["id"] for h in hits})} rows share a {N}-gram')
    for h in hits[:5]:
        print(f'  {h["subset"]:22s} {h["part"]:6s} {h["benchmark_item"]}: "{h["shared_ngram"]}"')
print(f'\nHumanEval function names defined in training rows: {len(report["function_name_hits"])}')
for h in report['function_name_hits'][:10]:
    print(f'  {h["subset"]:22s} {h["part"]:6s} def {h["function"]}(  ({h["benchmark_item"]})')
with open('results/contamination.json', 'w') as f:
    json.dump(report, f, indent=2)
