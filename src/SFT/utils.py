import transformers
from datasets import load_dataset
import os
import json
import random
from collections import defaultdict, Counter
import pandas as pd


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/SFT/utils.py -> repo
DATA_DIR = os.path.join(REPO_ROOT, 'data')
SOURCE_FILES = ['tulu_coding.jsonl', 'tulu_instruction-following.jsonl', 'wildguard.jsonl']


def create_dataset(files=None):
    """No argument: the 3 source files as one 'train' split. A dict {'train': name, 'val': name}: those files."""
    if files is None:
        return load_dataset('json', data_files=[os.path.join(DATA_DIR, f) for f in SOURCE_FILES])
    return load_dataset('json', data_files={split: os.path.join(DATA_DIR, f) for split, f in files.items()})


def make_split(tokenizer, max_len=2048, val_fraction=0.1, seed=0):
    """Drop rows longer than max_len tokens, split 90/10 per subset, write data/train.jsonl and data/val.jsonl."""
    rows = list(create_dataset()['train'])
    kept = [r for r in rows                                    # same rendering as training (think block in the prompt)
            if len(tokenizer.apply_chat_template(r['messages'], tokenize=True,
                                                 enable_thinking=False)['input_ids']) <= max_len]
    rng = random.Random(seed)                                  # own generator: the split never depends on other code
    train, val = [], []
    for subset in sorted({r['task_type'] for r in kept}):      # stratify: split each subset separately
        group = [r for r in kept if r['task_type'] == subset]
        rng.shuffle(group)
        n_val = round(len(group) * val_fraction)
        val += group[:n_val]
        train += group[n_val:]
    for name, part in (('train.jsonl', train), ('val.jsonl', val)):
        with open(os.path.join(DATA_DIR, name), 'w') as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')  # ensure_ascii=False keeps non-English text readable
    counts = {name: dict(Counter(r['task_type'] for r in part)) for name, part in (('train', train), ('val', val))}
    print(f'dropped {len(rows) - len(kept)} rows over {max_len} tokens; split: {counts}')
    return counts

def load_tokenizer(model_name_or_path: str | os.PathLike ):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path)
    return tokenizer

def tokens_count(ds, tokenizer):

    per_row_lengths = []

    for row in ds['train']:
       stat = defaultdict(dict)
       full_ids = tokenizer.apply_chat_template(row['messages'], tokenize=True)['input_ids']
       prompt_ids = tokenizer.apply_chat_template(row['messages'][:1], tokenize=True, add_generation_prompt=True)['input_ids']

       # total - prompt is the response length only if the prompt tokens are
       # exactly the first tokens of the full conversation.
       if full_ids[:len(prompt_ids)] != prompt_ids:
           raise ValueError(f"Prompt is not a prefix of the full chat for row {row['id']}")

       total = len(full_ids)
       prompt_total = len(prompt_ids)
       response = total - prompt_total

       stat['id'] = row['id']
       stat['subset'] =  row['task_type']
       stat['total'] = total
       stat['prompt'] = prompt_total
       stat['response'] = response
       stat['response_chars'] = len(row['messages'][1]['content'])
       per_row_lengths.append(stat)

    return per_row_lengths

def summarize(records, max_lengths=(1024, 2048, 4096)):
    """Per-subset token statistics from the records returned by tokens_count."""
    df = pd.DataFrame(records)
    quantiles = [0.5, 0.95, 0.99]
    all_response_tokens = df['response'].sum()

    summary = {}
    for subset, group in df.groupby('subset'):
        stats = {'rows': len(group)}

        for column in ['total', 'prompt', 'response']:
            q = group[column].quantile(quantiles)
            stats[column] = {
                'p50': float(q[0.5]),
                'p95': float(q[0.95]),
                'p99': float(q[0.99]),
                'max': int(group[column].max()),
            }

        # Fraction of all trained (response) tokens that come from this subset.
        stats['response_token_share'] = round(float(group['response'].sum() / all_response_tokens), 3)
        stats['chars_per_response_token'] = round(float(group['response_chars'].sum() / group['response'].sum()), 2)
        # Rows that would be dropped at each candidate max_seq_len.
        stats['rows_over_max_length'] = {L: int((group['total'] > L).sum()) for L in max_lengths}

        summary[subset] = stats
    return summary


def main():
    ds = create_dataset()
    tokenizer = load_tokenizer('Qwen/Qwen3-0.6B-Base')
    records = tokens_count(ds, tokenizer)
    summary = summarize(records)
    print(json.dumps(summary, indent=2))

    summary['split'] = make_split(tokenizer)                   # also writes data/train.jsonl and data/val.jsonl

    stats_path = os.path.join(DATA_DIR, 'stats.json')
    with open(stats_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Saved {stats_path}')


if __name__ == '__main__':
    main()