import transformers
from datasets import load_dataset
import os
import json
from collections import defaultdict
import pandas as pd


def create_dataset():
    ds = load_dataset('json', data_files=['/home/s85tatta/abliteration-resistant-sft/data/tulu_coding.jsonl', 
                                          '/home/s85tatta/abliteration-resistant-sft/data/tulu_instruction-following.jsonl',
                                          '/home/s85tatta/abliteration-resistant-sft/data/wildguard.jsonl'])
    return ds

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

    # src/SFT/utils.py -> two levels up is the repo root.
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    stats_path = os.path.join(repo_root, 'data', 'stats.json')
    with open(stats_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Saved {stats_path}')


if __name__ == '__main__':
    main()