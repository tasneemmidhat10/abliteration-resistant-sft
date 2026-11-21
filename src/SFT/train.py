import torch
import transformers
from .utils import create_dataset, load_tokenizer
from functools import partial


def encode(example, tokenizer):
    full_example_IDS = tokenizer.apply_chat_template(example['messages'], tokenize=True,
                                   enable_thinking=False)['input_ids']
    prompt_IDS = tokenizer.apply_chat_template(example['messages'][:1], tokenize=True, 
                                               add_generation_prompt=True, enable_thinking=False)['input_ids']
    if full_example_IDS[:len(prompt_IDS)] != prompt_IDS:
               raise ValueError(f"Prompt is not a prefix of the full chat for row {example['id']}")
    
    labels = [-100] * len(prompt_IDS) + full_example_IDS[len(prompt_IDS):]

    return {"input_ids": full_example_IDS, "labels":labels, "row_id":example['id'], "subset":example['task_type']}

def tokenize_dataset(ds, tokenizer, max_len, split='train'):
        dropped_rows = []
        encoded_rows = []
        for row in ds[split]:
            encoded_output = encode(row, tokenizer=tokenizer)
            if len(encoded_output['input_ids']) > max_len:
                    dropped_rows.append(
                            {
                                'row_id': row['id'],
                                'subset': row['task_type'],
                                'length': len(encoded_output['input_ids'])
                            })
                    continue
            encoded_rows.append(encoded_output)
        return encoded_rows, dropped_rows

def collate_fn(batch, pad_token_id):
       output = {}
       B_max_len = max([len(row['input_ids']) for row in batch])
       output['input_ids'] = torch.tensor([row['input_ids'] + [pad_token_id] * (B_max_len - len(row['input_ids'])) for row in batch], dtype=torch.long)
       output['labels'] = torch.tensor([row['labels'] + [-100] * (B_max_len - len(row['labels'])) for row in batch], dtype=torch.long)
       output['attention_mask'] = torch.tensor([[1] * len(row['input_ids']) + [0] * (B_max_len - len(row['input_ids'])) for row in batch], dtype=torch.long)
       output['row_id'] = [row['row_id'] for row in batch]
       output['subset'] = [row['subset'] for row in batch]
       return output

def main():
        ds = create_dataset()
        tokenizer = load_tokenizer('Qwen/Qwen3-0.6B-Base')
        encoded_rows, dropped_rows = tokenize_dataset(ds, tokenizer=tokenizer, max_len=2048)
        pad_token_id = tokenizer.pad_token_id
        output = collate_fn(encoded_rows[:6], pad_token_id=pad_token_id)

if __name__ == '__main__':
        main()