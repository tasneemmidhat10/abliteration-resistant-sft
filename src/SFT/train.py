import torch
from torch.utils.data import DataLoader
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

def compute_loss(logits, labels): #Logits are of shape (B, T, V) and labels (B, T)
       predictions = logits[:, :-1, :]
       targets = labels[:, 1:]

       predictions = torch.flatten(predictions, start_dim=0, end_dim=1)
       targets = torch.flatten(targets)

       loss = torch.nn.functional.cross_entropy(predictions, targets, ignore_index=-100)
       trained_tokens = (targets != -100).sum().item()

       return loss, trained_tokens


def per_token_losses(logits, labels):
    """Loss at every trained position of ONE row (logits (T, V), labels (T,)), no averaging."""
    losses = torch.nn.functional.cross_entropy(logits[:-1].float(), labels[1:],
                                               ignore_index=-100, reduction='none')
    return losses[labels[1:] != -100]


def check_padding_invariance(model, encoded_rows, pad_token_id, tol=1e-4):
    """Check 5: a short row must get the same per-token losses alone and when padded next to a long row."""
    short = next(r for r in encoded_rows if 100 <= len(r['input_ids']) <= 200)
    long = next(r for r in encoded_rows if 500 <= len(r['input_ids']) <= 700)
    n = len(short['input_ids'])

    alone = collate_fn([short], pad_token_id)          # (1, n), no padding
    padded = collate_fn([short, long], pad_token_id)   # (2, len(long)), row 0 = short + padding

    with torch.no_grad():
        logits_alone = model(input_ids=alone['input_ids'],
                             attention_mask=alone['attention_mask']).logits[0]
        # Row 0, cut back to the short row's real length: the rest is padding.
        logits_padded = model(input_ids=padded['input_ids'],
                              attention_mask=padded['attention_mask']).logits[0, :n]

    loss_alone = per_token_losses(logits_alone, alone['labels'][0])
    loss_padded = per_token_losses(logits_padded, padded['labels'][0, :n])

    if loss_alone.shape != loss_padded.shape:
        raise ValueError(f'Different number of trained tokens: {loss_alone.shape} vs {loss_padded.shape}')
    max_diff = (loss_alone - loss_padded).abs().max().item()
    if max_diff > tol:
        raise ValueError(f'Padding changes the short row\'s losses: max difference {max_diff:.2e} > {tol}')
    print(f'Check 5 passed: short row ({n} tokens) padded to {padded["input_ids"].shape[1]}, '
          f'{len(loss_alone)} trained tokens, max per-token difference {max_diff:.2e}')



def main():
        ds = create_dataset()
        tokenizer = load_tokenizer('Qwen/Qwen3-0.6B-Base')
        pad_token_id = tokenizer.pad_token_id

        encoded_rows, dropped_rows = tokenize_dataset(ds, tokenizer=tokenizer, max_len=2048)
        g = torch.Generator().manual_seed(0)

        custom_collate_fn = partial(collate_fn, pad_token_id=pad_token_id)
        loader = DataLoader(encoded_rows, batch_size=8, shuffle=True,
                             collate_fn=custom_collate_fn, generator=g)
        print(f'The number of batches is {len(loader)}', '\n')

        real_tokens_count = 0
        pad_tokens_count = 0

        for batch in loader:
               real_tokens_count+= (batch['attention_mask'] == 1).sum().item()
               pad_tokens_count += (batch['attention_mask'] == 0).sum().item()

        print(f'The padding fraction is {1 - (real_tokens_count/(pad_tokens_count + real_tokens_count))}')

        model = transformers.AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-Base", dtype=torch.float32)
        model.eval()
        batch = next(iter(loader))
        with torch.no_grad():
            outputs = model(input_ids = batch['input_ids'][:2, :], attention_mask= batch['attention_mask'][:2, :])
            outputs2 = model(input_ids = batch['input_ids'][:2, :], attention_mask= batch['attention_mask'][:2, :], labels=batch['labels'][:2, :])

        loss, trained_tokens = compute_loss(logits= outputs.logits.float(), labels=batch['labels'][:2, :])

        if abs(loss.item() - outputs2.loss) > 1e-5:
               raise ValueError('The difference is larger than 1e-5')
        check_padding_invariance(model, encoded_rows, pad_token_id)
        if trained_tokens == (batch['labels'][:2, 1:] != -100).sum():
               return print('Matched')


if __name__ == '__main__':
        main()