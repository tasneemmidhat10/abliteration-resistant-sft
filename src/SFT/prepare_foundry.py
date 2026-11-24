"""Prepare inputs for llm-foundry's sft_trainer.py so it trains on exactly what our own loop trains on.

Writes (both under checkpoints/, which git ignores):
  checkpoints/foundry-data/{train,validation}  our data/train.jsonl and data/val.jsonl, via Dataset.save_to_disk
  checkpoints/qwen3-0.6b-base-prepared         the base model + a tokenizer that TRL can use for assistant-only loss
Run once: cd src && python -m SFT.prepare_foundry
"""
import os
import torch
import transformers
from datasets import Dataset
from .utils import create_dataset, load_tokenizer, REPO_ROOT
from .train import tokenize_dataset

MODEL = 'Qwen/Qwen3-0.6B-Base'
DATA_OUT = os.path.join(REPO_ROOT, 'checkpoints', 'foundry-data')
MODEL_OUT = os.path.join(REPO_ROOT, 'checkpoints', 'qwen3-0.6b-base-prepared')

# Same text as Qwen3's template with enable_thinking=False, for user/assistant chats. The difference:
# {% generation %} ... {% endgeneration %} marks the tokens TRL trains on (answer + <|im_end|> + newline).
TEMPLATE = (
    "{%- for message in messages %}"
    "{%- if message.role == 'assistant' %}"
    "{{- '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' }}"
    "{% generation %}{{- message.content + '<|im_end|>\\n' }}{% endgeneration %}"
    "{%- else %}"
    "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' }}{%- endif %}"
)


def main():
    # 1. Data: our split, only the column TRL needs.
    ds = create_dataset({'train': 'train.jsonl', 'val': 'val.jsonl'})
    for split, folder in (('train', 'train'), ('val', 'validation')):
        Dataset.from_dict({'messages': ds[split]['messages']}).save_to_disk(os.path.join(DATA_OUT, folder))

    # 2. Check: new template = old rendering, and its assistant mask = our labels, on every row.
    original = load_tokenizer(MODEL)
    prepared = load_tokenizer(MODEL)
    prepared.chat_template = TEMPLATE
    prepared.eos_token = '<|im_end|>'                  # llm-foundry refuses pad == eos (both <|endoftext|> in the base)
    for split in ('train', 'val'):
        encoded, _ = tokenize_dataset(ds, original, max_len=10**9, split=split)  # what our loop trains on
        for row, enc in zip(ds[split], encoded):
            out = prepared.apply_chat_template(row['messages'], tokenize=True, return_dict=True,
                                               return_assistant_tokens_mask=True)
            assert out['input_ids'] == enc['input_ids'], f"tokens differ for {row['id']}"
            assert out['assistant_masks'] == [int(l != -100) for l in enc['labels']], f"mask differs for {row['id']}"
        print(f'{split}: {len(encoded)} rows, tokens and trained positions identical to our loop')

    # 3. Model: the unchanged base weights (fp32) saved next to the prepared tokenizer.
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    model.generation_config.eos_token_id = [prepared.convert_tokens_to_ids('<|im_end|>'), prepared.pad_token_id]
    model.generation_config.max_new_tokens = None    # base config says 2048; it would override lm-eval's limits
    model.save_pretrained(MODEL_OUT)
    prepared.save_pretrained(MODEL_OUT)
    print(f'saved {DATA_OUT} and {MODEL_OUT}')


if __name__ == '__main__':
    main()
