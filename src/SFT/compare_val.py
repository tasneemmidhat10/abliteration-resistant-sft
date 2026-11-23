"""Validation loss per subset for every checkpoint, all measured with the same function (train.evaluate).
Run on a GPU:  cd src && python -m SFT.compare_val      Output: results/val_loss.json + a printed table.
"""
import os
import json
import torch
import transformers
from .utils import create_dataset, load_tokenizer, REPO_ROOT
from .train import tokenize_dataset, evaluate

CHECKPOINTS = {                                    # name -> folder under checkpoints/
    'base': 'qwen3-0.6b-base-prepared',
    'own epoch 1': 'own-loop/epoch-1', 'own epoch 2': 'own-loop/epoch-2', 'own epoch 3': 'own-loop/epoch-3',
    'trl-fp32 epoch 1': 'foundry-fp32/checkpoint-169', 'trl-fp32 epoch 2': 'foundry-fp32/checkpoint-338',
    'trl-fp32 epoch 3': 'foundry-fp32/final',
    'trl-bf16 epoch 1': 'foundry-bf16/checkpoint-169', 'trl-bf16 epoch 2': 'foundry-bf16/checkpoint-338',
    'trl-bf16 epoch 3': 'foundry-bf16/final',
}


def main():
    tokenizer = load_tokenizer('Qwen/Qwen3-0.6B-Base')     # original template: the one our encode() was checked on
    val_rows, _ = tokenize_dataset(create_dataset({'val': 'val.jsonl'}), tokenizer, max_len=2048, split='val')
    results = {}
    for name, folder in CHECKPOINTS.items():
        path = os.path.join(REPO_ROOT, 'checkpoints', folder)
        model = transformers.AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).to('cuda')
        results[name] = evaluate(model, val_rows, tokenizer.pad_token_id, batch_size=4, use_bf16=True)
        print(f"{name:18s} " + '  '.join(f'{k} {v:.3f}' for k, v in results[name].items()), flush=True)
        del model
        torch.cuda.empty_cache()
    out = os.path.join(REPO_ROOT, 'results', 'val_loss.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'saved {out}')


if __name__ == '__main__':
    main()
