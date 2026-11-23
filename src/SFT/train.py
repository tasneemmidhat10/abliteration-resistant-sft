import torch
from torch.utils.data import DataLoader
import transformers
import math
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

       predictions = torch.flatten(predictions, start_dim=0, end_dim=1).float()
       targets = torch.flatten(targets)

       loss = torch.nn.functional.cross_entropy(predictions, targets, ignore_index=-100)
       trained_tokens = (targets != -100).sum().item()

       return loss, trained_tokens

def build_optimizer(model, lr, weight_decay):
    optimizer_grouped_parameters = [
            {'params': [], 'weight_decay': weight_decay},
            {'params': [], 'weight_decay': 0.0}
            ]
    for _, p in model.named_parameters():
        if p.ndim >= 2 and p.requires_grad:
              optimizer_grouped_parameters[0]['params'].append(p)
        if p.ndim < 2 and p.requires_grad:
              optimizer_grouped_parameters[1]['params'].append(p)   
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
    return optimizer

def lr_lambda(step, warmup_steps, total_steps):
    
    if step < warmup_steps:
        multiplier = step / warmup_steps
    else:
          progress = min(1.0, (step - warmup_steps) / (total_steps - warmup_steps))
          multiplier = 0.5 * (1 + math.cos(math.pi * progress))
    return multiplier

def forward_loss(model, batch, use_bf16):
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
        logits = model(batch['input_ids'], attention_mask=batch['attention_mask']).logits
        loss, trained_tokens = compute_loss(logits=logits, labels=batch['labels'])
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

def check_param_groups(model, optimizer):
    """Check 1: every parameter is in exactly one group, and every norm is in the no-decay group."""
    decay, no_decay = optimizer.param_groups
    counts = [sum(p.numel() for p in group['params']) for group in (decay, no_decay)]
    for group, count in zip((decay, no_decay), counts):
        print(f"weight_decay={group['weight_decay']}: {len(group['params'])} tensors, {count:,} parameters")

    model_total = sum(p.numel() for p in model.parameters())
    if sum(counts) != model_total:
        raise ValueError(f'Groups hold {sum(counts):,} parameters, the model has {model_total:,}')

    # The optimizer stores tensors, not names, so compare tensors by identity.
    decay_ids = {id(p) for p in decay['params']}
    no_decay_ids = {id(p) for p in no_decay['params']}
    if decay_ids & no_decay_ids:
        raise ValueError('Some parameters are in both groups')
    norms_with_decay = [name for name, p in model.named_parameters()
                        if 'norm' in name and id(p) not in no_decay_ids]
    if norms_with_decay:
        raise ValueError(f'Norm parameters outside the no-decay group: {norms_with_decay[:5]}')
    print(f'Check 1 passed: {model_total:,} parameters, all norms without weight decay')


def check_schedule(warmup_steps, total_steps, peak_lr, tol=1e-12):
    """Check 2: our LambdaLR schedule equals transformers' cosine-with-warmup schedule at every step."""

    def lrs_of(make_scheduler):
        # A throwaway one-parameter optimizer: stepping a scheduler changes its
        # optimizer's lr, so never run this on the real training optimizer.
        opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=peak_lr)
        scheduler = make_scheduler(opt)
        lrs = []
        for _ in range(total_steps):
            lrs.append(opt.param_groups[0]['lr'])
            opt.step()          # no gradients, so nothing changes; keeps the step order correct
            scheduler.step()
        return lrs

    ours = lrs_of(lambda opt: torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=partial(lr_lambda, warmup_steps=warmup_steps, total_steps=total_steps)))
    reference = lrs_of(lambda opt: transformers.get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps))

    max_diff = max(abs(a - b) for a, b in zip(ours, reference))
    if max_diff > tol:
        raise ValueError(f'Schedule differs from transformers by up to {max_diff:.2e}')

    middle = warmup_steps + (total_steps - warmup_steps) // 2
    if ours[0] != 0.0:
        raise ValueError(f'lr at step 0 should be 0, got {ours[0]}')
    if abs(ours[warmup_steps] - peak_lr) > tol:
        raise ValueError(f'lr at step {warmup_steps} should be the peak {peak_lr}, got {ours[warmup_steps]}')
    if abs(ours[middle] - peak_lr / 2) > 0.01 * peak_lr:
        raise ValueError(f'lr at step {middle} should be about half the peak, got {ours[middle]}')
    if ours[-1] > 1e-3 * peak_lr:
        raise ValueError(f'lr at the last step should be about 0, got {ours[-1]}')
    print(f'Check 2 passed: {total_steps} steps match transformers (max difference {max_diff:.1e}); '
          f'lr = 0 -> {ours[warmup_steps]:.1e} (step {warmup_steps}) -> {ours[middle]:.1e} (step {middle}) -> {ours[-1]:.1e}')

def check_one_step(model, rows, pad_token_id, lr, use_bf16, label, require_decrease=True):
    """Check 3 and the Part 5 experiments: one optimizer step on a fixed batch, on the GPU.

    - fp32 model + use_bf16=True  -> mixed precision (fp32 master weights, bf16 autocast)
    - fp32 model + use_bf16=False -> plain fp32
    - bf16 model + use_bf16=False -> pure bf16 (weights, gradients and Adam states all bf16)
    """
    print(f'\n--- One step: {label} ---')
    torch.cuda.reset_peak_memory_stats()
    memory_at_start = torch.cuda.memory_allocated()

    model.to('cuda')
    optimizer = build_optimizer(model=model, lr=lr, weight_decay=0.01)
    batch = collate_fn(rows, pad_token_id=pad_token_id)
    for key in ('input_ids', 'labels', 'attention_mask'):
        batch[key] = batch[key].to('cuda')
    weight = model.model.layers[0].self_attn.q_proj.weight

    with torch.no_grad():
        # Measured without autocast: fp32 for the fp32 model, bf16 for the bf16 model.
        loss_before, _ = forward_loss(model=model, batch=batch, use_bf16=False)
        if model.dtype == torch.float32:
            # Experiment 2: same batch, same fp32 weights, forward pass in bf16.
            loss_autocast, _ = forward_loss(model=model, batch=batch, use_bf16=True)
            print(f'Loss fp32 vs bf16 autocast (same batch): {loss_before.item():.5f} vs '
                  f'{loss_autocast.item():.5f}, difference {abs(loss_before.item() - loss_autocast.item()):.2e}')
        before = weight.detach().clone()

    model.train()
    loss, _ = forward_loss(model=model, batch=batch, use_bf16=use_bf16)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model.eval()

    after = weight.detach()
    # Compare in fp32 so the measurement itself adds no rounding.
    max_change = (after.float() - before.float()).abs().max().item()
    fraction_changed = (after != before).float().mean().item()

    with torch.no_grad():
        loss_after, _ = forward_loss(model=model, batch=batch, use_bf16=False)

    print(f'weight dtype {weight.dtype}, autocast {use_bf16}')
    print(f'fraction of q_proj entries changed: {fraction_changed:.4f}')
    if weight.dtype == torch.bfloat16:
        # A bf16 value can only move if lr is more than half the gap to its neighbour: |w| < 2**-7.
        predicted = (before.abs() < 2**-7).float().mean().item()
        print(f'predicted fraction (|w| < 2^-7):     {predicted:.4f}')
    print(f'largest change: {max_change:.4e} (lr = {lr:.1e})')
    print(f'loss before -> after: {loss_before.item():.4f} -> {loss_after.item():.4f}')
    print(f'gradient norm: {grad_norm.item():.3f}')
    print(f'peak memory for this run: {(torch.cuda.max_memory_allocated() - memory_at_start) / 1e9:.2f} GB')

    if not torch.isfinite(grad_norm):
        raise ValueError(f'{label}: gradient norm is not finite ({grad_norm.item()})')
    if require_decrease and loss_after >= loss_before:
        raise ValueError(f'{label}: loss did not decrease ({loss_before.item():.4f} -> {loss_after.item():.4f})')


def main():
        ds = create_dataset()
        tokenizer = load_tokenizer('Qwen/Qwen3-0.6B-Base')
        pad_token_id = tokenizer.pad_token_id

        encoded_rows, dropped_rows = tokenize_dataset(ds, tokenizer=tokenizer, max_len=2048)
        g = torch.Generator().manual_seed(0)

        custom_collate_fn = partial(collate_fn, pad_token_id=pad_token_id)
        loader = DataLoader(encoded_rows, batch_size=8, shuffle=True,
                             collate_fn=custom_collate_fn, generator=g)
        print(f'The number of batches is {len(loader)}')

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

        loss, trained_tokens = compute_loss(logits= outputs.logits, labels=batch['labels'][:2, :])

        if abs(loss.item() - outputs2.loss) > 1e-5:
               raise ValueError('The difference is larger than 1e-5')
        check_padding_invariance(model, encoded_rows, pad_token_id)
        if trained_tokens != (batch['labels'][:2, 1:] != -100).sum().item():
               raise ValueError('trained_tokens does not match the number of unmasked labels')

        peak_lr, warmup_steps, total_steps = 2e-5, 25, 510
        optimizer = build_optimizer(model=model, lr=peak_lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=partial(lr_lambda, warmup_steps=warmup_steps, total_steps=total_steps))
        check_param_groups(model, optimizer)
        check_schedule(warmup_steps, total_steps, peak_lr)
        check_one_step(model=model, rows=encoded_rows[:4], pad_token_id=pad_token_id, lr=2e-5,
                       use_bf16=True, label='mixed precision (fp32 weights, bf16 autocast)')

        # Experiment 4: a second copy of the model stored entirely in bf16. The loss may or
        # may not go down, so this run only reports and never fails on the loss.
        model_bf16 = transformers.AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-Base", dtype=torch.bfloat16)
        model_bf16.eval()
        check_one_step(model=model_bf16, rows=encoded_rows[:4], pad_token_id=pad_token_id, lr=2e-5,
                       use_bf16=False, label='pure bf16', require_decrease=False)
        print('All checks passed')


if __name__ == '__main__':
        main()