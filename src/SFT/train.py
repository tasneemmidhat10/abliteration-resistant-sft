import torch
from torch.utils.data import DataLoader
import transformers
import math
import time
import os
import sys
import json
from collections import defaultdict
from .utils import create_dataset, load_tokenizer, REPO_ROOT
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
       predictions = logits[:, :-1, :].flatten(0, 1)        # (B*(T-1), V): position t predicts token t+1
       targets = labels[:, 1:].flatten()                    # (B*(T-1),)
       keep = targets != -100                               # answer tokens only; prompt and padding dropped here,
       predictions = predictions[keep].float()              # so the fp32 copy covers only the trained positions
       loss = torch.nn.functional.cross_entropy(predictions, targets[keep])
       trained_tokens = int(keep.sum())

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


def benchmark_step(model, rows, pad_token_id, use_bf16, label, peak_tflops, n_steps=10, warmup_steps=2):
    """Experiment 5: time full training steps (forward, backward, clip, optimizer step) on one fixed batch.

    peak_tflops is the GPU's advertised peak for this number format, used for MFU
    (A40: fp32 37.4, TF32 74.8, bf16 149.7 dense TFLOPS).
    Note: this trains the model on `rows`, so run it after all correctness checks.
    """
    print(f'\n--- Benchmark: {label} ---')
    model.to('cuda')
    model.train()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    batch = collate_fn(rows, pad_token_id=pad_token_id)
    for key in ('input_ids', 'labels', 'attention_mask'):
        batch[key] = batch[key].to('cuda')
    B, T = batch['input_ids'].shape
    real_tokens = int(batch['attention_mask'].sum())
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = build_optimizer(model=model, lr=2e-5, weight_decay=0.01)

    def one_step():
        loss, _ = forward_loss(model=model, batch=batch, use_bf16=use_bf16)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    result = None
    try:
        # Not timed: the first steps pick CUDA kernels and fill PyTorch's memory cache.
        for _ in range(warmup_steps):
            one_step()
        # GPU work is asynchronous: wait for it to finish before reading the clock.
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_steps):
            one_step()
        torch.cuda.synchronize()
        seconds = (time.perf_counter() - start) / n_steps

        # About 6 FLOPs per parameter per token (2 forward + 4 backward). The GPU also
        # computes on padding, so count all B*T positions for utilisation.
        flops_per_step = 6 * n_params * B * T
        mfu = flops_per_step / seconds / (peak_tflops * 1e12)
        result = {'seconds_per_step': seconds,
                  'tokens_per_second': real_tokens / seconds,
                  'peak_memory_gb': torch.cuda.max_memory_allocated() / 1e9,
                  'mfu': mfu}
        print(f'batch {B} x {T} ({real_tokens} real tokens) | {seconds:.3f} s/step | '
              f'{result["tokens_per_second"]:,.0f} tokens/s | peak {result["peak_memory_gb"]:.1f} GB | MFU {mfu:.1%}')
    except torch.cuda.OutOfMemoryError:
        print(f'Out of memory with batch {B} x {T} '
              f'(peak before failing: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB)')
    finally:
        # Free gradients, optimizer states and cached blocks so the next run starts clean.
        del optimizer
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    return result


CONFIG = {
    'model': 'Qwen/Qwen3-0.6B-Base',
    'max_len': 2048,          # rows longer than this were already removed by make_split
    'micro_batch': 4,         # rows per forward/backward pass
    'grad_accum': 4,          # micro-batches per optimizer step -> effective batch 16
    'epochs': 3,
    'lr': 2e-5,
    'weight_decay': 0.01,
    'warmup_ratio': 0.05,     # 5% of optimizer steps
    'max_grad_norm': 1.0,
    'use_bf16': True,         # mixed precision: fp32 weights, bf16 autocast
    'seed': 0,
    'log_every': 10,          # optimizer steps between log lines
    'out_dir': os.path.join(REPO_ROOT, 'checkpoints', 'own-loop'),
}


def to_cuda(batch):
    """Move the tensors of a collated batch to the GPU; row_id and subset stay Python lists."""
    return {k: v.to('cuda') if torch.is_tensor(v) else v for k, v in batch.items()}


def evaluate(model, rows, pad_token_id, batch_size, use_bf16):
    """Validation loss per subset, averaged over trained tokens (not over rows)."""
    model.eval()
    loss_sum, token_count = defaultdict(float), defaultdict(int)
    with torch.no_grad():
        for i in range(0, len(rows), batch_size):
            batch = to_cuda(collate_fn(rows[i:i + batch_size], pad_token_id))
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(batch['input_ids'], attention_mask=batch['attention_mask']).logits
            for j, subset in enumerate(batch['subset']):            # per row, so each loss goes to its subset
                losses = per_token_losses(logits[j], batch['labels'][j])
                loss_sum[subset] += losses.sum().item()
                token_count[subset] += len(losses)
    model.train()
    result = {s: loss_sum[s] / token_count[s] for s in sorted(loss_sum)}
    result['all'] = sum(loss_sum.values()) / sum(token_count.values())
    return result


def generate_samples(model, tokenizer, raw_rows, max_new_tokens=200):
    """Greedy answers to the first validation prompt of each subset: does the model answer and stop?"""
    model.eval()
    im_end = tokenizer.convert_tokens_to_ids('<|im_end|>')
    for subset in sorted({r['task_type'] for r in raw_rows}):
        row = next(r for r in raw_rows if r['task_type'] == subset)
        inputs = tokenizer.apply_chat_template(row['messages'][:1], add_generation_prompt=True,
                                               enable_thinking=False, return_tensors='pt').to('cuda')
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                 eos_token_id=im_end, pad_token_id=tokenizer.pad_token_id)
        answer = out[0, inputs['input_ids'].shape[1]:]                # only the new tokens
        stopped = im_end in answer.tolist()
        print(f'\n[{subset}] stopped on <|im_end|>: {stopped}\n{tokenizer.decode(answer, skip_special_tokens=True)[:500]}')
    model.train()


def save_model(model, tokenizer, out_dir):
    """Save (in fp32), set up so generation and lm-eval use the same format as training."""
    tokenizer.chat_template = tokenizer.chat_template.replace(  # always put the empty think block in the prompt,
        'enable_thinking is defined and enable_thinking is false', 'true')  # as in training (enable_thinking=False)
    tokenizer.eos_token = '<|im_end|>'                           # the model was trained to end answers with this
    im_end = tokenizer.convert_tokens_to_ids('<|im_end|>')
    model.generation_config.eos_token_id = [im_end, tokenizer.convert_tokens_to_ids('<|endoftext|>')]
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.save_pretrained(out_dir)       # fp32 as trained; converting the live model to bf16 would round the master weights
    tokenizer.save_pretrained(out_dir)
    print(f'saved {out_dir}')


def train(cfg):
    torch.manual_seed(cfg['seed'])
    tokenizer = load_tokenizer(cfg['model'])
    pad_token_id = tokenizer.pad_token_id
    ds = create_dataset({'train': 'train.jsonl', 'val': 'val.jsonl'})
    train_rows, dropped = tokenize_dataset(ds, tokenizer, cfg['max_len'], split='train')
    val_rows, _ = tokenize_dataset(ds, tokenizer, cfg['max_len'], split='val')
    assert not dropped, 'make_split should already have removed long rows'

    loader = DataLoader(train_rows, batch_size=cfg['micro_batch'], shuffle=True,
                        collate_fn=partial(collate_fn, pad_token_id=pad_token_id),
                        generator=torch.Generator().manual_seed(cfg['seed']))   # same batch order every run
    steps_per_epoch = math.ceil(len(loader) / cfg['grad_accum'])
    total_steps = steps_per_epoch * cfg['epochs']
    warmup_steps = round(cfg['warmup_ratio'] * total_steps)

    model = transformers.AutoModelForCausalLM.from_pretrained(cfg['model'], dtype=torch.float32).to('cuda')
    model.train()
    optimizer = build_optimizer(model, cfg['lr'], cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, partial(lr_lambda, warmup_steps=warmup_steps, total_steps=total_steps))

    os.makedirs(cfg['out_dir'], exist_ok=True)
    log_file = open(os.path.join(cfg['out_dir'], 'log.jsonl'), 'w')
    def log(record):                                              # one JSON line per event, also printed
        print(record, flush=True)
        log_file.write(json.dumps(record) + '\n')
        log_file.flush()

    log({'step': 0, 'val': evaluate(model, val_rows, pad_token_id, cfg['micro_batch'], cfg['use_bf16']),
         'total_steps': total_steps, 'warmup_steps': warmup_steps, 'train_rows': len(train_rows)})
    step, start = 0, time.perf_counter()
    for epoch in range(1, cfg['epochs'] + 1):
        micro_batches = list(loader)                              # this epoch's shuffled micro-batches
        for i in range(0, len(micro_batches), cfg['grad_accum']):
            group = [to_cuda(b) for b in micro_batches[i:i + cfg['grad_accum']]]
            # Gradient accumulation: weight each micro-batch by its share of the step's trained tokens,
            # so the step's gradient is the per-token mean over all of them (not a mean of means).
            step_tokens = sum(int((b['labels'][:, 1:] != -100).sum()) for b in group)
            step_loss = 0.0
            for b in group:
                loss, n_tokens = forward_loss(model, b, cfg['use_bf16'])
                (loss * n_tokens / step_tokens).backward()       # gradients add up across micro-batches
                step_loss += loss.item() * n_tokens / step_tokens
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['max_grad_norm'])
            lr = scheduler.get_last_lr()[0]                       # the lr this step actually uses
            optimizer.step()
            scheduler.step()                                      # once per optimizer step, not per micro-batch
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % cfg['log_every'] == 0 or step == total_steps:
                log({'step': step, 'epoch': epoch, 'loss': round(step_loss, 4), 'lr': lr,
                     'grad_norm': round(grad_norm.item(), 3), 'minutes': round((time.perf_counter() - start) / 60, 2)})
        log({'step': step, 'epoch': epoch,
             'val': evaluate(model, val_rows, pad_token_id, cfg['micro_batch'], cfg['use_bf16'])})
        save_model(model, tokenizer, os.path.join(cfg['out_dir'], f'epoch-{epoch}'))

    generate_samples(model, tokenizer, list(ds['val']))
    log_file.close()


def main():
    if '--checks' in sys.argv:               # python -m SFT.train --checks  -> correctness checks and benchmarks
        run_checks()
    else:                                    # python -m SFT.train           -> full training run
        train(CONFIG)


def run_checks():
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

        # Experiment 5: speed and memory. Runs last because it trains `model` on these rows.
        del model_bf16
        torch.cuda.empty_cache()
        bench_rows = [r for r in encoded_rows if 900 <= len(r['input_ids']) <= 1100][:8]
        benchmark_step(model, bench_rows, pad_token_id, use_bf16=False, label='fp32', peak_tflops=37.4)
        torch.set_float32_matmul_precision('high')       # fp32 matmuls on tensor cores (TF32)
        benchmark_step(model, bench_rows, pad_token_id, use_bf16=False, label='fp32 + TF32', peak_tflops=74.8)
        torch.set_float32_matmul_precision('highest')    # back to true fp32
        benchmark_step(model, bench_rows, pad_token_id, use_bf16=True, label='bf16 autocast', peak_tflops=149.7)

        # The 8 longest rows (up to 2048 tokens): may run out of memory, which is itself a result.
        longest_rows = sorted(encoded_rows, key=lambda r: len(r['input_ids']), reverse=True)[:8]
        benchmark_step(model, longest_rows, pad_token_id, use_bf16=True,
                       label='bf16 autocast, 8 longest rows', peak_tflops=149.7)
        benchmark_step(model, longest_rows, pad_token_id, use_bf16=False,
                       label='fp32, 8 longest rows', peak_tflops=37.4)


if __name__ == '__main__':
        main()