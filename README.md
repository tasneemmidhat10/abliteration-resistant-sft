# abliteration-resistant-sft

Can fine-tuning entangle a model's refusal representation with its capability representation, so that
removing refusal by *abliteration* also destroys capability? This repository holds the proof of concept
(Qwen2.5-1.5B-Instruct + LoRA, with auxiliary policy and capability heads on one hidden layer).

- Research idea, study design and roadmap: [docs/HANDBOOK.md](docs/HANDBOOK.md)
- Training data: [docs/DATA.md](docs/DATA.md)

## Setup on Bender (HPC cluster, Bonn)

```bash
git clone <this repo> ~/abliteration-resistant-sft && cd ~/abliteration-resistant-sft
mkdir -p logs
sbatch scripts/slurm/create_venv_bender.sh           # A40 nodes  -> .venv_intel  (~5 min)
# sbatch --partition=A100devel scripts/slurm/create_venv_bender.sh   # A100 nodes -> .venv_amd
```

Then, in every new shell or job script:

```bash
source scripts/activate.sh        # loads CUDA 12.4 + Python 3.12 modules and the matching venv
```

Bender's newest CUDA module is 12.4, which caps PyTorch at 2.6.0. The modules and venv setup are adapted
from [llm-foundry](https://github.com/Polygl0t/llm-foundry). Exact package versions are in `requirements.lock`.

One-time Hugging Face setup (gated datasets):

```bash
hf auth login     # read token; accept the terms of allenai/wildguardmix and walledai/AdvBench on the website
```

## Setup on a laptop

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch                 # the build for your platform, see pytorch.org
pip install -r requirements.txt && pip install -e . --no-deps
```

## Reproducing

| Step | Command | Output |
|---|---|---|
| 1. Data | `sbatch scripts/slurm/build_data.sh` | `data/{tulu_coding,tulu_instruction-following,wildguard}.jsonl` |
| 2. Split + token stats | `cd src && python -m SFT.utils` | `data/{train,val}.jsonl`, `data/stats.json` |
| 3. SFT, own loop | `sbatch scripts/slurm/train_own.sh` | `checkpoints/own-loop/epoch-{1,2,3}`, `log.jsonl` |
| 4. SFT, llm-foundry | `cd src && python -m SFT.prepare_foundry`, then `sbatch scripts/slurm/train_foundry.sh fp32` (and `bf16`) | `checkpoints/foundry-{fp32,bf16}/` |
| 5. Validation loss | `sbatch scripts/slurm/compare_val.sh` | `results/val_loss.json` |
| 6. Benchmarks | `sbatch evals/run_lm_eval.sh checkpoints/<model>` (one job per model), then `python evals/summarize_results.py` | `results/lm-eval/` |
| 7. Contamination | `python evals/contamination.py` | `results/contamination.json` |
| Checks (GPU) | `cd src && python -m SFT.train --checks` | correctness checks and benchmarks of the training loop |
| Tests | `pytest` | |

## SFT cycle: own training loop vs llm-foundry

Qwen3-0.6B-Base, full fine-tune on the 3-domain SFT data (refusal, code, instruction following), trained
twice: with our own PyTorch loop ([src/SFT/train.py](src/SFT/train.py)) and with llm-foundry's TRL
script ([alignment/sft_trainer.py](https://github.com/Polygl0t/llm-foundry/tree/main/alignment), commit `b86fc63`).
Both runs use the same data, the same chat format, the same trained tokens and the same hyperparameters,
so differences come from the implementation, not the setup.

### Setup

| | |
|---|---|
| Data | 2,694 train / 299 validation rows (about 900 / 100 per domain); the 7 rows over 2,048 tokens are dropped ([data/stats.json](data/stats.json)) |
| Chat format | Qwen3 template with the empty `<think>\n\n</think>\n\n` block in the prompt; loss only on the answer + `<|im_end|>` |
| Hyperparameters | lr 2e-5, cosine schedule, 5% warmup, AdamW (β₂ 0.999, weight decay 0.01, none on norms), grad-norm clip 1.0, batch 4 × 4 accumulation = 16, 3 epochs = 507 steps, max length 2,048 |
| Precision | own loop: fp32 weights + bf16 autocast (mixed precision). llm-foundry: run twice, without `--bf16` (fp32) and with `--bf16`, which stores the weights in bf16 (pure bf16) |
| Hardware / time | 1 × A40: own loop 16 min, llm-foundry fp32 27 min, llm-foundry bf16 13 min |

[src/SFT/prepare_foundry.py](src/SFT/prepare_foundry.py) gives llm-foundry the same split and a tokenizer whose
template marks the answer with `{% generation %}` (needed for `--assistant_only_loss`). It checks on every row
that this produces the same tokens and the same trained positions as the own loop.

### Validation loss (token-weighted mean, same function for every checkpoint)

| | coding | instruction following | refusal | all |
|---|---|---|---|---|
| base | 0.672 | 1.542 | 1.792 | 1.425 |
| own loop, epoch 1 / 2 / 3 | 0.287 / 0.306 / 0.358 | 1.424 / 1.498 / 1.666 | 1.291 / 1.286 / 1.389 | 1.152 / 1.194 / 1.321 |
| llm-foundry fp32, epoch 1 / 2 / 3 | 0.282 / 0.299 / 0.353 | 1.426 / 1.498 / 1.661 | 1.273 / 1.277 / 1.372 | 1.147 / 1.190 / 1.313 |
| llm-foundry bf16, epoch 1 / 2 / 3 | 0.284 / 0.279 / 0.281 | 1.381 / 1.376 / 1.378 | 1.310 / 1.286 / 1.286 | 1.133 / 1.124 / 1.125 |

TRL's own `eval_loss` is lower (0.960 for llm-foundry fp32 epoch 1) because it averages per-batch means
instead of tokens; the table above re-measures every checkpoint with [train.evaluate](src/SFT/train.py).

### Benchmarks (lm-evaluation-harness 0.4.13, chat template applied, 0-shot)

| Model | IFEval prompt strict | IFEval instr. strict | HumanEval (instruct) pass@1 | IFEval replies that never stop |
|---|---|---|---|---|
| base (no SFT) | 0.128 | 0.236 | 0.317 | 90.8% |
| own loop, epoch 1 | 0.381 | 0.487 | 0.311 | 31.4% |
| own loop, epoch 3 | 0.375 | 0.487 | 0.360 | 7.6% |
| llm-foundry fp32, epoch 1 | 0.342 | 0.447 | 0.311 | 24.4% |
| llm-foundry fp32, epoch 3 | 0.366 | 0.482 | 0.354 | 6.3% |
| llm-foundry bf16, epoch 3 | 0.322 | 0.447 | 0.402 | 14.4% |

Standard errors: about ±0.021 for IFEval (541 prompts) and ±0.037 for HumanEval (164 problems).
"Never stop" counts IFEval replies that reach the 1,280-token limit without producing `<|im_end|>`.

### Findings

1. **SFT teaches the chat format and instruction following.** IFEval prompt-level accuracy rises from 0.13 to
   0.37–0.38. The base model never ends its reply on 91% of prompts; after 3 epochs, 6–8%.
2. **The own loop and llm-foundry (fp32) give the same model.** Validation losses agree within 0.01 in every
   epoch and domain, and the epoch-3 benchmarks agree within noise (IFEval 0.375 vs 0.366, HumanEval 0.360 vs 0.354).
   At epoch 1 IFEval differs more (0.381 vs 0.342, about 2 standard errors); mid-training checkpoints also depend
   on the batch order, which differs between the two trainers.
3. **Lowest validation loss is not the best model.** Validation loss is lowest after epoch 1 and rises afterwards
   (the training set is small), but epoch 3 stops reliably (7.6% vs 31.4% unterminated replies) and scores at least
   as well. Epoch 3 is the checkpoint to use.
4. **llm-foundry's `--bf16` (pure bf16) changes the result.** With bf16 weights, Adam's small updates are rounded
   away: in a one-step test on this model, 77% of the weights of a `q_proj` matrix did not change at all at lr 2e-5.
   The run learns less (final training loss 0.93 vs 0.39 for fp32, validation loss flat after epoch 1), scores lower
   on IFEval (0.322 vs 0.366) and stops less reliably (14.4% vs 6.3%). Its higher HumanEval (0.402) is within noise.
   Keeping fp32 master weights (mixed precision) avoids this.
5. **HumanEval hardly changes with this data.** `humaneval_instruct` puts the function signature in the answer
   prefix, so it mostly tests code completion, which the base model already does. For reference, llm-foundry's
   evaluation README reports 0.366 for the released Qwen3-0.6B.

Contamination ([evals/contamination.py](evals/contamination.py), `results/contamination.json`): no 13-gram overlap with
IFEval; no HumanEval problem is copied in the training data, and 12 coding rows solve similar common exercises
(related to 3 of the 164 problems).

Limitations: one seed per configuration; one model size (0.6B); training batch order differs between the two
trainers. Generated samples are saved in `results/lm-eval/*/samples_*.jsonl`.

## Layout

```
configs/        training conditions: baseline / coupled / control
src/arft/       the package (data, training, directions, interventions)
src/SFT/        own SFT loop (train.py), data split and token stats (utils.py), llm-foundry inputs, validation comparison
evals/          lm-evaluation-harness job script and results summary
training/       evaluation, sweep and plotting scripts
scripts/        activate.sh, run_poc.sh, slurm/ job files (training, llm-foundry, validation)
checkpoints/    trained models (not committed)
results/        val_loss.json, lm-eval/ (scores and generated samples)
tests/          pytest
data/           SFT source files (3 x 1,000 rows), train/val split, stats.json, dataset card (README.md)
docs/           HANDBOOK.md, DATA.md, POC_REPORT.md
```
