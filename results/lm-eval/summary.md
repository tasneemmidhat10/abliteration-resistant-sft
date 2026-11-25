# Benchmark results: Qwen3-0.6B-Base after SFT

Generated on 2026-11-25 by `evals/summarize_results.py` from the newest lm-eval run of each model.
All numbers are percentages; ± is one standard error.

| Model | IFEval: all instructions followed | IFEval: single instructions followed | HumanEval: problems solved | IFEval replies that never end |
|---|---|---|---|---|
| Base model (no SFT) | 12.8 ± 1.4 | 23.6 | 31.7 ± 3.6 | 90.8 |
| Own loop, epoch 1 | 38.1 ± 2.1 | 48.7 | 31.1 ± 3.6 | 31.4 |
| Own loop, epoch 3 | 37.5 ± 2.1 | 48.7 | 36.0 ± 3.8 | 7.6 |
| llm-foundry fp32, epoch 1 | 34.2 ± 2.0 | 44.7 | 31.1 ± 3.6 | 24.4 |
| llm-foundry fp32, epoch 3 | 36.6 ± 2.1 | 48.2 | 35.4 ± 3.7 | 6.3 |
| llm-foundry pure bf16, epoch 3 | 32.2 ± 2.0 | 44.7 | 40.2 ± 3.8 | 14.4 |

## What the columns mean

- **IFEval: all instructions followed** – share of the 541 IFEval prompts where the reply met every instruction
  (for example "no commas" *and* "at least 300 words"), checked strictly by code. The usual headline IFEval number.
- **IFEval: single instructions followed** – share of the individual instructions met, counted one by one.
  Always higher than the column before, because one missed instruction fails a whole prompt.
- **HumanEval: problems solved** – share of the 164 Python problems whose one answer passes all hidden unit tests
  (pass@1, greedy decoding).
- **IFEval replies that never end** – replies that run into the 1,280-token limit without the end-of-turn token
  `<|im_end|>`. A high value means the model has not learned when to stop.

## How to read differences

With one standard error of about 2 points on IFEval and about 4 points on HumanEval, differences smaller than
roughly twice that (4 and 8 points) can be noise. Each configuration was trained once (one seed).

## How the numbers were produced

- lm-evaluation-harness, `hf` backend, model in bf16, 0-shot, greedy decoding, batch size 32
  ([evals/run_lm_eval.sh](../../evals/run_lm_eval.sh)).
- Every prompt is wrapped in the model's chat template, which ends with the empty think block
  `<think>\n\n</think>\n\n`, the same format the models were trained on.
- IFEval stops at `<|im_end|>` or 1,280 new tokens; HumanEval at `<|im_end|>`, a new top-level
  `def`/`class`/`if`/`print`/comment, or 1,024 new tokens.
- Every prompt and reply is saved next to the scores (`samples_*.jsonl`), so any number can be checked by reading
  the outputs.

## Source files

- Base model (no SFT): `qwen3-0.6b-base-prepared/checkpoints__qwen3-0.6b-base-prepared/results_2026-11-23T20-45-27.407401.json`
- Own loop, epoch 1: `own-loop_epoch-1/checkpoints__own-loop__epoch-1/results_2026-11-23T20-45-18.981040.json`
- Own loop, epoch 3: `own-loop_epoch-3/checkpoints__own-loop__epoch-3/results_2026-11-23T20-41-29.886891.json`
- llm-foundry fp32, epoch 1: `foundry-fp32_checkpoint-169/checkpoints__foundry-fp32__checkpoint-169/results_2026-11-23T20-42-30.673236.json`
- llm-foundry fp32, epoch 3: `foundry-fp32_final/checkpoints__foundry-fp32__final/results_2026-11-23T20-39-45.939459.json`
- llm-foundry pure bf16, epoch 3: `foundry-bf16_final/checkpoints__foundry-bf16__final/results_2026-11-23T20-42-42.957949.json`
