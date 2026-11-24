#!/bin/bash -l
# Evaluate one chat model with lm-evaluation-harness on IFEval and HumanEval (instruct).
# Usage (from the repo root), one job per model:
#   sbatch evals/run_lm_eval.sh checkpoints/own-loop/epoch-1
# Results: results/lm-eval/<model-name>/ (scores JSON + every generated sample)
#
# The model folder's tokenizer must carry our chat template (think block in the prompt, eos <|im_end|>):
# true for own-loop/epoch-*, foundry-*/... and checkpoints/qwen3-0.6b-base-prepared (the untrained base).
#SBATCH --partition=A40short
#SBATCH --job-name=lm-eval
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=logs/lm-eval-%j.out
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/abliteration-resistant-sft}"
source scripts/activate.sh
export HF_ALLOW_CODE_EVAL=1                      # HumanEval executes the generated code to test it

MODEL=$1                                         # path to the model folder
NAME=$(echo "${MODEL#checkpoints/}" | tr '/' '_')   # checkpoints/own-loop/epoch-1 -> own-loop_epoch-1

# --apply_chat_template : wrap each prompt in the model's chat template (it is a chat model now)
# no think_end_token    : our template already closes the think block, and that option breaks HumanEval
#                         (it lstrips every output; see llm-foundry/evals/README.md)
# --log_samples         : save every prompt + output, so we can check the scores by reading them
lm_eval run --model hf \
    --model_args pretrained=$MODEL,dtype=bfloat16 \
    --tasks ifeval,humaneval_instruct \
    --apply_chat_template \
    --batch_size 32 \
    --confirm_run_unsafe_code \
    --log_samples \
    --output_path results/lm-eval/$NAME
