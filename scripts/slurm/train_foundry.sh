#!/bin/bash -l
# SFT with llm-foundry's TRL script, using the same data, tokens and hyperparameters as our own loop.
# Needs once before:  cd src && python -m SFT.prepare_foundry
# Usage (from the repo root):
#   sbatch scripts/slurm/train_foundry.sh fp32   # fp32 weights (closest to our mixed-precision loop)
#   sbatch scripts/slurm/train_foundry.sh bf16   # llm-foundry's --bf16: weights stored in bf16 (pure bf16)
# Output: checkpoints/foundry-<precision>/{checkpoint-169, checkpoint-338, final}
#SBATCH --partition=A40short
#SBATCH --job-name=sft-foundry
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=logs/sft-foundry-%j.out
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/abliteration-resistant-sft}"
source scripts/activate.sh
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PRECISION=${1:-fp32}                                  # first argument to sbatch, default fp32
BF16_FLAG=""; [[ "$PRECISION" == "bf16" ]] && BF16_FLAG="--bf16"

python ~/llm-foundry/alignment/sft_trainer.py \
    --model_name_or_path checkpoints/qwen3-0.6b-base-prepared \
    --prebuilt_dataset_dir checkpoints/foundry-data \
    --checkpoint_dir checkpoints/foundry-$PRECISION \
    --max_length 2048 \
    --assistant_only_loss \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 3 \
    --learning_rate 2e-5 \
    --lr_scheduler_type cosine \
    --warmup_steps 0.05 \
    --weight_decay 0.01 \
    --adam_beta2 0.999 \
    --max_grad_norm 1.0 \
    --attn_implementation sdpa \
    --logging_steps 10 \
    --eval_steps 169 \
    --save_steps 169 \
    --seed 0 \
    --num_proc 8 \
    --report_to none \
    $BF16_FLAG
