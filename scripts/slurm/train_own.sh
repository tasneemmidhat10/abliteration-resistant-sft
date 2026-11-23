#!/bin/bash -l
# Full SFT run with our own training loop (src/SFT/train.py, settings in CONFIG).
# Usage (from the repo root):  sbatch scripts/slurm/train_own.sh
# Output: checkpoints/own-loop/{log.jsonl, epoch-1, epoch-2, epoch-3}
#SBATCH --partition=A40short
#SBATCH --job-name=sft-own
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/sft-own-%j.out
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/abliteration-resistant-sft}"
source scripts/activate.sh
export HF_HUB_OFFLINE=1          # the model is already in the cache; don't depend on internet on the node
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # less memory lost to fragmentation
cd src && python -m SFT.train
