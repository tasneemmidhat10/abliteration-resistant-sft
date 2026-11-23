#!/bin/bash -l
# Validation loss per subset for all checkpoints (src/SFT/compare_val.py).
# Usage (from the repo root):  sbatch scripts/slurm/compare_val.sh
#SBATCH --partition=A40short
#SBATCH --job-name=compare-val
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/compare-val-%j.out
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/abliteration-resistant-sft}"
source scripts/activate.sh
export HF_HUB_OFFLINE=1
cd src && python -m SFT.compare_val
