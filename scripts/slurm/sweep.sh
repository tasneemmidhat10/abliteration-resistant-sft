#!/bin/bash -l
# Attack + evaluate one model (handbook Steps 2/4).
# Usage (from the repo root):
#   sbatch scripts/slurm/sweep.sh base Qwen/Qwen2.5-1.5B-Instruct
#   sbatch --partition=A40devel --time=1:00:00 scripts/slurm/sweep.sh smoke Qwen/Qwen2.5-0.5B-Instruct --smoke
#SBATCH --partition=A40short
#SBATCH --job-name=arft-sweep
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --gpus=1
#SBATCH --output=logs/sweep-%x-%j.out
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/abliteration-resistant-sft}"
source scripts/activate.sh
export TOKENIZERS_PARALLELISM=false
python -m arft.sweep --name "$1" --model "$2" "${@:3}"
