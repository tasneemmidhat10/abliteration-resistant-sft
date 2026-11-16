#!/bin/bash -l
# Build data/{train,val}.jsonl, stats.json, samples.jsonl, eval_indices.json.
# Usage (from the repo root):  sbatch scripts/slurm/build_data.sh
# Needs a Hugging Face login with the WildGuardMix terms accepted.
#SBATCH --partition=A40devel
#SBATCH --job-name=arft-data
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --gpus=1
#SBATCH --output=logs/build-data-%j.out
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/abliteration-resistant-sft}"
source scripts/activate.sh
python -m arft.data
