#!/bin/bash -l
#SBATCH --partition=A40devel
#SBATCH --job-name=sft-data
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --gpus=1
#SBATCH --output=logs/build-data-%j.out
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/abliteration-resistant-sft}"
source scripts/activate.sh
python -m SFT.utils
