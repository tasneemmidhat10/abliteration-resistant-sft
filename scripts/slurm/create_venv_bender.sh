#!/bin/bash -l
#############################################
# One-time setup: build the project venv on a Bender GPU node.
# Adapted from llm-foundry/alignment/slurm/create_venv_bender.sh.
#
# Usage (from the repo root):
#   sbatch scripts/slurm/create_venv_bender.sh                          # A40 -> .venv_intel
#   sbatch --partition=A100devel scripts/slurm/create_venv_bender.sh    # A100 -> .venv_amd
#
# Bender has two software stacks (Intel on A40 nodes, AMD on A100 nodes);
# scripts/slurm/modules.sh picks the right one from the partition. Their
# compiled libraries can differ, so each stack gets its own venv.
#
# Bender's newest CUDA module is 12.4, which caps PyTorch at 2.6.0.
#############################################
#SBATCH --partition=A40devel
#SBATCH --job-name=arft-venv
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --gpus=1
#SBATCH --output=logs/create-venv-%j.out

set -euo pipefail

repo="${SLURM_SUBMIT_DIR:-$HOME/abliteration-resistant-sft}"
cd "$repo"
mkdir -p logs
ulimit -c 0

echo "# [$SLURM_JOB_ID] $(date)  host=$(hostname)  partition=$SLURM_JOB_PARTITION"

#############################################
# Modules (CUDA 12.4 + Python 3.12.3 for this node's stack)
#############################################
# modules.sh references unset variables, so relax -u while sourcing it.
set +u
source scripts/slurm/modules.sh
set -u
stack="$LLM_FOUNDRY_STACK"           # exported by modules.sh: intel | amd
venv_dir="$repo/.venv_$stack"

echo "  Python: $(which python3) - $(python3 --version)"
echo "  nvcc:   $(nvcc --version | tail -n1)"
echo "  venv:   $venv_dir"

#############################################
# Create venv
#############################################
if [[ -d "$venv_dir" ]]; then
    echo "ERROR: $venv_dir already exists. Delete it first to rebuild." >&2
    exit 1
fi
python3 -m venv "$venv_dir"
source "$venv_dir/bin/activate"
pip install --upgrade pip
pip install uv

echo "===== PyTorch 2.6.0+cu124 ====="
uv pip install --no-cache \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Keep torch fixed for every install that follows.
constraints="$(mktemp)"
echo "torch==2.6.0+cu124" > "$constraints"

echo "===== flash-attn 2.8.3 (prebuilt wheel: cu124, torch2.6, py312) ====="
FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE uv pip install --no-cache \
    https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3+cu124torch2.6-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl

echo "===== Project requirements ====="
uv pip install --no-cache -c "$constraints" -r requirements.txt
rm -f "$constraints"

# Record exactly what got installed, for reproducibility.
uv pip freeze > "logs/venv_${stack}_freeze.txt"

#############################################
# Verification
#############################################
echo "===== Verifying ====="
python3 - <<'PY'
from importlib.metadata import version
import torch

for pkg in ["torch", "transformers", "datasets", "accelerate", "trl", "peft",
            "lm_eval", "evaluate", "flash-attn", "wandb", "pandas", "matplotlib"]:
    try:
        print(f"  {pkg}=={version(pkg)}")
    except Exception as e:
        print(f"  {pkg}: NOT FOUND ({e})")

print(f"  torch CUDA runtime: {torch.version.cuda}")
print(f"  CUDA available: {torch.cuda.is_available()}")
print(f"  GPU: {torch.cuda.get_device_name(0)}")
x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
print(f"  bf16 matmul OK: {(x @ x).shape}")
from flash_attn import flash_attn_func
print("  flash_attn import OK")
PY

echo "===== Done: $(date) ====="
echo "Activate with: source scripts/activate.sh"
