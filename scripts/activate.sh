# Load Bender modules + the matching project venv. SOURCE it, don't execute:
#
#   source scripts/activate.sh              # login node / A40 node -> Intel stack
#   source scripts/activate.sh amd          # force AMD stack (.venv_amd)
#
# Inside a SLURM job the stack is picked from the partition automatically.

_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ -n "${1:-}" ]] && export LLM_FOUNDRY_STACK="$1"

source "$_repo/scripts/slurm/modules.sh" || return 1

_venv="$_repo/.venv_${LLM_FOUNDRY_STACK}"
if [[ ! -d "$_venv" ]]; then
    echo "No venv at $_venv - build it with: sbatch scripts/slurm/create_venv_bender.sh" >&2
    return 1
fi
source "$_venv/bin/activate"
echo "Activated $_venv ($(python --version))"
unset _repo _venv
