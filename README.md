# abliteration-resistant-sft

Can fine-tuning entangle a model's refusal representation with its capability representation, so that
removing refusal by *abliteration* also destroys capability? This repository holds the proof of concept
(Qwen2.5-1.5B-Instruct + LoRA, with auxiliary policy and capability heads on one hidden layer).

- Research idea, study design and roadmap: [docs/HANDBOOK.md](docs/HANDBOOK.md)
- Training data: [docs/DATA.md](docs/DATA.md)

## Setup on Bender (HPC cluster, Bonn)

```bash
git clone <this repo> ~/abliteration-resistant-sft && cd ~/abliteration-resistant-sft
mkdir -p logs
sbatch scripts/slurm/create_venv_bender.sh           # A40 nodes  -> .venv_intel  (~5 min)
# sbatch --partition=A100devel scripts/slurm/create_venv_bender.sh   # A100 nodes -> .venv_amd
```

Then, in every new shell or job script:

```bash
source scripts/activate.sh        # loads CUDA 12.4 + Python 3.12 modules and the matching venv
```

Bender's newest CUDA module is 12.4, which caps PyTorch at 2.6.0. The modules and venv setup are adapted
from [llm-foundry](https://github.com/Polygl0t/llm-foundry). Exact package versions are in `requirements.lock`.

One-time Hugging Face setup (gated datasets):

```bash
hf auth login     # read token; accept the terms of allenai/wildguardmix and walledai/AdvBench on the website
```

## Setup on a laptop

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch                 # the build for your platform, see pytorch.org
pip install -r requirements.txt && pip install -e . --no-deps
```

## Reproducing

| Step | Command | Output |
|---|---|---|
| 1. Data | `sbatch scripts/slurm/build_data.sh` | `data/{train,val}.jsonl`, `data/stats.json` |
| Tests | `pytest` | |

## Layout

```
configs/        training conditions: baseline / coupled / control
src/arft/       the package (data, training, directions, interventions)
training/       evaluation, sweep and plotting scripts
scripts/        activate.sh, run_poc.sh, slurm/ job files
tests/          pytest
data/           only stats.json, samples.jsonl, eval_indices.json are committed
docs/           HANDBOOK.md, DATA.md, POC_REPORT.md
```
