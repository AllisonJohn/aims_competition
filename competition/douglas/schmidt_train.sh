#!/bin/bash
# Submit from the Schmidt cluster after copying/cloning this repo there:
#   sbatch competition/douglas/schmidt_train.sh

#SBATCH --job-name=douglas-train
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --partition=cs321m
#SBATCH --qos=cs321m
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --mem=64G

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"

python3 -m venv .venv-douglas
source .venv-douglas/bin/activate
python -m pip install --upgrade pip
python -m pip install -r competition/douglas/requirements.txt

export HF_HOME="${HF_HOME:-$REPO_ROOT/.hf_cache}"
mkdir -p "$HF_HOME"

python competition/douglas/training.py
