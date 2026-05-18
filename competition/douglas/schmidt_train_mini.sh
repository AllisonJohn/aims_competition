#!/bin/bash
# Submit from the Schmidt cluster after copying/cloning this repo there:
#   sbatch competition/douglas/schmidt_train_mini.sh

#SBATCH --job-name=douglas-mini
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --partition=cs321m
#SBATCH --qos=cs321m
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --mem=64G

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"

if "$PYTHON_BIN" -m venv .venv-douglas; then
    source .venv-douglas/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -r competition/douglas/requirements.txt
    RUN_PYTHON="python"
else
    echo "venv creation failed; falling back to user-site pip installs"
    "$PYTHON_BIN" -m pip install --user --upgrade pip
    "$PYTHON_BIN" -m pip install --user -r competition/douglas/requirements.txt
    export PATH="$HOME/.local/bin:$PATH"
    RUN_PYTHON="$PYTHON_BIN"
fi

export HF_HOME="${HF_HOME:-$REPO_ROOT/.hf_cache}"
mkdir -p "$HF_HOME"

"$RUN_PYTHON" competition/douglas/training_mini.py
