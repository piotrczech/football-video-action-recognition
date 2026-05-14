#!/bin/bash
#SBATCH -A hpc-piotrczech-1774303727
#SBATCH -p lem-cpu-short
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=16gb
#SBATCH --time=01:30:00
#SBATCH --job-name=football-setup-env
#SBATCH --output=/lustre/pd03/hpc-piotrczech-1774303727/football-video-action-recognition/logs/%x_%j.out
#SBATCH --error=/lustre/pd03/hpc-piotrczech-1774303727/football-video-action-recognition/logs/%x_%j.err

set -euo pipefail
umask 0002

source /usr/local/sbin/modules.sh
module purge
module load Python/3.12.3-GCCcore-13.3.0

PD="/lustre/pd03/hpc-piotrczech-1774303727"
PROJECT_SHARED="$PD/football-video-action-recognition"

# Submit this script from the repository root:
#   sbatch scripts/slurm/setup_env.sbatch
REPO="${REPO:-$SLURM_SUBMIT_DIR}"
VENV="${VENV:-$PROJECT_SHARED/venvs/football-$USER}"

mkdir -p "$PROJECT_SHARED/logs"
mkdir -p "$PROJECT_SHARED/venvs"
mkdir -p "$PROJECT_SHARED/cache/$USER"

for dir in "$PROJECT_SHARED/logs" "$PROJECT_SHARED/venvs" "$PROJECT_SHARED/cache" "$PROJECT_SHARED/cache/$USER"; do
  chmod g+rwx "$dir" 2>/dev/null || true
  chmod g+s "$dir" 2>/dev/null || true
done

export XDG_CACHE_HOME="$PROJECT_SHARED/cache/$USER"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export YOLO_CONFIG_DIR="$XDG_CACHE_HOME/ultralytics"

cd "$REPO"

echo "=== Setup env ==="
echo "User: $USER"
echo "Host: $(hostname)"
echo "Repo: $(pwd)"
echo "Venv: $VENV"
echo "Python module:"
python3 --version

python3 -m venv "$VENV"
source "$VENV/bin/activate"

echo "VIRTUAL_ENV=$VIRTUAL_ENV"
which python
python --version

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e .

python - <<'PY'
import sys

print(sys.version)

import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
PY

echo "=== Done ==="
date
