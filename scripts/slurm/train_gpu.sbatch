#!/bin/bash
#SBATCH -A hpc-piotrczech-1774303727
#SBATCH -p lem-gpu-normal
#SBATCH --gres=gpu:hopper:1
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=128gb
#SBATCH --time=12:00:00
#SBATCH --job-name=football-train
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
#   sbatch scripts/slurm/train_gpu.sbatch
REPO="${REPO:-$SLURM_SUBMIT_DIR}"

VENV="${VENV:-$PROJECT_SHARED/venvs/football-$USER}"

MODEL="${MODEL:-yolo}"
DATASET_VARIANT="${DATASET_VARIANT:-base}"
PROFILE="${PROFILE:-quick}"
RUN_NAME="${RUN_NAME:-${MODEL}--${DATASET_VARIANT}--${PROFILE}--slurm-${SLURM_JOB_ID}}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "$PROJECT_SHARED/logs"
mkdir -p "$PROJECT_SHARED/cache/$USER"

export XDG_CACHE_HOME="$PROJECT_SHARED/cache/$USER"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export YOLO_CONFIG_DIR="$XDG_CACHE_HOME/ultralytics"

cd "$REPO"
source "$VENV/bin/activate"

echo "=== SLURM ==="
echo "Job ID: $SLURM_JOB_ID"
echo "User: $USER"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Partition: ${SLURM_JOB_PARTITION:-unknown}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"

echo "=== Project ==="
echo "Repo: $(pwd)"
echo "Data: $(readlink -f data)"
echo "Models: $(readlink -f models)"
echo "Outputs: $(readlink -f outputs)"
echo "Venv: $VIRTUAL_ENV"

echo "=== Config ==="
echo "MODEL=$MODEL"
echo "DATASET_VARIANT=$DATASET_VARIANT"
echo "PROFILE=$PROFILE"
echo "RUN_NAME=$RUN_NAME"
echo "EXTRA_ARGS=$EXTRA_ARGS"

echo "=== GPU ==="
nvidia-smi

python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("cuda device:", torch.cuda.get_device_name(0))
else:
    raise SystemExit("CUDA is not available, stopping before training.")
PY

echo "=== Dataset check ==="
test -f "data/ready/${DATASET_VARIANT}/train/_annotations.coco.json"

echo "=== Training ==="
python scripts/train.py \
  --model "$MODEL" \
  --dataset-variant "$DATASET_VARIANT" \
  --profile "$PROFILE" \
  --name "$RUN_NAME" \
  $EXTRA_ARGS

echo "=== Done ==="
date