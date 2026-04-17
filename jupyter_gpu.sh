#!/bin/bash
# ==============================================================================
# Jupyter GPU Launcher — adapted for zhang.yicheng on Discovery
#
# Usage:
#   ./jupyter_gpu.sh                     # defaults: H200, 2h, port 8888
#   ./jupyter_gpu.sh --port=9999
#   ./jupyter_gpu.sh --time=04:00:00
#
# Then on LOCAL machine:
#   ssh -N -L <port>:<node>:<port> discovery
# Open http://localhost:<port>
# ==============================================================================

set -euo pipefail

PARTITION="gpu"
GPU_TYPE="h200"
NUM_GPUS="1"
TIME="02:00:00"
PORT="8888"

for arg in "$@"; do
    case "$arg" in
        --partition=*) PARTITION="${arg#*=}" ;;
        --gpu-type=*)  GPU_TYPE="${arg#*=}" ;;
        --num-gpus=*)  NUM_GPUS="${arg#*=}" ;;
        --time=*)      TIME="${arg#*=}" ;;
        --port=*)      PORT="${arg#*=}" ;;
        --help|-h)
            echo "Usage: $0 [--partition=X] [--gpu-type=X] [--num-gpus=N] [--time=HH:MM:SS] [--port=PORT]"
            exit 0 ;;
        *) echo "Unknown option: $arg"; exit 1 ;;
    esac
done

# --- Your paths ---
CONDA_ENV="gemma4_env"
WORK_DIR="/scratch/zhang.yicheng/llm_ft/neural_mechanics_v7"
HF_CACHE="/scratch/zhang.yicheng/gemma4/hf_cache"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "============================================================"
    echo " Jupyter GPU Launcher (zhang.yicheng)"
    echo "============================================================"
    echo " Partition: $PARTITION | GPU: ${GPU_TYPE} x${NUM_GPUS}"
    echo " Time: $TIME | Port: $PORT"
    echo " Conda env: $CONDA_ENV"
    echo " Work dir:  $WORK_DIR"
    echo "============================================================"
    echo ""
    echo "Requesting GPU allocation..."

    SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"

    exec salloc \
        --partition="$PARTITION" \
        --nodes=1 \
        --gres="gpu:${GPU_TYPE}:${NUM_GPUS}" \
        --mem=64G \
        --cpus-per-task=8 \
        --time="$TIME" \
        srun --pty bash -c "PORT=$PORT $SCRIPT_PATH --partition=$PARTITION --gpu-type=$GPU_TYPE --num-gpus=$NUM_GPUS --time=$TIME --port=$PORT"
fi

NODE="$(hostname)"

echo ""
echo "============================================================"
echo " Jupyter running on node: $NODE"
echo " Job ID: $SLURM_JOB_ID"
echo ""
echo " >>> LOCAL machine:"
echo "     ssh -N -L ${PORT}:${NODE}:${PORT} explorer"
echo ""
echo " >>> Browser:"
echo "     http://localhost:${PORT}"
echo "============================================================"
echo ""

# --- Activate conda env ---
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export HF_HOME="$HF_CACHE"
cd "$WORK_DIR"

echo "Conda env: $CONDA_DEFAULT_ENV"
echo "Working dir: $(pwd)"
echo ""

exec jupyter notebook \
    --no-browser \
    --port="$PORT" \
    --ip=0.0.0.0 \
    --IdentityProvider.token='' \
    --PasswordIdentityProvider.hashed_password=''