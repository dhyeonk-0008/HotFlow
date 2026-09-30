#!/bin/bash
# Unified graph architecture training — 4 GPU DDP
# Effective batch size: 8 (per-GPU) × 4 (GPUs) × 1 (accum) = 32
#
# Usage:
#   nohup bash scripts/train_unified_ddp.sh > logs_unified/nohup.out 2>&1 &

cd "$(dirname "$0")/.."

# conda activate in non-interactive shell
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${HOTFLOW_ENV:-hotflow}"

mkdir -p logs_unified

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NCCL_P2P_DISABLE=1
export PYTHONPATH=$PWD

torchrun --nproc_per_node=4 hotflow/train_unified.py \
    --config hotflow/configs/train_unified.yaml \
    --num_workers 4 \
    --name hotflow_unified
