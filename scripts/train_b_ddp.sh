#!/bin/bash
#SBATCH --job-name=hotflow_b_ddp
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs_b/slurm_%j.out
#SBATCH --error=logs_b/slurm_%j.err

cd "${HOTFLOW_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${HOTFLOW_ENV:-hotflow}"

mkdir -p logs_b

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NCCL_P2P_DISABLE=1

torchrun --nproc_per_node=4 hotflow/train_b.py \
    --config hotflow/configs/train_b.yaml \
    --num_workers 4 \
    --name hotflow_b
