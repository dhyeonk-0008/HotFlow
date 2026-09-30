#!/bin/bash
#SBATCH --job-name=hotflow_cw05
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs_b/slurm_%j.out
#SBATCH --error=logs_b/slurm_%j.err

cd "${HOTFLOW_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${HOTFLOW_ENV:-hotflow}"

mkdir -p logs_b

# SLURM may assign a physical GPU ID (e.g. 7) that doesn't match the
# cgroup-visible devices (e.g. 0,1). Reset to use the first visible GPU.
export CUDA_VISIBLE_DEVICES=0

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
nvidia-smi -L 2>&1 || echo "nvidia-smi failed"

CUDA_LAUNCH_BLOCKING=1 python hotflow/train_b.py \
    --config hotflow/configs/train_b_cw05.yaml \
    --device cuda:0 \
    --num_workers 4 \
    --name hotflow_b_cw05
