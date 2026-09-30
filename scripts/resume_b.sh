#!/bin/bash
#SBATCH --job-name=hotflow_b
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs_b/slurm_%j.out
#SBATCH --error=logs_b/slurm_%j.err

cd "${HOTFLOW_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${HOTFLOW_ENV:-hotflow}"

mkdir -p logs_b

# Find the latest checkpoint automatically
CKPT_DIR="${CKPT_DIR:?set CKPT_DIR=./logs_b/<run>/checkpoints}"
LATEST_CKPT=$(ls -t ${CKPT_DIR}/*.pt 2>/dev/null | head -1)

if [ -z "$LATEST_CKPT" ]; then
    echo "No checkpoint found in ${CKPT_DIR}"
    exit 1
fi

echo "Resuming from: ${LATEST_CKPT}"

CUDA_LAUNCH_BLOCKING=1 python hotflow/train_b.py \
    --config hotflow/configs/train_b.yaml \
    --device cuda:0 \
    --num_workers 4 \
    --name hotflow_b \
    --resume ${LATEST_CKPT}
