#!/bin/bash
#SBATCH --job-name=hotspot_v2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs_hotspot/slurm_%j.out
#SBATCH --error=logs_hotspot/slurm_%j.err

cd "${HOTFLOW_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${HOTFLOW_ENV:-hotflow}"

mkdir -p logs_hotspot

# Plain V2: spatial-bias attention + dihedrals + pos emb
#           + rotation aug + masking + length filter + best-val save
python -m hotflow.train_hotspot \
    --config hotflow/configs/train_hotspot_v2.yaml \
    --logdir logs_hotspot \
    --device cuda:0 \
    --batch_size 128 \
    --num_workers 4
