#!/bin/bash
#SBATCH --job-name=hotflow_eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-eval-%j.out
#SBATCH --error=slurm-eval-%j.err

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${HOTFLOW_ENV:-hotflow}"

cd "${HOTFLOW_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

# Point this at your trained Approach B checkpoint.
APPROACH_B_CKPT="${APPROACH_B_CKPT:?set APPROACH_B_CKPT=logs_b/<run>/checkpoints/<iter>.pt}"

python -m hotflow.eval_compare \
    --config hotflow/configs/train_b.yaml \
    --pepflow_ckpt PepFlowww/model2.pt \
    --approach_b_ckpt "$APPROACH_B_CKPT" \
    --num_samples 200 \
    --num_steps 100 \
    --guidance_scales 1.0 \
    --device cuda:0 \
    --outdir results/eval_test_relax \
    --rosetta_score_gt