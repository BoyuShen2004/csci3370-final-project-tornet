#!/bin/bash
#SBATCH --job-name=dinov3-inference
#SBATCH --output=/projects/weilab/shenb/csci3370/code_submission/slurm_scripts/logs/%x_%j.out
#SBATCH --error=/projects/weilab/shenb/csci3370/code_submission/slurm_scripts/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --partition=short
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=shenb@bc.edu

echo ">>> Starting DINOv3 inference..."
echo ">>> Job ID: $SLURM_JOB_ID"
echo ">>> Start time: $(date)"

# Set up environment
source /projects/weilab/shenb/miniconda3/etc/profile.d/conda.sh
conda activate tornet2

# Set working directory
cd /projects/weilab/shenb/csci3370/code_submission/dinov3_vit

# Set environment variables
export TORNET_ROOT=/projects/weilab/shenb/csci3370/data
export KERAS_BACKEND=torch

# Optional override for finetune checkpoint via first arg
export DINOV3_FINETUNE_CKPT="${1:-/projects/weilab/shenb/csci3370/code_submission/outputs/dino_finetune_latest/checkpoints/checkpoint_best.pth}"

echo ">>> TORNET_ROOT: $TORNET_ROOT"
echo ">>> MODEL_CKPT:  $DINOV3_FINETUNE_CKPT"
echo ">>> GPU info:"
nvidia-smi || true
echo "=========================================="

python inference/run_inference.py --config inference/config.yaml

echo ">>> Inference completed at $(date)"


