#!/bin/bash
#SBATCH --job-name=dinov3-finetune
#SBATCH --output=/projects/weilab/shenb/csci3370/code_submission/slurm_scripts/logs/%x_%j.out
#SBATCH --error=/projects/weilab/shenb/csci3370/code_submission/slurm_scripts/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64GB
#SBATCH --time=120:00:00
#SBATCH --partition=long
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=shenb@bc.edu

echo ">>> Starting DINOv3 fine-tuning..."
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

# Optional override for supervised checkpoint via first arg
export DINOV3_SUPERVISED_CKPT="${1:-/projects/weilab/shenb/csci3370/code_submission/outputs/dino_supervised_latest/checkpoints/checkpoint_best.pth}"

echo ">>> TORNET_ROOT: $TORNET_ROOT"
echo ">>> ENCODER:     $DINOV3_SUPERVISED_CKPT"
echo ">>> GPU info:"
nvidia-smi || true
echo "=========================================="

python finetune/train_finetune.py --config finetune/config.yaml

echo ">>> Fine-tuning completed at $(date)"


