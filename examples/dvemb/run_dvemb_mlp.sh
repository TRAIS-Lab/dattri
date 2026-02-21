#!/bin/bash
#SBATCH --job-name=dvemb_mlp_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=train_logs/dvemb_mlp_%j.out
#SBATCH --error=train_logs/dvemb_mlp_%j.err

# Create output directory if it doesn't exist
mkdir -p train_logs

echo "========================================"
echo "Job start: $(date)"
echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Memory: ${SLURM_MEM_PER_NODE} MB"
echo "========================================"
echo ""

# Activate conda environment
source activate dattri 2>/dev/null || conda activate dattri 2>/dev/null || true

echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "torch version: $(python -c 'import torch; print(torch.__version__)')"
echo ""

# Run the training script
cd /storage/ice1/5/4/wsun372/dattri
python examples/dvemb/dvemb_mlp.py

echo ""
echo "Job end: $(date)"
