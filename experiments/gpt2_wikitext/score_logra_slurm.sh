#!/bin/bash
#SBATCH --gres=gpu:A40:1
#SBATCH --array=0-49  # 50 models with seeds 0-49
#SBATCH --job-name=gpt2_wikitext_score_logra
#SBATCH --output=logs/score_logra_%A_%a.out
#SBATCH --error=logs/score_logra_%A_%a.err
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

echo "job is starting on `hostname`"
echo "Running LoGra scoring for seed: ${SLURM_ARRAY_TASK_ID}"

# Force use GPU 1 (index 1)
#export CUDA_VISIBLE_DEVICES=1

# PyTorch memory management
export PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0

# Create logs directory if it doesn't exist
mkdir -p logs

SEED=${SLURM_ARRAY_TASK_ID:-0}

# Change to results directory to save score files there
cd results

python ../score_logra.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --model_name_or_path openai-community/gpt2 \
    --output_dir ../checkpoints \
    --block_size 512 \
    --seed ${SEED}

