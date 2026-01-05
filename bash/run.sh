#!/bin/bash
#SBATCH --job-name=lens     # Job name
#SBATCH --output=../logs/lens_%j.out  # Path for the standard output file
#SBATCH --error=../logs/lens_%j.err   # Path for the error file

#SBATCH --mail-type=ALL                 # Email notification for all states
#SBATCH --mail-user=anobajaj@iu.edu     # Email address for notifications
#SBATCH -p gpu-debug
#SBATCH --gpus-per-node=2
#SBATCH -A r00117
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4


# Load required modules
# module load python/gpu/3.11.5

#activate venv
# source /N/scratch/anobajaj/v_envs/lens/bin/activate

# Run the Python script
python ../scripts/run_ablations.py \
  --model_name Llama-3.1-8B-Instruct \
  --permutations 300 \
  --max_heads_abl 50 \
  --head_type induction \
  --to_abl 0-1-5-10-20-50 \
  --mode zero \
  --layer_abl full \
  --tokens_path ../data/llama-1k-tokens.pkl \
  --induction_scores_csv ../data/induction_scores_Llama-3.1-8B-Instruct_sorted.csv \
  --output_dir ../outputs/ablations/Llama-3.1-8B-Instruct \
  --n_devices 2 \
  --dtype bfloat16