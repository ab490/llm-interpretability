#!/bin/bash
cd "$(dirname "$0")/.."

# Run the Python script
python scripts/run_ablations.py \
  --model_name Llama-3.1-8B-Instruct \
  --permutations 300 \
  --max_heads_abl 50 \
  --head_type induction \
  --to_abl 0-1-5-10-20-50 \
  --mode zero \
  --layer_abl full \
  --tokens_path data/llama-1k-tokens.pkl \
  --induction_scores_csv data/induction_scores_Llama-3.1-8B-Instruct_sorted.csv \
  --output_dir outputs/ablations/Llama-3.1-8B-Instruct \
  --n_devices 2 \
  --dtype bfloat16