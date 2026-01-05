# attention-ablation-analysis

This repo contains a set of experiments analyzing attention head ablations in large language models.  
It focuses on how ablating different heads affects token prediction probabilities across lags.


## Steps to run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Hugging Face token
Some models require gated access.
```bash
export HUGGINGFACE_HUB_TOKEN=hf_your_token_here
```

### 3. Run the experiment
From the repo root:
```bash
bash bash/run_ablations.sh
```

Outputs are saved under ```outputs/ablations/<model_name>/```


#### Arguments that can be passed

The main script (`run_ablations.py`) supports the following key arguments:

- `--model_name`  
  Model to run (e.g. `Llama-3.1-8B-Instruct`, `Mistral-7B-Instruct-v0.1`)

- `--permutations`  
  Number of random permutations per setting (default: `300`)

- `--max_heads_abl`  
  Maximum number of heads considered for ablation (default: `50`)

- `--head_type`  
  Which heads to ablate: `induction` or `random`

- `--to_abl`  
  Number of heads ablated at each step (e.g. `0-1-5-10-20-50`)  
  `0` corresponds to the no ablation baseline.

- `--mode`  
  Ablation mode: `zero` or `mean`
  Controls how attention scores are ablated for selected heads:
  - `zero`: Sets all attention scores for the ablated head to −∞, effectively removing the head entirely.
  - `mean`: Replaces all attention scores for the ablated head with their mean value.

- `--layer_abl`  
  Layers to consider: `full`, `top`, or `bottom`
  - `full`: Consider heads from all layers in the model.
  - `top`: Consider heads only from the lower half of layers (closer to the input).
  - `bottom`: Consider heads only from the upper half of layers (closer to the output).

- `--tokens_path`  
  Path to token pickle used for prompt construction

- `--induction_scores_csv`  
  CSV containing precomputed induction head scores

- `--output_dir`  
  Directory where results are saved


The current repo is set to be used for Llama-3.1-8B-Instruct, with the corresponding token set and induction-head scores CSV already provided.
To test other models, provide a sorted induction-head scores CSV in the same format as the existing Llama file and update the relevant arguments and add the model name to the ```MODEL_MAPPING`` dictionary if it is not already included.