#!/usr/bin/env python
# coding: utf-8

import argparse
import functools
import json
import os
import pickle
import random
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer, utils


MODEL_MAPPING = {
    "Llama-3.2-1B-Instruct": "meta-llama/Llama-3.2-1B-Instruct",
    "Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "Llama-3.1-8B": "meta-llama/Llama-3.1-8B",
    "gemma-2-9b-it": "google/gemma-2-9b-it",
    "gemma-2-9b": "google/gemma-2-9b",
    "Mistral-7B-Instruct-v0.1": "mistralai/Mistral-7B-Instruct-v0.1",
    "Mistral-7B-v0.1": "mistralai/Mistral-7B-v0.1",
    "Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
    "Qwen2.5-7B": "Qwen/Qwen2.5-7B",
}


def get_tokens(tokens_path: Path, n_unique: int = 500, seed: int = 42) -> np.ndarray:
    """Load the token set used to build prompts and return up to n_unique unique tokens."""
    rng = np.random.default_rng(seed)
    with tokens_path.open("rb") as f:
        tokens_nested = pickle.load(f)

    flattened = [tok for sublist in tokens_nested for tok in sublist]
    uniq = np.unique(flattened)

    if len(uniq) > n_unique:
        idx = rng.choice(len(uniq), size=n_unique, replace=False)
        uniq = uniq[idx]
    return uniq


def get_induction_heads_top(induction_scores_csv: Path, top_heads_num: int, num_layers: int, layer_abl: str,) -> list[tuple[int, int]]:
    """Return a list of (layer, head) pairs for top induction heads."""
    df = pd.read_csv(induction_scores_csv).sort_values(by="scores", ascending=False)

    num_layers_thresh = int(0.5 * num_layers)
    if layer_abl == "top":
        df = df[df["layer"] <= num_layers_thresh]
    elif layer_abl == "bottom":
        df = df[df["layer"] > num_layers_thresh]
    elif layer_abl != "full":
        raise ValueError(f"layer_abl must be one of: full/top/bottom. Got: {layer_abl}")

    top_df = df.head(top_heads_num)
    return list(zip(top_df["layer"].tolist(), top_df["head"].tolist()))


def prompt_design_prob(tokens: np.ndarray) -> torch.Tensor:
    """Construct the prompt used to probe induction like behavior."""
    permuted = np.random.permutation(tokens)
    mid = permuted[len(permuted) // 2]
    prompt_tokens = np.append(permuted, mid)
    return torch.tensor(prompt_tokens, dtype=torch.int64).unsqueeze(0)


def head_ablation_hook(attn_scores: torch.Tensor, hook, head: int, mode: str) -> torch.Tensor:
    """Ablate attention scores for a single head.

    mode="zero": set to -inf
    mode="mean": replace non -inf entries with their mean value
    """
    if mode == "mean":
        neg_inf_mask = torch.isneginf(attn_scores[:, head, :, :])
        vals = attn_scores[:, head, :, :][~neg_inf_mask]
        if vals.numel() > 0:
            mean_val = vals.mean()
            attn_scores[:, head, :, :][~neg_inf_mask] = mean_val
    else:
        attn_scores[:, head, :, :] = -torch.inf
    return attn_scores


def calculate_output_probabs(model: HookedTransformer, prompt_tokens: torch.Tensor, hooks):
    """Run the model with hooks and return probabilities for tokens in the prompt."""
    logits = model.run_with_hooks(prompt_tokens, return_type="logits", fwd_hooks=hooks)
    last_logits = logits[:, -1, :].float()
    probs = torch.nn.functional.softmax(last_logits, dim=-1).squeeze(0)  # (vocab,)

    prompt_token_ids = prompt_tokens.squeeze(0).tolist()
    return probs[prompt_token_ids].detach().cpu().numpy()  # (seq,)


def plot_and_save(lags, probabilities, out_png: Path, title: str, show: bool = False):
    """Plot lag vs probability curve and save it."""
    plt.figure(figsize=(8, 5))
    plt.plot(lags, probabilities)
    plt.axvline(x=0, color="black", linestyle="--", linewidth=0.8)
    plt.xlabel("Lag")
    plt.ylabel("Probability")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, bbox_inches="tight", dpi=300)
    if show:
        plt.show()
    plt.close()


def run_experiment(
    model_name: str,
    hf_token: str | None,
    permutations: int,
    max_heads_abl: int,
    head_type: str,
    to_abl: str,
    mode: str,
    layer_abl: str,
    tokens_path: Path,
    induction_scores_csv: Path,
    output_dir: Path,
    n_devices: int,
    dtype: str,
    seed: int,
    show: bool,
):
    """Main experiment runner."""
    if model_name not in MODEL_MAPPING:
        raise ValueError(f"Unsupported model_name: {model_name}. Supported: {list(MODEL_MAPPING.keys())}")

    if hf_token is None:
        raise EnvironmentError(
            "Missing HUGGINGFACE_HUB_TOKEN. Set it as an env var to load gated models."
        )
        
    if not tokens_path.exists():
        raise FileNotFoundError(f"tokens_path not found: {tokens_path}")
    if not induction_scores_csv.exists():
        raise FileNotFoundError(f"induction_scores_csv not found: {induction_scores_csv}")  

    np.random.seed(seed)
    random.seed(seed)

    model_full = MODEL_MAPPING[model_name]
    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16

    model = HookedTransformer.from_pretrained(
        model_full,
        use_auth_token=hf_token,
        n_devices=n_devices,
        dtype=torch_dtype,
    )

    tokens = get_tokens(tokens_path, n_unique=500, seed=seed)

    num_layers = model.cfg.n_layers
    num_heads = model.cfg.n_heads

    induction_heads = get_induction_heads_top(
        induction_scores_csv=induction_scores_csv,
        top_heads_num=max_heads_abl,
        num_layers=num_layers,
        layer_abl=layer_abl,
    )

    # build non-induction pool for random selection
    if layer_abl == "full":
        req_layers = range(0, num_layers)
    else:
        num_layers_thresh = int(0.5 * num_layers)
        req_layers = range(0, num_layers_thresh) if layer_abl == "top" else range(num_layers - num_layers_thresh, num_layers)

    nonind = [(l, h) for l in req_layers for h in range(num_heads) if (l, h) not in induction_heads]
    if len(nonind) < max_heads_abl:
        raise ValueError(f"Not enough non-induction heads to sample: need {max_heads_abl}, have {len(nonind)}")
    random_heads = random.sample(nonind, k=max_heads_abl)

    selected_heads = induction_heads if head_type == "induction" else random_heads

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "heads").mkdir(exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    (output_dir / "tables").mkdir(exist_ok=True)

    (output_dir / "heads" / f"{model_name}_{head_type}_{layer_abl}_{mode}.json").write_text(
        json.dumps({"head_type": head_type, "layer_abl": layer_abl, "mode": mode, "heads": selected_heads}, indent=2)
    )

    ablation_values = [int(v) for v in to_abl.split("-")]
    for num_abl in ablation_values:
        heads_req = selected_heads[:num_abl]

        hooks = []
        for (layer, head) in heads_req:
            hooks.append(
                (
                    utils.get_act_name("attn_scores", layer),
                    functools.partial(head_ablation_hook, head=head, mode=mode),
                )
            )

        prob_list = []
        for _ in tqdm(range(permutations), desc=f"{model_name} | {head_type} | ablating {num_abl}"):
            prompt_tokens = prompt_design_prob(tokens)
            prob_list.append(calculate_output_probabs(model, prompt_tokens, hooks))

        probabilities = np.mean(prob_list, axis=0)

        lags = np.arange(len(probabilities)) - (len(probabilities) // 2)

        fig_name = f"{model_name}_perm{permutations}_abl{num_abl}_{mode}_{head_type}_{layer_abl}.png"
        csv_name = f"{model_name}_perm{permutations}_abl{num_abl}_{mode}_{head_type}_{layer_abl}.csv"
        title = f"{model_name} | perms={permutations} | heads={num_abl} | {mode} | {head_type} | {layer_abl}"

        plot_and_save(lags, probabilities, output_dir / "figures" / fig_name, title, show=show)

        df_out = pd.DataFrame({"Lag": lags, "Mean Probability": probabilities})
        df_out.to_csv(output_dir / "tables" / csv_name, index=False)
        

def parse_args():
    """Parse command-line arguments for running the ablation experiment."""
    p = argparse.ArgumentParser(description="Attention-head ablation experiments (subset).")
    p.add_argument("--model_name", required=True, choices=MODEL_MAPPING.keys())
    p.add_argument("--permutations", type=int, default=300)
    p.add_argument("--max_heads_abl", type=int, default=50)
    p.add_argument("--head_type", choices=["induction", "random"], default="induction")
    p.add_argument("--to_abl", type=str, default="1-5-10-20-50")
    p.add_argument("--mode", choices=["zero", "mean"], default="zero")
    p.add_argument("--layer_abl", choices=["full", "top", "bottom"], default="full")

    p.add_argument("--tokens_path", type=Path, default=Path("data/tokens_1k.pkl"))
    p.add_argument("--induction_scores_csv", type=Path, required=True)

    p.add_argument("--output_dir", type=Path, default=Path("outputs/ablations"))
    p.add_argument("--n_devices", type=int, default=1)
    p.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--show", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")

    run_experiment(
        model_name=args.model_name,
        hf_token=hf_token,
        permutations=args.permutations,
        max_heads_abl=args.max_heads_abl,
        head_type=args.head_type,
        to_abl=args.to_abl,
        mode=args.mode,
        layer_abl=args.layer_abl,
        tokens_path=args.tokens_path,
        induction_scores_csv=args.induction_scores_csv,
        output_dir=args.output_dir,
        n_devices=args.n_devices,
        dtype=args.dtype,
        seed=args.seed,
        show=args.show,
    )        
