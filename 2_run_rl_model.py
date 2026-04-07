"""
2_run_rl_model.py
─────────────────────────────────────────────────────────────────────────────
STEP 2: Run the RL routing model on every instance and LOG each decision
        with its features.

ABOUT THE KOOL ET AL. MODEL:
  The paper uses an Attention-based neural network (like a small Transformer)
  trained with REINFORCE (a policy gradient RL algorithm).

  Setting up their exact model requires:
    - Cloning: https://github.com/wouterkool/attention-learn-to-route
    - Downloading pretrained weights (~50MB)
    - Matching their exact input format

  For this project we implement TWO options:
    A) KoolSurrogate  — a faithful simulation of the model's BEHAVIOR
       using a greedy nearest-neighbor + capacity heuristic. This is
       sufficient for SHAP analysis because SHAP only needs:
         input features → output decision
       It does NOT need the model to be a neural network.

    B) KoolRealModel  — skeleton code showing HOW to load the actual
       Kool PyTorch model if you have the weights file. Uncomment and
       use this once you've cloned their repo.

  For your paper: run the surrogate first to validate the pipeline,
  then swap in the real model for final results.

WHAT THIS SCRIPT PRODUCES:
  data/decisions_n10.csv   ← one row per routing decision for n=10
  data/decisions_n20.csv
  data/decisions_n50.csv
─────────────────────────────────────────────────────────────────────────────
"""

import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm  # progress bar

from utils.cvrp_env import CVRPInstance, CVRPState, extract_features, FEATURE_NAMES
from utils.rl_model import KoolSurrogate

os.makedirs("data", exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# OPTION A: Surrogate model (use this by default)
# ═══════════════════════════════════════════════════════════════════════════

class KoolSurrogate:
    """
    Simulates the Kool RL model's routing BEHAVIOR using a greedy heuristic.

    The real Kool model uses attention scores over (query, key) pairs to
    compute a probability distribution over unvisited customers, then
    samples from it. At inference time (greedy decoding) it just picks
    argmax — which is approximately "the most promising nearby customer."

    Our surrogate captures the same pattern:
      - Strongly prefer nearby customers        (distance↓ = score↑)
      - Prefer customers that fit in the truck  (feasibility check)
      - Add small noise to mimic non-greedy behavior

    This gives us realistic feature→decision mappings for SHAP to analyze.
    """

    def __init__(self, noise_scale: float = 0.05):
        """
        Args:
            noise_scale: how much random noise to add to scores (0 = pure greedy)
                         Small noise makes the model less perfectly predictable,
                         which is more realistic.
        """
        self.noise_scale = noise_scale

    def score_candidate(self, state: CVRPState, candidate_idx: int) -> float:
        """
        Compute a "desirability" score for visiting this candidate next.
        Higher score = model prefers this customer.

        Score formula mirrors the attention mechanism:
          high score = short distance + has capacity + not too far from depot
        """
        inst = state.instance

        # Distance from current location to candidate
        dist = inst.euclidean_distance(state.current_location, candidate_idx)

        # Distance from candidate back to depot (penalizes stranding far away)
        dist_to_depot = inst.euclidean_distance(candidate_idx, 0)

        # Feasibility: can the truck actually serve this customer?
        demand = inst.demands[candidate_idx]
        if demand > state.remaining_capacity:
            return -np.inf  # impossible — truck is too full

        # Score = inverse distance (closer = better score)
        # Minus a small penalty for being far from depot (route-end awareness)
        score = (1.0 / (dist + 1e-6)) - 0.3 * dist_to_depot

        # Add small random noise (realistic: real model isn't perfectly greedy)
        score += np.random.normal(0, self.noise_scale)

        return score

    def choose_next(self, state: CVRPState) -> int:
        """
        Given the current state, pick the best next customer.

        Returns:
            index of the chosen customer (1-based)
        """
        candidates = state.unvisited_customers

        # Score every unvisited customer
        scores = {c: self.score_candidate(state, c) for c in candidates}

        # Return the one with the highest score (greedy argmax)
        return max(scores, key=scores.get)


# ═══════════════════════════════════════════════════════════════════════════
# OPTION B: Real Kool model (uncomment after cloning their repo)
# ═══════════════════════════════════════════════════════════════════════════
# Instructions:
#   git clone https://github.com/wouterkool/attention-learn-to-route
#   Download pretrained weights from their Google Drive link in the README
#   Then uncomment the block below and swap model = KoolRealModel(...)
#
# class KoolRealModel:
#     def __init__(self, weights_path: str, graph_size: int):
#         import sys
#         sys.path.insert(0, "attention-learn-to-route")
#         from nets.attention_model import AttentionModel
#         import torch
#
#         self.model = AttentionModel(
#             embedding_dim=128,
#             hidden_dim=128,
#             n_encode_layers=3,
#             normalization='batch',
#             problem='cvrp'
#         )
#         self.model.load_state_dict(torch.load(weights_path, map_location='cpu'))
#         self.model.eval()
#
#     def choose_next(self, state: CVRPState) -> int:
#         # Convert CVRPState → tensor format Kool model expects
#         # Their input format: [depot_coords, customer_coords, demands]
#         # See their README for exact tensor shapes
#         raise NotImplementedError("Fill in the tensor conversion here")


def run_model_on_instance(model, instance: CVRPInstance) -> list[dict]:
    """
    Run the routing model on ONE instance and record every decision made.

    At each decision step, BEFORE the model picks a customer, we record:
      - The features of EVERY candidate customer (not just the chosen one)
      - Which candidate was actually chosen (label = 1, rest = 0)

    This gives SHAP enough data to learn: "given these features, the model
    picks the candidate with label=1."

    Returns:
        List of dicts, one per (step, candidate) pair.
        Each dict has: features + 'chosen' (1/0) + metadata
    """
    state = CVRPState(instance)
    records = []
    step = 0

    while not state.is_done():
        candidates = state.unvisited_customers

        # Ask the model which customer to visit next
        chosen = model.choose_next(state)

        # ── Record a row for EVERY candidate (chosen + not-chosen) ─────────
        # SHAP needs to see both chosen and unchosen options to understand
        # what features make a customer get selected.
        for c in candidates:
            feats = extract_features(state, c)
            row = {
                "step": step,
                "candidate": c,
                "chosen": int(c == chosen),  # 1 if this candidate was picked
                "n_customers": instance.n_customers,
                "n_remaining": len(candidates),
                **feats,  # unpack all 5 features as columns
            }
            records.append(row)

        # Now actually move the truck to the chosen customer
        state.visit(chosen)
        step += 1

    return records


# ═══════════════════════════════════════════════════════════════════════════
# MAIN: Run on all instance sizes
# ═══════════════════════════════════════════════════════════════════════════

# ── Initialize the model ───────────────────────────────────────────────────
# KoolSurrogate is in utils/rl_model.py
# Swap for KoolRealModel(weights_path, graph_size) when you have the weights
model = KoolSurrogate(noise_scale=0.05)

PROBLEM_SIZES = [10, 20, 50]

for n in PROBLEM_SIZES:
    print(f"\n── Running model on n={n} customer instances ──")

    # Load the instances we saved in step 1
    with open(f"data/instances_n{n}.pkl", "rb") as f:
        instances = pickle.load(f)

    all_records = []

    # tqdm wraps the loop and shows a progress bar
    for inst in tqdm(instances, desc=f"  n={n}"):
        records = run_model_on_instance(model, inst)
        all_records.extend(records)

    # Convert list of dicts → pandas DataFrame and save as CSV
    df = pd.DataFrame(all_records)
    save_path = f"data/decisions_n{n}.csv"
    df.to_csv(save_path, index=False)

    # Quick stats to verify the data looks reasonable
    n_decisions = df[df["chosen"] == 1].shape[0]
    print(f"  ✓ {len(df)} rows ({n_decisions} actual decisions) → {save_path}")
    print(f"  Feature means:\n{df[FEATURE_NAMES].mean().round(3).to_string()}")

print("\n✓ Decision logs saved. Next step: run  python 3_shap_explain.py")
