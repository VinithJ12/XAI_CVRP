"""
2_run_rl_model.py

STEP 2: Run the RL routing model on every instance and LOG each decision
        with its features.

ABOUT THE KOOL ET AL. MODEL:
  The paper uses an Attention-based neural network (like a small Transformer)
  trained with REINFORCE (a policy gradient RL algorithm).

  We use KoolSurrogate from utils/rl_model.py — an upgraded surrogate that
  mimics the real model's behavior with:
    - Context-aware scoring (relative closeness vs all candidates)
    - Softmax sampling (matches real RL inference)
    - Fixed learned weights (stable decision function for SHAP)

WHAT THIS SCRIPT PRODUCES:
  data/decisions_n10.csv
  data/decisions_n20.csv
  data/decisions_n50.csv

HOW TO RUN:
    python 2_run_rl_model.py
"""

import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.cvrp_env import CVRPInstance, CVRPState, extract_features, FEATURE_NAMES
from utils.rl_model import KoolSurrogate  # ← comes from utils now, not defined here

os.makedirs("data", exist_ok=True)


def run_model_on_instance(model, instance: CVRPInstance) -> list[dict]:
    """
    Run the routing model on ONE instance and record every decision made.

    At each decision step, BEFORE the model picks a customer, we record:
      - The features of EVERY candidate customer (not just the chosen one)
      - Which candidate was actually chosen (label = 1, rest = 0)

    Why record ALL candidates, not just the chosen one?
      SHAP needs contrast. To learn that "short distance → chosen",
      it needs to see both: the close customer that WAS picked (chosen=1)
      AND the far customers that were NOT picked (chosen=0).
      Without the rejected candidates, SHAP has nothing to compare against.

    Returns:
        List of dicts, one per (step, candidate) pair.
        Each dict has: features + 'chosen' (1/0) + metadata.
    """
    state = CVRPState(instance)
    records = []
    step = 0

    while not state.is_done():
        candidates = state.unvisited_customers

        # Ask the model which customer to visit next
        chosen = model.choose_next(state)

        # Record a row for EVERY candidate (chosen + not-chosen)
        for c in candidates:
            feats = extract_features(state, c)
            row = {
                "step": step,
                "candidate": c,
                "chosen": int(c == chosen),  # 1 if picked, 0 if rejected
                "n_customers": instance.n_customers,
                "n_remaining": len(candidates),
                **feats,
            }
            records.append(row)

        # Move the truck to the chosen customer and update state
        state.visit(chosen)
        step += 1

    return records


# Initialize the upgraded surrogate model 
# noise_scale=0.02 : small per-decision noise (realistic uncertainty)
# temperature=1.0  : softmax sampling — not purely greedy
# seed=42          : reproducible fixed weights every run
model = KoolSurrogate(noise_scale=0.02, temperature=1.0, seed=42)

PROBLEM_SIZES = [10, 20, 50]

for n in PROBLEM_SIZES:
    print(f"\n── Running model on n={n} customer instances ──")

    with open(f"data/instances_n{n}.pkl", "rb") as f:
        instances = pickle.load(f)

    all_records = []

    for inst in tqdm(instances, desc=f"  n={n}"):
        records = run_model_on_instance(model, inst)
        all_records.extend(records)

    df = pd.DataFrame(all_records)
    save_path = f"data/decisions_n{n}.csv"
    df.to_csv(save_path, index=False)

    n_decisions = df[df["chosen"] == 1].shape[0]
    print(f"  ✓ {len(df)} rows ({n_decisions} actual decisions) → {save_path}")
    print(f"  Feature means:\n{df[FEATURE_NAMES].mean().round(3).to_string()}")

print("\n✓ Decision logs saved. Next step: run  python 3_shap_explain.py")
