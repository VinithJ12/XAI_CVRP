"""
1_generate_problem.py

STEP 1: Generate CVRP problem instances of different sizes and save them.

WHY THIS STEP EXISTS:
  We test our XAI approach on SMALL (10 customers), MEDIUM (20), and
  LARGE (50) problems because the abstract says:
    "We expect SHAP to work better on simpler routing decisions and
     become less reliable on larger or more complicated problems."
  So we need instances of each size to compare.

WHAT THIS SCRIPT PRODUCES:
  data/instances_n10.pkl   ← 50 small problems
  data/instances_n20.pkl   ← 50 medium problems
  data/instances_n50.pkl   ← 50 large problems
  plots/sample_route_n10.png  ← visual of one small instance
  plots/sample_route_n20.png  ← visual of one medium instance
  plots/sample_route_n50.png  ← visual of one large instance

HOW TO RUN THIS SCRIPT:
  python 1_generate_problem.py

"""

import os
import pickle
import numpy as np

# Our own utility classes defined in utils/cvrp_env.py
from utils.cvrp_env import CVRPInstance
from utils.visualize import plot_route

# Configuration 
# How many customers in each "size tier"
PROBLEM_SIZES   = [10, 20, 50]

# How many random instances to generate per size
# More instances = more SHAP data = more reliable conclusions
INSTANCES_PER_SIZE = 50

# Create output folders if they don't exist yet
os.makedirs("data",  exist_ok=True)
os.makedirs("plots", exist_ok=True)

# Main generation loop
for n in PROBLEM_SIZES:
    print(f"\n Generating {INSTANCES_PER_SIZE} instances with n={n} customers")

    instances = []
    for i in range(INSTANCES_PER_SIZE):
        # Each instance gets a unique seed so results are reproducible.
        # seed = n * 1000 + i ensures seeds don't overlap across sizes.
        seed = n * 1000 + i
        inst = CVRPInstance(n_customers=n, seed=seed)
        instances.append(inst)

        if i == 0:
            # Print a quick sanity check on the first instance
            print(f"  Instance 0: depot at {inst.coords[0].round(3)}")
            print(f"  Customer 1: coords={inst.coords[1].round(3)}, "
                  f"demand={inst.demands[1]:.3f}")

    # Save all instances to a pickle file for later use in training and SHAP analysis.
    # pickle = Python's way of saving ANY object to disk
    save_path = f"data/instances_n{n}.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(instances, f)
    print(f"  ✓ Saved {len(instances)} instances → {save_path}")

    # Plot the first instance so we can visually inspect it and verify the coordinates and demands look reasonable. 
    # We don't have a route yet, so we just show depot-to-all-customers
    # as a "star" pattern to verify the map looks reasonable.
    first_inst = instances[0]
    # Simple "visit all customers in order, return home" for visualization
    naive_route = [0] + list(range(1, n + 1)) + [0]
    plot_route(
        first_inst,
        route=naive_route,
        title=f"Sample instance: n={n} customers (naive order, not optimized)",
        save_path=f"plots/sample_route_n{n}.png"
    )

print("\n✓ All instances generated. Check data/ and plots/ folders.")
print("Next step: run  python 2_run_rl_model.py")
