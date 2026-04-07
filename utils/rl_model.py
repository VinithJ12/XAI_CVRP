"""
utils/rl_model.py
─────────────────────────────────────────────────────────────────────────────
The routing model classes. Importable from any script.

Contains:
  KoolSurrogate  — greedy heuristic that mimics the Kool RL model's behavior.
                   Use this by default.
  (See commented block at bottom for how to swap in the real Kool model)
─────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
from utils.cvrp_env import CVRPState


class KoolSurrogate:
    """
    Simulates the Kool RL model's routing behavior.

    The real Kool model computes attention scores between:
      - A "query" vector representing the current truck state
      - "Key" vectors for each unvisited customer

    At greedy inference, it picks argmax of these scores — which
    strongly correlates with: short distance + sufficient capacity.

    Our surrogate captures this directly without the neural network:
      score(candidate) = 1/distance - 0.3 × dist_to_depot + noise

    This gives SHAP the same feature→decision relationship to analyze.
    """

    def __init__(self, noise_scale: float = 0.05):
        """
        Args:
            noise_scale: random noise added to scores (0 = perfectly greedy).
                         Use 0.05 for data generation, 0.0 for sensitivity tests.
        """
        self.noise_scale = noise_scale

    def score_candidate(self, state: CVRPState, candidate_idx: int) -> float:
        """
        Compute a desirability score for visiting this candidate next.

        Higher score → model prefers this customer.
        Returns -inf if the customer's demand exceeds remaining capacity
        (infeasible — we can never visit them without refueling first).
        """
        inst = state.instance

        dist = inst.euclidean_distance(state.current_location, candidate_idx)
        dist_to_depot = inst.euclidean_distance(candidate_idx, 0)
        demand = inst.demands[candidate_idx]

        if demand > state.remaining_capacity:
            return -np.inf  # infeasible

        score = (1.0 / (dist + 1e-6)) - 0.3 * dist_to_depot
        score += np.random.normal(0, self.noise_scale)
        return score

    def choose_next(self, state: CVRPState) -> int:
        """Return the index of the next customer to visit."""
        candidates = state.unvisited_customers
        scores = {c: self.score_candidate(state, c) for c in candidates}
        return max(scores, key=scores.get)


# ── HOW TO SWAP IN THE REAL KOOL MODEL ─────────────────────────────────────
#
# 1. Clone their repo:
#      git clone https://github.com/wouterkool/attention-learn-to-route
#
# 2. Download pretrained weights from the link in their README (Google Drive)
#      e.g. pretrained/cvrp_20/epoch-99.pt
#
# 3. Uncomment and complete this class:
#
# class KoolRealModel:
#     def __init__(self, weights_path, graph_size):
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
#         state_dict = torch.load(weights_path, map_location='cpu')
#         self.model.load_state_dict(state_dict['model'])
#         self.model.eval()
#         self.graph_size = graph_size
#
#     def choose_next(self, state: CVRPState) -> int:
#         # You need to:
#         #   1. Convert CVRPState → the tensor format Kool expects
#         #   2. Run self.model.forward() in eval mode (torch.no_grad())
#         #   3. Extract the argmax customer from the output logits
#         # See their problems/vrp/vrp_baseline.py for the input format
#         raise NotImplementedError("Fill in tensor conversion here")
