"""
utils/cvrp_env.py
─────────────────────────────────────────────────────────────────────────────
CVRP Environment: defines what a routing problem looks like and how to
extract human-understandable FEATURES from the model's perspective at
each decision step.

KEY CONCEPTS:
  - A CVRP "instance" = a set of customer locations + demands + one depot
  - At each step, the truck is at some location and must pick the NEXT customer
  - We compute features that a human logistics manager would care about:
      1. distance_to_candidate   — how far away is this customer?
      2. remaining_capacity_ratio — how full is the truck (0=empty, 1=full)?
      3. candidate_demand_ratio  — how big is this customer's order?
      4. distance_to_depot       — how far is the candidate from home base?
      5. urgency_score           — are we running low on capacity?
"""

import numpy as np


class CVRPInstance:
    """
    Represents one CVRP problem instance.

    Attributes:
        n_customers  : number of customers (NOT counting the depot)
        coords       : (n_customers+1, 2) array — row 0 is the depot,
                       rows 1..n are customers, all in [0, 1] normalized space
        demands      : (n_customers+1,) array — depot demand = 0,
                       customer demands are in (0, 1] normalized by truck capacity
        capacity     : maximum load the truck can carry (always 1.0 normalized)
    """

    def __init__(self, n_customers: int, seed: int = None):
        """
        Generate a random CVRP instance.

        Args:
            n_customers: how many customers to place on the map
            seed: random seed for reproducibility (same seed = same problem)
        """
        rng = np.random.RandomState(seed)

        self.n_customers = n_customers
        self.capacity = 1.0  # normalized — demands will be fractions of this

        # ── Coordinates ────────────────────────────────────────────────────
        # Row 0 = depot (the warehouse trucks start from)
        # Rows 1..n = customers, randomly scattered in [0,1]² space
        self.coords = rng.uniform(0, 1, size=(n_customers + 1, 2))

        # ── Demands ────────────────────────────────────────────────────────
        # Each customer needs between 10% and 40% of truck capacity.
        # Depot has demand 0 (we never need to "deliver" to the warehouse).
        customer_demands = rng.uniform(0.1, 0.4, size=n_customers)
        self.demands = np.concatenate([[0.0], customer_demands])

    def euclidean_distance(self, i: int, j: int) -> float:
        """Return straight-line distance between location i and location j."""
        return float(np.linalg.norm(self.coords[i] - self.coords[j]))

    def __repr__(self):
        return f"CVRPInstance(n_customers={self.n_customers})"


class CVRPState:
    """
    Tracks the truck's current state as it moves through the route.

    Think of this as the truck's "dashboard" at any moment:
      - Where am I now?
      - Which customers have I already visited?
      - How much cargo capacity do I have left?
    """

    def __init__(self, instance: CVRPInstance):
        self.instance = instance
        self.current_location = 0          # start at depot (index 0)
        self.visited = set([0])            # depot is always "visited"
        self.remaining_capacity = instance.capacity

    @property
    def unvisited_customers(self) -> list:
        """Return indices of customers not yet served."""
        all_customers = set(range(1, self.instance.n_customers + 1))
        return list(all_customers - self.visited)

    def visit(self, customer_idx: int):
        """
        Move the truck to a customer, serve them, update state.

        Args:
            customer_idx: the customer index (1-based) to visit next
        """
        self.current_location = customer_idx
        self.visited.add(customer_idx)
        self.remaining_capacity -= self.instance.demands[customer_idx]

        # If truck is too full to serve any remaining customer, return to depot
        remaining = self.unvisited_customers
        if remaining:
            min_demand = min(self.instance.demands[c] for c in remaining)
            if self.remaining_capacity < min_demand:
                # Return to depot to refill
                self.current_location = 0
                self.remaining_capacity = self.instance.capacity

    def is_done(self) -> bool:
        """True when all customers have been visited."""
        return len(self.unvisited_customers) == 0


def extract_features(state: CVRPState, candidate_idx: int) -> dict:
    """
    Extract human-interpretable features for ONE candidate customer.

    This is the heart of the XAI pipeline. Instead of feeding raw
    coordinates into SHAP, we compute features a logistics manager
    would actually understand.

    Args:
        state         : current truck state (location, capacity, visited set)
        candidate_idx : the customer we're considering visiting next

    Returns:
        dict with 5 float features (all normalized to [0, 1] range)
    """
    inst = state.instance

    # ── Feature 1: Distance to candidate ──────────────────────────────────
    # How far is the truck from this candidate customer?
    # Normalized by the max possible distance (diagonal of unit square ≈ 1.414)
    dist_to_candidate = inst.euclidean_distance(
        state.current_location, candidate_idx
    ) / np.sqrt(2)

    # ── Feature 2: Remaining capacity ratio ───────────────────────────────
    # How much room does the truck have left?
    # 1.0 = completely empty (lots of room), 0.0 = completely full
    remaining_capacity_ratio = state.remaining_capacity / inst.capacity

    # ── Feature 3: Candidate demand ratio ─────────────────────────────────
    # How big is this customer's order relative to truck capacity?
    # High value = this customer will fill up a lot of the truck
    candidate_demand_ratio = inst.demands[candidate_idx] / inst.capacity

    # ── Feature 4: Distance from candidate back to depot ──────────────────
    # After visiting this customer, how far would we be from home?
    # Useful for end-of-route planning (don't strand yourself far away)
    dist_candidate_to_depot = inst.euclidean_distance(
        candidate_idx, 0  # index 0 = depot
    ) / np.sqrt(2)

    # ── Feature 5: Urgency score ───────────────────────────────────────────
    # Are we running so low on capacity that we MUST return to depot soon?
    # High value = we're nearly full and urgently need to refill
    # Formula: how many more "average" customers can we fit?
    avg_demand = np.mean(inst.demands[1:])  # average customer demand
    remaining_customers = len(state.unvisited_customers)
    if remaining_customers > 0 and avg_demand > 0:
        # If remaining_capacity < avg_demand, urgency > 1 (very urgent)
        urgency_score = min(1.0, avg_demand / max(state.remaining_capacity, 1e-6))
    else:
        urgency_score = 0.0

    return {
        "distance_to_candidate": dist_to_candidate,
        "remaining_capacity_ratio": remaining_capacity_ratio,
        "candidate_demand_ratio": candidate_demand_ratio,
        "dist_candidate_to_depot": dist_candidate_to_depot,
        "urgency_score": urgency_score,
    }


# ── List of feature names (used throughout for consistent ordering) ─────────
FEATURE_NAMES = [
    "distance_to_candidate",
    "remaining_capacity_ratio",
    "candidate_demand_ratio",
    "dist_candidate_to_depot",
    "urgency_score",
]
