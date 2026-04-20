"""
3_shap_explain.py

STEP 3: Apply SHAP to explain the routing decisions logged in step 2.

HOW SHAP WORKS (plain English):
  Imagine the model decided to visit Customer 7 instead of Customer 3.
  SHAP asks: "How much did each feature contribute to that choice?"

  It works by playing a game theory trick called the "Shapley value":
    1. Take all possible SUBSETS of features
    2. For each subset, see how much the model's prediction changes
       with vs. without that feature
    3. Average these changes across all subsets → that's the SHAP value

  SHAP value > 0 → this feature PUSHED the model toward choosing this candidate
  SHAP value < 0 → this feature PUSHED the model AWAY from this candidate
  SHAP value ≈ 0 → this feature had little effect on this specific decision

WHAT WE'RE EXPLAINING:
  We train a simple surrogate model (Random Forest) on our decision log:
    Input : [distance_to_candidate, remaining_capacity_ratio, ...]
    Output: was this candidate CHOSEN? (1) or not? (0)

  Then we run SHAP on THAT surrogate. This is the standard approach when
  the original model (Kool's neural network) is too complex to explain directly.

WHAT THIS SCRIPT PRODUCES:
  data/shap_values_n10.csv   ← SHAP value for each feature, each decision
  data/shap_values_n20.csv
  data/shap_values_n50.csv
  plots/shap_bar_n*.png      ← bar charts of feature importance
"""

import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend (saves files without a display)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from utils.cvrp_env import FEATURE_NAMES
from utils.visualize import plot_shap_bar

os.makedirs("plots", exist_ok=True)

PROBLEM_SIZES = [10, 20, 50]


def train_surrogate(df: pd.DataFrame):
    """
    Train a Random Forest classifier on the decision log.

    WHY RANDOM FOREST?
      SHAP has a dedicated TreeExplainer that works EXACTLY and EFFICIENTLY
      with tree-based models. It doesn't need to sample subsets randomly —
      it can traverse the tree structure to compute exact Shapley values.
      This makes it much faster and more accurate than KernelSHAP.

    Args:
        df: DataFrame with columns FEATURE_NAMES + 'chosen'

    Returns:
        (trained_model, X_train, X_test, y_test)
    """
    X = df[FEATURE_NAMES].values  # features matrix (n_rows × 5)
    y = df["chosen"].values       # target: 1 = chosen, 0 = not chosen

    # Split into train/test so we can report accuracy honestly
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train the forest
    # n_estimators=100: 100 decision trees (enough for stability)
    # max_depth=6: prevents overfitting on our small feature set
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=42,
        n_jobs=-1  # use all CPU cores
    )
    clf.fit(X_train, y_train)

    # Report accuracy (sanity check — should be >70% for the surrogate)
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"  Surrogate accuracy on held-out data: {acc:.1%}")

    return clf, X_train, X_test, y_test


def compute_shap_values(clf, X_background: np.ndarray,
                         X_explain: np.ndarray) -> np.ndarray:
    """
    Use SHAP's TreeExplainer to compute Shapley values.

    Args:
        clf         : trained RandomForest model
        X_background: training data used as SHAP "background" distribution
                      (SHAP needs this to understand what's "normal")
        X_explain   : the data points we want explanations for

    Returns:
        shap_values : array of shape (n_samples, n_features)
                      SHAP value for each feature for each sample
                      (we take class-1 values = probability of being chosen)
    """
    # TreeExplainer uses the exact Shapley formula for tree models.
    # It's much faster than KernelSHAP which approximates with sampling.
    explainer = shap.TreeExplainer(
        clf,
        data=X_background,  # background dataset for marginal expectations
        feature_perturbation="interventional"
        # "interventional" = what happens if we CHANGE a feature value
        # (vs. "tree_path_dependent" which conditions on the tree's logic)
    )

    # Compute SHAP values
    shap_vals = explainer.shap_values(X_explain)

    # Different SHAP versions return different shapes — handle all:
    #   Old SHAP: list of 2 arrays, each (n_samples, n_features)
    #   New SHAP: single array of shape (n_samples, n_features, n_classes)
    if isinstance(shap_vals, list):
        return explainer, shap_vals[1]       # class 1 = "was chosen"
    elif shap_vals.ndim == 3:
        return explainer, shap_vals[:, :, 1] # new format — take class 1
    return explainer, shap_vals              # already (n_samples, n_features)

# MAIN: Explain decisions for each problem size

for n in PROBLEM_SIZES:
    print(f"\n══ SHAP analysis for n={n} ══")

    # Load decision log
    df = pd.read_csv(f"data/decisions_n{n}.csv")
    print(f"  Loaded {len(df)} rows ({df['chosen'].sum()} chosen decisions)")

    # Train surrogate 
    clf, X_train, X_test, y_test = train_surrogate(df)

    # Compute SHAP values 
    # We explain a random subset of test rows (full set can be slow)
    np.random.seed(42)
    n_explain = min(500, len(X_test))
    idx = np.random.choice(len(X_test), n_explain, replace=False)
    X_explain = X_test[idx]

    print(f"  Computing SHAP values for {n_explain} samples...")
    explainer, shap_vals = compute_shap_values(clf, X_train, X_explain)
    # shap_vals shape: (n_explain, 5)  — 5 features

    # Save SHAP values as a DataFrame 
    # One row per sample, columns = shap_<feature_name>
    shap_df = pd.DataFrame(X_explain, columns=FEATURE_NAMES)
    for fi, fname in enumerate(FEATURE_NAMES):
        shap_df[f"shap_{fname}"] = shap_vals[:, fi]

    save_path = f"data/shap_values_n{n}.csv"
    shap_df.to_csv(save_path, index=False)
    print(f"  ✓ Saved SHAP values → {save_path}")

    # Mean |SHAP| = overall feature importance
    # Taking the ABSOLUTE value because we care about MAGNITUDE of influence,
    # not direction. Then average across all explained samples.
    mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)

    print("\n  Feature importances (mean |SHAP value|):")
    importance_df = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False)
    print(importance_df.to_string(index=False))

    # Bar chart
    plot_shap_bar(
        shap_values_mean=mean_abs_shap,
        feature_names=FEATURE_NAMES,
        title=f"SHAP feature importance — n={n} customers",
        save_path=f"plots/shap_bar_n{n}.png"
    )

    # SHAP summary plot (built-in beeswarm)
    # This is the classic SHAP visualization:
    #   - Each dot = one data point
    #   - Color = feature value (red=high, blue=low)
    #   - X position = SHAP value (right=pushed toward "chosen")
    fig, ax = plt.subplots(figsize=(9, 5))
    shap.summary_plot(
        shap_vals,
        X_explain,
        feature_names=FEATURE_NAMES,
        show=False,
        plot_type="dot"
    )
    plt.title(f"SHAP summary plot — n={n} customers", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"plots/shap_summary_n{n}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved SHAP summary plot → plots/shap_summary_n{n}.png")


    # Waterfall plot — explains ONE specific decision in detail ──────────
    # Shows exactly how each feature pushed THIS specific decision
    # up or down from the baseline. Much more intuitive for a human
    # operator than aggregate charts — it answers "why THIS choice?"
    #
    # Read it like this:
    #   Base value     = what the model predicts on average (no info)
    #   Each bar       = how much one feature changed that prediction
    #   Red bar        = this feature INCREASED probability of being chosen
    #   Blue bar       = this feature DECREASED probability of being chosen
    #   Final value    = the model's actual prediction for this candidate
    base_val = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = float(np.array(base_val).flat[1])
    else:
        base_val = float(base_val)

    shap.plots.waterfall(
        shap.Explanation(
            values        = shap_vals[0],    # SHAP values for 1st sample
            base_values   = base_val,        # average model output
            data          = X_explain[0],    # actual feature values
            feature_names = FEATURE_NAMES
        ),
        show=False
    )
    plt.title(f"Single decision explanation — n={n} customers", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"plots/shap_waterfall_n{n}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved waterfall plot → plots/shap_waterfall_n{n}.png")

print("\n✓ SHAP analysis complete. Next step: run  python 4_evaluate.py")
