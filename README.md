# CVRP · Explainable AI
### Can we explain WHY an AI makes delivery routing decisions?
**Authors:** Vinith, Ian, Trevor · DS4024 Final Project · April 2026

---

## What This Project Is

AI models are being deployed right now to route delivery trucks at companies like Amazon, UPS, and DHL. These models are fast and often outperform humans — but nobody can explain *why* they make specific decisions. A logistics manager has no way to verify whether the AI is routing correctly or exploiting some weird pattern it learned during training.

This project builds a complete **Explainable AI (XAI) pipeline** for reinforcement learning routing models. We use **SHAP** (SHapley Additive exPlanations) to explain every routing decision in human-understandable terms — and then rigorously test whether those explanations are actually telling the truth.

---

## The Problem We're Solving

The **Capacitated Vehicle Routing Problem (CVRP)** asks: given a truck with a weight limit and N customers scattered on a map, what's the best order to visit them all?

State-of-the-art RL models (like Kool et al. 2019) solve this better than classical algorithms — but they're black boxes. Our pipeline answers:

> *"The AI chose Customer 7 next. But WHY? Was it distance? Capacity? Urgency? And can a human manager trust that explanation enough to intervene?"*

---

## Hypotheses

We tested three specific hypotheses:

| # | Hypothesis | Result |
|---|---|---|
| H1 | Distance will be the dominant decision factor identified by SHAP | ✓ Confirmed (sanity check) |
| H2 | SHAP explanations will be faithful — changing distance will change decisions | ✓ Confirmed (100% / 99%) |
| H3 | Explanation signal will weaken as problem complexity grows | ✓ Confirmed (74% drop) |

> **Important note on H1:** Distance was built into our surrogate model's scoring formula, so H1 functions as a pipeline sanity check rather than a discovery. H2 and H3 are the genuine empirical contributions — they were not designed in and emerged from the data.

---

## Results

### H1 — Distance ranks #1 across all problem sizes

| Feature | n=10 | n=20 | n=50 |
|---|---|---|---|
| **distance_to_candidate** | **0.0241 ← #1** | **0.0157 ← #1** | **0.0062 ← #1** |
| candidate_demand_ratio | 0.0151 | 0.0079 | 0.0039 |
| urgency_score | 0.0108 | 0.0055 | 0.0028 |
| dist_candidate_to_depot | 0.0085 | 0.0034 | 0.0026 |
| remaining_capacity_ratio | 0.0065 | 0.0054 | 0.0025 |

Distance ranked #1 every single time. However, this is expected given the surrogate model's design — treat it as a sanity check confirming the pipeline works correctly.

---

### H2 — SHAP explanations are faithful (strongest finding)

```
n=10 → 100% of decisions changed when distance was perturbed
n=50 →  99% of decisions changed when distance was perturbed
```

**This is the most meaningful result.** We physically relocated the chosen customer to coordinates (1.5, 1.5) — far outside the map — and observed whether the model changed its mind. It did, almost every single time. This was not designed in. It proves SHAP is not making up stories — distance genuinely controls decisions.

**Practical implication:** A logistics manager who marks a customer as unreachable can trust the model will reliably reroute.

---

### H3 — Signal weakens with problem complexity (second real finding)

```
n=10 → mean |SHAP| for distance = 0.024  ← loud signal
n=20 → mean |SHAP| for distance = 0.016  ← quieter
n=50 → mean |SHAP| for distance = 0.006  ← 74% weaker than n=10
```

As problems grow larger, the distance signal gets progressively weaker. On a 10-customer map, one customer is clearly closest so distance dominates loudly. On a 50-customer map, many customers cluster nearby so distance still wins but capacity, urgency, and depot distance all compete more strongly.

**Practical implication:** SHAP explanations are most actionable for small to medium routing problems (≤20 customers per route segment). For larger problems, no single feature tells the whole story.

---

### Surrogate accuracy

| Problem size | Accuracy |
|---|---|
| n=10 | 81.8% |
| n=20 | 90.5% |
| n=50 | 96.1% |

---

## Project Structure

```
cvrp_xai_project/
│
├── README.md                    ← You are here
├── requirements.txt             ← Python packages
├── dashboard.py                 ← Streamlit interactive dashboard
│
├── 1_generate_problem.py        ← Step 1: Create 150 CVRP instances
├── 2_run_rl_model.py            ← Step 2: Run model, log 77k decisions
├── 3_shap_explain.py            ← Step 3: Train surrogate + run SHAP
├── 4_evaluate.py                ← Step 4: Test H1, H2, H3
│
└── utils/
    ├── __init__.py
    ├── cvrp_env.py              ← CVRP instance, state, feature extraction
    ├── rl_model.py              ← KoolSurrogate routing model
    └── visualize.py             ← Matplotlib plotting helpers
```

---

## How It Works

### The pipeline in plain English

```
150 random delivery maps
        ↓
KoolSurrogate makes routing decisions
(context-aware scoring + softmax sampling)
        ↓
77,000 decisions logged with 5 human features each
        ↓
Random Forest learns to imitate those decisions
(81-96% accuracy)
        ↓
SHAP TreeExplainer explains the Random Forest
(exact Shapley values, not approximations)
        ↓
Three evaluation tests (H1, H2, H3)
        ↓
Human-readable explanations + visualizations
```

### The 5 features

These were designed to be understandable to a logistics manager:

| Feature | Plain English |
|---|---|
| `distance_to_candidate` | How far is the next customer? |
| `remaining_capacity_ratio` | How full is the truck? (1=empty, 0=full) |
| `candidate_demand_ratio` | How big is the customer's order? |
| `dist_candidate_to_depot` | How far from home base will we be? |
| `urgency_score` | Do we urgently need to refuel? |

### The upgraded KoolSurrogate

Our routing model has three key design features that make it SHAP-compatible:

1. **Context-aware scoring** — evaluates each candidate relative to all others, mirroring the real Kool attention mechanism
2. **Softmax decoding** — samples from a probability distribution rather than always picking the top candidate, matching real RL inference
3. **Fixed learned weights** — weights are set once at initialization, never re-sampled per decision. Critical for SHAP: a stable decision function is necessary for meaningful Shapley attribution

---

## Honest Limitations

We believe in being transparent about what this project does and does not prove.

**The surrogate chain problem:**

```
Real Kool RL model
      ↓ approximated by
KoolSurrogate (behavioral match, not exact)
      ↓ approximated by
Random Forest (81-96% accurate)
      ↓ explained by
SHAP
```

Our explanations are three layers removed from the real model. H1 (distance dominates) is partly circular — we built distance into the surrogate formula. The next step is plugging in the real pretrained Kool neural network. If SHAP still finds distance #1, the finding is fully validated on a true black-box model.

**Why H2 and H3 still matter despite this:**
- H2 (sensitivity) does not depend on the surrogate design — we perturbed physical coordinates and observed real behavioral change
- H3 (degradation) emerged from the math of SHAP across problem sizes, not from anything we designed in

---

## How to Run

```bash
# 1. Clone and set up
git clone <your-repo-url>
cd cvrp_xai_project

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the pipeline in order
python 1_generate_problem.py   # ~5 seconds
python 2_run_rl_model.py       # ~30 seconds
python 3_shap_explain.py       # ~2 minutes
python 4_evaluate.py           # ~1 minute

# 5. Launch the dashboard
pip install streamlit plotly
streamlit run dashboard.py
```

---

## Outputs

After running the full pipeline:

```
data/
  instances_n10/20/50.pkl      ← 150 CVRP problem instances
  decisions_n10/20/50.csv      ← 77,000 logged decisions
  shap_values_n10/20/50.csv    ← SHAP values for 500 samples each

plots/
  sample_route_n*.png          ← Route map visualizations
  shap_bar_n*.png              ← Feature importance bar charts
  shap_summary_n*.png          ← SHAP beeswarm summary plots
  shap_waterfall_n*.png        ← Single decision waterfall plots
  shap_scatter_distance_n*.png ← Distance vs SHAP scatter plots
  degradation_by_size.png      ← Signal strength across sizes

results/
  evaluation_report.txt        ← Full text report of all findings
```

---

## Key References
---

## Future Work

- **Plug in the real Kool model** — validate H1 on a true black-box neural network
- **Test at n=100, n=200** — characterize whether degradation continues or plateaus
- **Add time-window constraints** — test whether SHAP detects when time pressure overrides distance
- **User study** — does a logistics manager actually make better decisions with SHAP explanations?
---

*Built with Python · SHAP · scikit-learn · Streamlit · Plotly · Matplotlib*
