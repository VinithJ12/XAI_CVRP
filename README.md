CVRP + Explainable AI Project
Authors: Vinith, Ian, Trevor

What This Project Does
We take a pre-trained Reinforcement Learning model that solves the Capacitated Vehicle Routing Problem (CVRP), and use SHAP to explain WHY the model makes each routing decision.

Project Structure
cvrp_xai_project/
├── README.md                  ← You are here
├── requirements.txt           ← Python packages to install
├── 1_generate_problem.py      ← Step 1: Create CVRP instances
├── 2_run_rl_model.py          ← Step 2: Run the RL model, log decisions
├── 3_shap_explain.py          ← Step 3: Run SHAP on logged decisions
├── 4_evaluate.py              ← Step 4: Evaluate + visualize explanations
└── utils/
    ├── cvrp_env.py            ← CVRP environment (state, features)
    └── visualize.py           ← Plotting helpers
How to Run (in order)
pip install -r requirements.txt
python 1_generate_problem.py
python 2_run_rl_model.py
python 3_shap_explain.py
python 4_evaluate.py
Key Hypothesis
SHAP will identify DISTANCE as the dominant decision variable — meaning the RL model mostly picks the nearest unvisited customer.
