# rl_exercises/week_6/plot_ppo_comparison.py
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from rliable import metrics
from rliable.library import get_interval_estimates
from rliable.plot_utils import plot_sample_efficiency_curve

PLOTS_DIR = "rl_exercises/week_6/plots"
PPO_DIR = "results/ppo"
AC_DIR = "results/actor_critic"
os.makedirs(PLOTS_DIR, exist_ok=True)

env = "LunarLander-v3"
seeds = [0, 892735897, 142312]
ppo_algorithms = ["ppo_vanilla", "ppo_enhanced"]
ac_algorithms = ["gae"]

score_dict = {}
steps = None

# Load the PPO results
for alg in ppo_algorithms:
    all_returns = []
    for seed in seeds:
        pickle_path = f"{PPO_DIR}/{env}_{alg}_seed{seed}.pkl"
        if os.path.exists(pickle_path):
            with open(pickle_path, "rb") as f:
                data = pickle.load(f)
                # PPO stores returns as Dict {step: return}
                if isinstance(data["returns"], dict):
                    all_returns.append(list(data["returns"].values()))
                else:
                    all_returns.append(data["returns"])
                if steps is None:
                    steps = np.array(data["steps"])
    if all_returns:
        score_dict[alg.replace("_", " ").title()] = np.array(all_returns)

# Load the actor critic results
for alg in ac_algorithms:
    all_returns = []
    for seed in seeds:
        pickle_path = f"{AC_DIR}/{env}_{alg}_seed{seed}.pkl"
        if os.path.exists(pickle_path):
            with open(pickle_path, "rb") as f:
                data = pickle.load(f)
                # Actor-Critic stores returns as List
                all_returns.append(data["returns"])
                if steps is None:
                    steps = np.array(data["steps"])
    if all_returns:
        score_dict[alg.replace("_", " ").title()] = np.array(all_returns)

if score_dict:
    iqm = lambda x: np.array(
        [metrics.aggregate_iqm(x[:, i]) for i in range(x.shape[1])]
    )
    iqm_scores, iqm_cis = get_interval_estimates(score_dict, iqm, reps=2000)

    plt.figure(figsize=(12, 8))
    plot_sample_efficiency_curve(
        steps + 1,
        iqm_scores,
        iqm_cis,
        algorithms=list(score_dict.keys()),
        xlabel="Environment Steps",
        ylabel="IQM Evaluation Return",
    )
    plt.title(f"PPO vs. Actor-Critic on {env}", fontsize=16)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/{env}_ppo_comparison.png", dpi=300)
    plt.close()
    print(f"Plot saved to {PLOTS_DIR}/{env}_ppo_comparison.png")
