import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from rliable import metrics
from rliable.library import get_interval_estimates
from rliable.plot_utils import plot_sample_efficiency_curve

PLOTS_DIR = "rl_exercises/week_6/plots"
SAC_DIR = "results/sac"
os.makedirs(PLOTS_DIR, exist_ok=True)

env = "LunarLanderContinuous-v3"
# seeds = [0, 892735897, 142312]
seeds = [0]

algorithms = ["SAC"]
algorithm_dirs = [SAC_DIR]

score_dict = {}
steps = None

for alg_name, alg_dir in zip(algorithms, algorithm_dirs):
    all_returns = []
    for seed in seeds:
        pickle_path = f"{alg_dir}/{env}_{alg_name.lower()}_seed{seed}.pkl"
        if os.path.exists(pickle_path):
            with open(pickle_path, "rb") as f:
                data = pickle.load(f)
                if isinstance(data["returns"], dict):
                    returns = list(data["returns"].values())
                else:
                    returns = data["returns"]
                all_returns.append(returns)
                if steps is None:
                    steps = np.array(data["steps"])
    if all_returns:
        score_dict[alg_name] = np.array(all_returns)

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
    plt.title(f"SAC on {env}", fontsize=16)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/{env}_sac.png", dpi=300)
    plt.close()
    print(f"Plot saved to {PLOTS_DIR}/{env}_sac.png")
