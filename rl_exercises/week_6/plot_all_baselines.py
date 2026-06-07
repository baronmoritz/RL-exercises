import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from rliable import metrics
from rliable.library import get_interval_estimates
from rliable.plot_utils import plot_sample_efficiency_curve

baselines = ["none", "avg", "value", "gae"]
envs = ["CartPole-v1", "LunarLander-v3"]
seeds = [0, 892735897, 142312]

os.makedirs("rl_exercises/week_6/plots", exist_ok=True)

for env in envs:
    score_dict = {}
    steps = None
    for baseline in baselines:
        all_returns = []
        for seed in seeds:
            pickle_path = f"results/actor_critic/{env}_{baseline}_seed{seed}.pkl"
            if os.path.exists(pickle_path):
                with open(pickle_path, "rb") as f:
                    data = pickle.load(f)
                    all_returns.append(data["returns"])
                    if steps is None:
                        steps = np.array(data["steps"])

        if all_returns:
            score_dict[baseline] = np.array(all_returns)

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
        plt.title(f"Actor-Critic Baselines on {env}", fontsize=16)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"rl_exercises/week_6/plots/{env}_baselines.png", dpi=300)
        plt.close()
        print(f"Plot saved to rl_exercises/week_6/plots/{env}_baselines.png")
