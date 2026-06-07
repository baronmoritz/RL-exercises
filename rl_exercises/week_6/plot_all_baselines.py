import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import rliable.library as rly

baselines = ["none", "avg", "value", "gae"]
envs = ["CartPole-v1", "LunarLander-v3"]
seeds = [0, 892735897, 142312]

os.makedirs("rl_exercises/week_6/plots", exist_ok=True)

for env in envs:
    plt.figure(figsize=(12, 8))
    for baseline in baselines:
        all_returns = []
        all_steps = None
        for seed in seeds:
            pickle_path = f"results/actor_critic/{env}_{baseline}_seed{seed}.pkl"
            if os.path.exists(pickle_path):
                with open(pickle_path, "rb") as f:
                    data = pickle.load(f)
                    all_returns.append(data["returns"])
                    if all_steps is None:
                        all_steps = np.array(data["steps"])

        if all_returns:
            returns_array = np.array(all_returns)
            score_dict = {baseline: returns_array}
            point_estimates, interval_estimates = rly.get_interval_estimates(
                score_dict, lambda x: np.mean(x, axis=0)
            )
            mean_returns = point_estimates[baseline]
            ci_lower = interval_estimates[baseline][0]
            ci_upper = interval_estimates[baseline][1]

            plt.plot(all_steps, mean_returns, label=baseline, linewidth=2)
            plt.fill_between(all_steps, ci_lower, ci_upper, alpha=0.2)

    plt.xlabel("Environment Steps", fontsize=12)
    plt.ylabel("Average Return", fontsize=12)
    plt.title(f"Actor-Critic Baselines on {env}", fontsize=16)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"rl_exercises/week_6/plots/{env}_baselines.png", dpi=300)
    plt.close()
    print(f"Plot saved to rl_exercises/week_6/plots/{env}_baselines.png")
