# This script is used to compare SAC and PPO
# The training of SAC on 3 seeds takes too long. Therefore, we only use one seed in this comparison.
# In Addition to that, notice that the PPO results are still from the discrete environment whereas the SAC results are from the continuous environment.
# In order to make PPO ready for the continuous environment, several changes would have to be made to networks.py and the ppo.py itself.
# This could have lead to other problems when using the earlier tests on the discrete environments.
# Therefore, we have decided to compare the discrete PPO results with the continuous SAC results even if this is not perfect.
# However, it still allows us to analyze the differences and evaluate the claims of SAC
# To train sac, we used the train_sac.py and abort after the first seed.

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

PLOTS_DIR = "rl_exercises/week_6/plots"
PPO_DIR = "results/ppo"
SAC_DIR = "results/sac"
os.makedirs(PLOTS_DIR, exist_ok=True)

seed = 0
algorithms = {
    "SAC": f"{SAC_DIR}/LunarLanderContinuous-v3_sac_seed{seed}.pkl",
    "PPO Enhanced": f"{PPO_DIR}/LunarLander-v3_ppo_enhanced_seed{seed}.pkl",
    "PPO Vanilla": f"{PPO_DIR}/LunarLander-v3_ppo_vanilla_seed{seed}.pkl",
}

plt.figure(figsize=(12, 8))
for alg_name, pickle_path in algorithms.items():
    if os.path.exists(pickle_path):
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)
        steps = np.array(data["steps"])
        returns = (
            np.array(list(data["returns"].values()))
            if isinstance(data["returns"], dict)
            else np.array(data["returns"])
        )
        plt.plot(steps, returns, label=alg_name, linewidth=2)

plt.xlabel("Environment Steps", fontsize=12)
plt.ylabel("Evaluation Return", fontsize=12)
plt.title("SAC vs PPO", fontsize=16)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/sac_ppo_single_seed.png", dpi=300)
plt.close()
print(f"Plot saved to {PLOTS_DIR}/sac_ppo_single_seed.png")
