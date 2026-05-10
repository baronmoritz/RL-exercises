# import numpy as np
# from rliable import library as rll
# from rliable import metrics as rlm
# from rliable import plotting as rlp
# import matplotlib.pyplot as plt

# # 1. Load your data into a dictionary
# # Format: { 'algorithm_name': array(num_seeds, num_points) }
# dqn_rewards = []
# for seed in range(5):
#     # Load your saved rewards here
#     # Ensure all runs have the same length for the matrix
#     dqn_rewards.append(loaded_rewards_from_seed)

# algorithm_data = {'DQN': np.array(dqn_rewards)}

# # 2. Compute aggregate metrics
# # We define a 'score' (e.g., average reward over the last 100 frames)
# # or use the full training curves.
# # Target score for CartPole-v1 is 500.
# aggregate_func = lambda x: np.mean(x, axis=-1)
# aggregate_scores, aggregate_interval = rll.get_interval_estimates(
#     {'DQN': aggregate_func(algorithm_data['DQN'])},
#     rlm.aggregate_iqm,
#     reps=2000
# )

# # 3. Plotting the IQM and Probability Distributions
# fig, ax = plt.subplots(figsize=(7, 5))
# rlp.plot_interval_estimates(aggregate_scores, aggregate_interval, ax=ax)
# plt.title("Interquartile Mean (IQM) across 5 Seeds")
# plt.savefig("dqn_rliable_iqm.png")

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from rliable import library as rly
from rliable import metrics, plot_utils

# Load data into a dictionary
script_dir = os.path.dirname(__file__)  # absolute dir the script is in
rel_path = "multirun/2026-05-10/16-57-54/all_seed_reward"
data_dir = os.path.join(script_dir, rel_path)
dqn_rewards = {}
seeds = []

for seed in range(5):
    file_path = os.path.join(data_dir, f"seed_{seed}.pkl")
    name = f"Seed_{seed}"
    seeds.append(name)

    with open(file_path, "rb") as f:
        data = pickle.load(f)
        # Assuming 'rewards' is a list/array of the mean rewards saved during training
        rewards = data["rewards"]
        dqn_rewards[name] = np.array([rewards])


# Load ALE scores as a dictionary mapping algorithms to their human normalized
# score matrices, each of which is of size `(num_runs x num_games)`.
aggregate_func = lambda x: np.array(
    [
        metrics.aggregate_median(x),
        metrics.aggregate_iqm(x),
        metrics.aggregate_mean(x),
        metrics.aggregate_optimality_gap(x),
    ]
)
aggregate_scores, aggregate_score_cis = rly.get_interval_estimates(
    dqn_rewards, aggregate_func, reps=50000
)
fig, axes = plot_utils.plot_interval_estimates(
    aggregate_scores,
    aggregate_score_cis,
    metric_names=["Median", "IQM", "Mean", "Optimality Gap"],
    algorithms=seeds,
    xlabel="Human Normalized Score",
)

plt.show()

# TODO see if the plot is correct, then save it
# TODO write down What changes when using RLiable vs. plain averages? Do you feel more confident in the results? Why or why not?
