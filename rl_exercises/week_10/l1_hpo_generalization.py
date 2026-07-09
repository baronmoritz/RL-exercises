"""
Level 1: How well does an HPO configuration generalize across seeds?

Generalization task
-------------------
We keep the environment (a stochastic MarsRover) and the algorithm (tabular
Q-Learning) fixed and only vary the random seed. The seed controls the
environment stochasticity, the exploration and the training trajectory, so a
"good" hyperparameter configuration should not be tailored to a single seed.

We optimise the hyperparameters (alpha, epsilon, gamma) with Optuna on a small
set of TRAINING seeds and then measure the very same configurations on a
disjoint set of HELD-OUT TEST seeds. Two quantities summarise the result:

* the generalization gap = train score - test score of the incumbent,
* the Spearman rank correlation between the train and test scores of all
  trials, i.e. does the ranking that HPO optimises transfer to unseen seeds?

Expectation
-----------
Because only the seed changes (same MDP, same algorithm), we expect a small
generalization gap and a high rank correlation: the configuration that is best
on the training seeds should still be among the best on the test seeds.
"""

from __future__ import annotations

import csv

import matplotlib.pyplot as plt
import numpy as np
import optuna
from rl_exercises.week_10.hpo_utils import (
    ALPHA_RANGE,
    EPSILON_RANGE,
    GAMMA_RANGE,
    make_marsrover,
    train_and_evaluate,
)

# Experiment configuration
N_TRIALS = 60
N_STEPS = 5000
EVAL_EPISODES = 50
TRAIN_SEEDS = [0, 1, 2, 3, 4]
TEST_SEEDS = [10, 11, 12, 13, 14]
EVAL_SEED_OFFSET = 1000  # Decouple the evaluation RNG from the training RNG


def evaluate_config_on_seeds(
    alpha: float, epsilon: float, gamma: float, seeds: list[int]
) -> float:
    """Mean evaluation return of a configuration over a set of seeds.

    Parameters
    ----------
    alpha, epsilon, gamma : float
        Hyperparameters to evaluate.
    seeds : list of int
        Seeds to train and evaluate on.

    Returns
    -------
    float
        Mean evaluation return across the seeds.
    """
    returns = [
        train_and_evaluate(
            make_marsrover,
            alpha=alpha,
            epsilon=epsilon,
            gamma=gamma,
            n_steps=N_STEPS,
            train_seed=seed,
            eval_seed=seed + EVAL_SEED_OFFSET,
            eval_episodes=EVAL_EPISODES,
        )
        for seed in seeds
    ]
    return float(np.mean(returns))


def objective(trial: optuna.Trial) -> float:
    """Optuna objective: mean training-seed return of a sampled configuration."""
    alpha = trial.suggest_float("alpha", *ALPHA_RANGE, log=True)
    epsilon = trial.suggest_float("epsilon", *EPSILON_RANGE)
    gamma = trial.suggest_float("gamma", *GAMMA_RANGE)
    return evaluate_config_on_seeds(alpha, epsilon, gamma, TRAIN_SEEDS)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation coefficient (Pearson correlation of ranks).

    Parameters
    ----------
    x, y : np.ndarray
        Paired samples.

    Returns
    -------
    float
        Spearman rank correlation in [-1, 1].
    """
    rank_x = np.argsort(np.argsort(x))
    rank_y = np.argsort(np.argsort(y))
    return float(np.corrcoef(rank_x, rank_y)[0, 1])


# Run the optimization on the training seeds
print("Level 1: HPO generalization across seeds")
sampler = optuna.samplers.TPESampler(seed=42)
study = optuna.create_study(direction="maximize", sampler=sampler)
optuna.logging.set_verbosity(optuna.logging.WARNING)
study.optimize(objective, n_trials=N_TRIALS)

# Re-evaluate every trial on the disjoint test seeds
train_scores = np.zeros(len(study.trials))
test_scores = np.zeros(len(study.trials))
rows = []
for i, trial in enumerate(study.trials):
    alpha = trial.params["alpha"]
    epsilon = trial.params["epsilon"]
    gamma = trial.params["gamma"]
    train_scores[i] = trial.value
    test_scores[i] = evaluate_config_on_seeds(alpha, epsilon, gamma, TEST_SEEDS)
    rows.append([trial.number, alpha, epsilon, gamma, train_scores[i], test_scores[i]])

# Incumbent (best on training seeds) and its test performance
best_idx = int(np.argmax(train_scores))
best = rows[best_idx]
gap = train_scores[best_idx] - test_scores[best_idx]
rho = spearman(train_scores, test_scores)

print(
    f"Best config (on train seeds): alpha={best[1]:.4f}, "
    f"epsilon={best[2]:.4f}, gamma={best[3]:.4f}"
)
print(
    f"  train score = {train_scores[best_idx]:.2f}, "
    f"test score = {test_scores[best_idx]:.2f}, gap = {gap:.2f}"
)
print(f"Spearman rank correlation (train vs test) = {rho:.3f}")

# Save the per-trial results to a CSV
with open("rl_exercises/week_10/l1_trials.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["trial", "alpha", "epsilon", "gamma", "train_score", "test_score"])
    writer.writerows(rows)
print("Saved l1_trials.csv")

# Scatter plot: train score vs test score of every trial
plt.figure(figsize=(8, 6))
plt.scatter(train_scores, test_scores, c="tab:blue", alpha=0.7, label="trials")
plt.scatter(
    train_scores[best_idx],
    test_scores[best_idx],
    c="red",
    s=120,
    marker="*",
    label="incumbent",
)
lims = [0, max(train_scores.max(), test_scores.max()) * 1.05]
plt.plot(lims, lims, "k--", alpha=0.5, label="train = test")
plt.xlabel("Train-seed score")
plt.ylabel("Test-seed score")
plt.title(f"Level 1: HPO generalization across seeds (Spearman ρ = {rho:.2f})")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("rl_exercises/week_10/l1_train_vs_test.pdf")
print("Saved l1_train_vs_test.pdf")
