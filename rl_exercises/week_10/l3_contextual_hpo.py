"""
Level 3: Context & Hyperparameters -- are optimal HPs situational?

This script reproduces the central message of "Hyperparameters in Contextual RL
are Highly Situational" (Eimer et al., 2022, arXiv 2212.10876):
The best hyperparameters depend strongly on the context, so a single
configuration tuned for one context (or averaged over all of them) is often
clearly sub-optimal for the others.

Setting
-------
We use the ContextualMarsRover, in which a context is a (friction, goal_pos)
pair. friction changes the transition dynamics (how often an action slips) and
goal_pos changes where the reward of 10 is placed. We define four contexts that
differ in both features (goal on the right vs. left, high vs. low friction),
each of which is a genuinely different MDP. goal_pos is integer-valued so the
reward actually fires (the reward check compares it to the integer position).

Experiments
-----------
1. Per-context HPO: tune (alpha, epsilon, gamma) separately for every context
   with Optuna and compare the resulting best configurations.
2. Transfer matrix: take the HP tuned on context i, use it to train+solve
   context j, and record the return M[i, j]. Off-diagonal degradation shows
   that HPs do not transfer between contexts.
3. Per-context vs. global: tune a single "global" configuration on all contexts
   in round-robin and compare it, context by context, to the per-context
   optima.
"""

from __future__ import annotations

from typing import Callable

import csv

import matplotlib.pyplot as plt
import numpy as np
import optuna
from rl_exercises.week_10.hpo_utils import (
    ALPHA_RANGE,
    EPSILON_RANGE,
    GAMMA_RANGE,
    make_contextual,
    train_and_evaluate,
)

# Four contexts spanning a difficulty gradient (friction) and both goals.
# friction drives how stochastic the dynamics are; goal_pos flips the task.
CONTEXTS: dict[str, dict[str, float]] = {
    "C1_easy_right": {"friction": 0.9, "goal_pos": 4},
    "C2_medium_left": {"friction": 0.7, "goal_pos": 0},
    "C3_hard_right": {"friction": 0.5, "goal_pos": 4},
    "C4_veryhard_left": {"friction": 0.35, "goal_pos": 0},
}
CONTEXT_NAMES = list(CONTEXTS.keys())

# Experiment configuration
N_STEPS = 8000
N_TRIALS = 30
SEEDS = [0, 1, 2]
EVAL_EPISODES = 50
EVAL_SEED_OFFSET = 1000


def env_factory(contexts: list[dict[str, float]]) -> Callable[[int], object]:
    """Return a seed -> env factory restricted to the given contexts."""
    return lambda seed: make_contextual(contexts, seed)


def score_config(
    contexts: list[dict[str, float]],
    alpha: float,
    epsilon: float,
    gamma: float,
) -> float:
    """Mean return of a configuration on a context set, averaged over seeds.

    The agent is trained and evaluated on the given contexts; for a single
    context this measures how well the configuration solves that specific MDP.

    Parameters
    ----------
    contexts : list of dict
        Contexts the environment cycles through.
    alpha, epsilon, gamma : float
        Hyperparameters.

    Returns
    -------
    float
        Mean evaluation return over the seeds.
    """
    make_env = env_factory(contexts)
    returns = [
        train_and_evaluate(
            make_env,
            alpha=alpha,
            epsilon=epsilon,
            gamma=gamma,
            n_steps=N_STEPS,
            train_seed=seed,
            eval_seed=seed + EVAL_SEED_OFFSET,
            eval_episodes=EVAL_EPISODES,
        )
        for seed in SEEDS
    ]
    return float(np.mean(returns))


def tune(contexts: list[dict[str, float]], study_name: str) -> dict[str, float]:
    """Run an Optuna study and return the best hyperparameters.

    Parameters
    ----------
    contexts : list of dict
        Contexts to tune on.
    study_name : str
        Name used only for logging.

    Returns
    -------
    dict
        Best hyperparameters plus the achieved score under key "score".
    """

    def objective(trial: optuna.Trial) -> float:
        alpha = trial.suggest_float("alpha", *ALPHA_RANGE, log=True)
        epsilon = trial.suggest_float("epsilon", *EPSILON_RANGE)
        gamma = trial.suggest_float("gamma", *GAMMA_RANGE)
        return score_config(contexts, alpha, epsilon, gamma)

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=N_TRIALS)
    best = dict(study.best_params)
    best["score"] = study.best_value
    print(
        f"  [{study_name}] best: alpha={best['alpha']:.4f}, "
        f"epsilon={best['epsilon']:.4f}, gamma={best['gamma']:.4f} "
        f"-> {best['score']:.2f}"
    )
    return best


# Experiment 1: Per-context HPO
print("Level 3: Contextual HPO")
optuna.logging.set_verbosity(optuna.logging.WARNING)

print("Experiment 1: per-context hyperparameter optimization")
per_context_best: dict[str, dict[str, float]] = {}
for name in CONTEXT_NAMES:
    per_context_best[name] = tune([CONTEXTS[name]], name)

# Save the per-context best configurations
with open("rl_exercises/week_10/l3_best_configs.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
        ["context", "friction", "goal_pos", "alpha", "epsilon", "gamma", "tuned_score"]
    )
    for name in CONTEXT_NAMES:
        b = per_context_best[name]
        writer.writerow(
            [
                name,
                CONTEXTS[name]["friction"],
                CONTEXTS[name]["goal_pos"],
                b["alpha"],
                b["epsilon"],
                b["gamma"],
                b["score"],
            ]
        )
print("Saved l3_best_configs.csv")

# Experiment 2: Hyperparameter transfer matrix
print("\nExperiment 2: hyperparameter transfer matrix")
n = len(CONTEXT_NAMES)
transfer = np.zeros((n, n))
for i, src in enumerate(CONTEXT_NAMES):
    hp = per_context_best[src]
    for j, tgt in enumerate(CONTEXT_NAMES):
        transfer[i, j] = score_config(
            [CONTEXTS[tgt]], hp["alpha"], hp["epsilon"], hp["gamma"]
        )
    print(
        f"  HP({src}) applied to all contexts: "
        f"{np.array2string(transfer[i], precision=1)}"
    )

# Save transfer matrix
with open("rl_exercises/week_10/l3_transfer_matrix.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["hp_tuned_on \\ evaluated_on", *CONTEXT_NAMES])
    for i, src in enumerate(CONTEXT_NAMES):
        writer.writerow([src, *[f"{v:.2f}" for v in transfer[i]]])
print("Saved l3_transfer_matrix.csv")

# Average relative degradation when using foreign HPs
# For each target column j the best is the diagonal (HP tuned for j).
degradations = []
for j in range(n):
    best_j = transfer[j, j]
    for i in range(n):
        if i != j and best_j > 0:
            degradations.append(1.0 - transfer[i, j] / best_j)
mean_degradation = float(np.mean(degradations))
print(
    f"  Mean relative return drop when using a foreign context's HPs: "
    f"{100 * mean_degradation:.1f}%"
)

# Experiment 3: per-context tuning vs a single global configuration
print("\nExperiment 3: per-context optima vs one global configuration")
global_best = tune(list(CONTEXTS.values()), "global")
global_per_context = np.array(
    [
        score_config(
            [CONTEXTS[name]],
            global_best["alpha"],
            global_best["epsilon"],
            global_best["gamma"],
        )
        for name in CONTEXT_NAMES
    ]
)
tuned_per_context = np.array([transfer[j, j] for j in range(n)])

print(f"  Per-context tuned mean : {tuned_per_context.mean():.2f}")
print(f"  Global config    mean : {global_per_context.mean():.2f}")
for j, name in enumerate(CONTEXT_NAMES):
    print(
        f"    {name}: tuned={tuned_per_context[j]:.2f}, "
        f"global={global_per_context[j]:.2f}"
    )

# Plot 1: best alpha and epsilon per context
fig, ax1 = plt.subplots(figsize=(9, 6))
x = np.arange(n)
width = 0.35
alphas = [per_context_best[name]["alpha"] for name in CONTEXT_NAMES]
epsilons = [per_context_best[name]["epsilon"] for name in CONTEXT_NAMES]
ax1.bar(x - width / 2, alphas, width, label="best alpha", color="tab:blue")
ax1.set_ylabel("best alpha", color="tab:blue")
ax1.set_xticks(x)
ax1.set_xticklabels(CONTEXT_NAMES, rotation=20, ha="right")
ax2 = ax1.twinx()
ax2.bar(x + width / 2, epsilons, width, label="best epsilon", color="tab:orange")
ax2.set_ylabel("best epsilon", color="tab:orange")
ax1.set_title("Level 3: best hyperparameters differ per context")
fig.tight_layout()
fig.savefig("rl_exercises/week_10/l3_best_hp_per_context.pdf")
print("\nSaved l3_best_hp_per_context.pdf")

# Plot 2: transfer heatmap
plt.figure(figsize=(8, 6.5))
im = plt.imshow(transfer, cmap="viridis", aspect="auto")
plt.colorbar(im, label="evaluation return")
plt.xticks(np.arange(n), CONTEXT_NAMES, rotation=20, ha="right")
plt.yticks(np.arange(n), CONTEXT_NAMES)
plt.xlabel("evaluated on context")
plt.ylabel("hyperparameters tuned on context")
for i in range(n):
    for j in range(n):
        plt.text(
            j,
            i,
            f"{transfer[i, j]:.1f}",
            ha="center",
            va="center",
            color="white" if transfer[i, j] < transfer.max() / 2 else "black",
        )
plt.title("Level 3: hyperparameter transfer matrix")
plt.tight_layout()
plt.savefig("rl_exercises/week_10/l3_transfer_heatmap.pdf")
print("Saved l3_transfer_heatmap.pdf")
