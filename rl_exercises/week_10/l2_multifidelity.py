"""
Level 2: Multi-fidelity HPO in RL -- when are early scores deceiving?

Deceiving setting
-----------------
We tune tabular Q-Learning on the stochastic MarsRover, but with a search space
that includes very large learning rates (alpha up to 1.0). A large alpha lets
the agent bootstrap towards a good policy very quickly, so after only a few
hundred steps such a configuration often has an excellent evaluation return.
With more training, however, a large learning rate makes the Q-values oscillate
under the environment stochasticity, so the return collapses again. The early
(low-fidelity) score is therefore a poor -- and even misleading -- predictor of
the final (high-fidelity) score.

A multi-fidelity optimizer that prunes trials based on their early performance
can be hurt twice: it keeps deceptive fast-but-unstable configurations, and it
prunes slower configurations that would have been strong at full budget.

We (a) measure the correlation between low- and high-fidelity scores directly
and (b) run Optuna's SuccessiveHalvingPruner and check its optimization history
for exactly this negative influence.
"""

from __future__ import annotations

import csv

import matplotlib.pyplot as plt
import numpy as np
import optuna
from rl_exercises.week_3 import EpsilonGreedyPolicy, TDAgent
from rl_exercises.week_10.hpo_utils import (
    ALPHA_RANGE,
    EPSILON_RANGE,
    GAMMA_RANGE,
    evaluate_agent,
    make_marsrover,
)

# Experiment configuration
FIDELITIES = [500, 1000, 2000, 4000, 8000]  # Cumulative training steps
EVAL_EPISODES = 50
N_CONFIGS = 60  # Random configs for the correlation study
SEEDS = [0, 1, 2]  # Scores are averaged over these seeds
EVAL_SEED_OFFSET = 1000
N_TRIALS = 40  # Trials for the pruned / unpruned Optuna studies
TOP_K = 5  # Size of the "good configuration" set


def fidelity_curve(
    alpha: float,
    epsilon: float,
    gamma: float,
    fidelities: list[int],
    train_seed: int,
    eval_seed: int,
) -> list[float]:
    """Train a single agent and evaluate it at each cumulative fidelity.

    One incremental training pass yields the whole low-to-high fidelity curve,
    which is exactly the information a multi-fidelity optimiser bases its
    pruning decisions on.

    Parameters
    ----------
    alpha, epsilon, gamma : float
        Hyperparameters.
    fidelities : list of int
        Increasing cumulative training-step budgets.
    train_seed : int
        Seed for the training environment and policy.
    eval_seed : int
        Seed for the evaluation environment.

    Returns
    -------
    list of float
        Evaluation return after each fidelity.
    """
    train_env = make_marsrover(train_seed)
    policy = EpsilonGreedyPolicy(train_env, epsilon=epsilon, seed=train_seed)
    agent = TDAgent(train_env, policy, alpha=alpha, gamma=gamma, algorithm="qlearning")
    eval_env = make_marsrover(eval_seed)

    scores: list[float] = []
    prev = 0
    state, info = train_env.reset()
    for budget in fidelities:
        for _ in range(budget - prev):
            action, info = agent.predict_action(state, info)
            next_state, reward, terminated, truncated, info = train_env.step(action)
            done = terminated or truncated
            agent.update_agent([(state, action, reward, next_state, done, {})])
            state = next_state
            if done:
                state, info = train_env.reset()
        prev = budget
        scores.append(evaluate_agent(eval_env, agent, EVAL_EPISODES, eval_seed))
    return scores


def averaged_curve(alpha: float, epsilon: float, gamma: float) -> np.ndarray:
    """Fidelity curve averaged over the seeds for robustness."""
    curves = [
        fidelity_curve(alpha, epsilon, gamma, FIDELITIES, s, s + EVAL_SEED_OFFSET)
        for s in SEEDS
    ]
    return np.mean(curves, axis=0)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation coefficient."""
    rank_x = np.argsort(np.argsort(x))
    rank_y = np.argsort(np.argsort(y))
    return float(np.corrcoef(rank_x, rank_y)[0, 1])


# Part A: How well does the low-fidelity score predict the final one?
print("Level 2: Multi-fidelity HPO -- Part A (fidelity correlation)")
rng = np.random.default_rng(42)
low_scores = np.zeros(N_CONFIGS)
high_scores = np.zeros(N_CONFIGS)
rows = []
for i in range(N_CONFIGS):
    alpha = float(10 ** rng.uniform(np.log10(ALPHA_RANGE[0]), np.log10(ALPHA_RANGE[1])))
    epsilon = float(rng.uniform(*EPSILON_RANGE))
    gamma = float(rng.uniform(*GAMMA_RANGE))
    curve = averaged_curve(alpha, epsilon, gamma)
    low_scores[i] = curve[0]
    high_scores[i] = curve[-1]
    rows.append([i, alpha, epsilon, gamma, *curve])

rho = spearman(low_scores, high_scores)
print(
    f"  Spearman(low fidelity {FIDELITIES[0]} steps, "
    f"high fidelity {FIDELITIES[-1]} steps) = {rho:.3f}"
)

# "Deceptive" configs: good at low fidelity, but not among the best at high
low_topk = set(np.argsort(low_scores)[-TOP_K:])
high_topk = set(np.argsort(high_scores)[-TOP_K:])
deceptive = sorted(low_topk - high_topk)  # would be kept early but are not good
overlooked = sorted(high_topk - low_topk)  # would be pruned early but are good
print(
    f"  Top-{TOP_K} by low fidelity that are NOT top-{TOP_K} at high fidelity "
    f"(deceptive): {deceptive}"
)
print(
    f"  Top-{TOP_K} at high fidelity that would be pruned early "
    f"(overlooked): {overlooked}"
)

# Cost of an aggressive "pick by low fidelity" strategy
low_best = int(np.argmax(low_scores))
high_best = int(np.argmax(high_scores))
aggressive_gap = high_scores[high_best] - high_scores[low_best]
print(
    f"  Config that looks best after {FIDELITIES[0]} steps: #{low_best} "
    f"(low={low_scores[low_best]:.2f}, final={high_scores[low_best]:.2f})"
)
print(
    f"  Config that is actually best after {FIDELITIES[-1]} steps: #{high_best} "
    f"(low={low_scores[high_best]:.2f}, final={high_scores[high_best]:.2f})"
)
print(
    f"  --> Picking purely by low fidelity loses {aggressive_gap:.2f} final "
    f"return ({100 * aggressive_gap / high_scores[high_best]:.0f}%)."
)

# Save the fidelity scores
with open("rl_exercises/week_10/l2_fidelity_scores.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
        ["config", "alpha", "epsilon", "gamma", *[f"f_{b}" for b in FIDELITIES]]
    )
    writer.writerows(rows)
print("Saved l2_fidelity_scores.csv")

# Plot low vs high fidelity
plt.figure(figsize=(8, 6))
plt.scatter(low_scores, high_scores, c="tab:blue", alpha=0.7, label="configs")
for idx in deceptive:
    plt.scatter(low_scores[idx], high_scores[idx], c="red", s=90, marker="x")
plt.scatter([], [], c="red", marker="x", label="deceptive (good early, weak final)")
plt.xlabel(f"Low-fidelity score ({FIDELITIES[0]} steps)")
plt.ylabel(f"High-fidelity score ({FIDELITIES[-1]} steps)")
plt.title(f"Level 2: early scores are deceiving (Spearman ρ = {rho:.2f})")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("rl_exercises/week_10/l2_low_vs_high.pdf")
print("Saved l2_low_vs_high.pdf")


# Part B: does a multi-fidelity optimiser get misled by early scores?
print("\nLevel 2: Part B (SuccessiveHalving vs full-budget Optuna)")
OPT_SEED = 0
EVAL_SEED = OPT_SEED + EVAL_SEED_OFFSET


def sample_params(trial: optuna.Trial) -> tuple[float, float, float]:
    """Sample the three hyperparameters from the shared search space."""
    alpha = trial.suggest_float("alpha", *ALPHA_RANGE, log=True)
    epsilon = trial.suggest_float("epsilon", *EPSILON_RANGE)
    gamma = trial.suggest_float("gamma", *GAMMA_RANGE)
    return alpha, epsilon, gamma


def objective_multifidelity(trial: optuna.Trial) -> float:
    """Report the return at every fidelity so the pruner can stop early."""
    alpha, epsilon, gamma = sample_params(trial)
    train_env = make_marsrover(OPT_SEED)
    policy = EpsilonGreedyPolicy(train_env, epsilon=epsilon, seed=OPT_SEED)
    agent = TDAgent(train_env, policy, alpha=alpha, gamma=gamma, algorithm="qlearning")
    eval_env = make_marsrover(EVAL_SEED)

    score = 0.0
    prev = 0
    state, info = train_env.reset()
    for step_idx, budget in enumerate(FIDELITIES):
        for _ in range(budget - prev):
            action, info = agent.predict_action(state, info)
            next_state, reward, terminated, truncated, info = train_env.step(action)
            done = terminated or truncated
            agent.update_agent([(state, action, reward, next_state, done, {})])
            state = next_state
            if done:
                state, info = train_env.reset()
        prev = budget
        score = evaluate_agent(eval_env, agent, EVAL_EPISODES, EVAL_SEED)
        trial.report(score, step=step_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return score


def full_fidelity_score(alpha: float, epsilon: float, gamma: float) -> float:
    """True score of a config at the highest fidelity (for ground truth)."""
    return float(averaged_curve(alpha, epsilon, gamma)[-1])


# Multi-fidelity study with SuccessiveHalving pruning
pruned_study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.SuccessiveHalvingPruner(min_resource=1, reduction_factor=2),
)
optuna.logging.set_verbosity(optuna.logging.WARNING)
pruned_study.optimize(objective_multifidelity, n_trials=N_TRIALS)

n_pruned = len(pruned_study.get_trials(states=(optuna.trial.TrialState.PRUNED,)))
n_complete = len(pruned_study.get_trials(states=(optuna.trial.TrialState.COMPLETE,)))
best_pruned = pruned_study.best_value
print(f"  SuccessiveHalving: {n_complete} completed, {n_pruned} pruned")
print(f"  Best full-fidelity value found by the pruned study: {best_pruned:.2f}")

# Ground truth: what would the pruned trials have reached at full fidelity?
wrongly_pruned = []
for trial in pruned_study.get_trials(states=(optuna.trial.TrialState.PRUNED,)):
    true_score = full_fidelity_score(
        trial.params["alpha"], trial.params["epsilon"], trial.params["gamma"]
    )
    if true_score > best_pruned:
        wrongly_pruned.append((trial.number, true_score))

print(
    f"  Pruned trials whose TRUE full-fidelity score beats the incumbent "
    f"(wrongly pruned): {len(wrongly_pruned)}"
)
if wrongly_pruned:
    best_missed = max(wrongly_pruned, key=lambda t: t[1])
    print(
        f"    e.g. trial {best_missed[0]} would have reached "
        f"{best_missed[1]:.2f} (> {best_pruned:.2f})"
    )

print("\nDone. See observations_l2.txt for the discussion.")
