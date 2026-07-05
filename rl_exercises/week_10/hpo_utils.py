"""
Shared helpers for the Week 10 AutoRL / HPO experiments.
"""

from __future__ import annotations

from typing import Any, Literal

import gymnasium as gym
import numpy as np
from rl_exercises.environments import ContextualMarsRover, MarsRover
from rl_exercises.week_3 import EpsilonGreedyPolicy, TDAgent

# Search space bounds (shared across levels so the studies stay comparable)
ALPHA_RANGE = (1e-3, 1.0)  # Learning rate, sampled on a log scale
EPSILON_RANGE = (0.0, 0.5)  # Exploration rate
GAMMA_RANGE = (0.9, 1.0)  # Discount factor


def reseed_env(env: gym.Env, seed: int) -> None:
    """Reseed the internal RNG of a MarsRover environment.

    Both MarsRover variants keep their own ``np.random.Generator`` in
    ``self.rng`` and (for the contextual variant) ignore the ``seed`` argument
    of ``reset``. Resetting it explicitly makes evaluation reproducible.

    Parameters
    ----------
    env : gym.Env
        Environment (or wrapper) to reseed.
    seed : int
        Seed for the new RNG.
    """
    env.unwrapped.rng = np.random.default_rng(seed)


def make_marsrover(
    seed: int,
    success_prob: float = 0.8,
    rewards: list[float] = [1, 0, 0, 0, 10],
    horizon: int = 10,
) -> MarsRover:
    """Create a stochastic MarsRover environment.

    The transition probabilities are set below 1.0 so that the seed actually
    matters -- this is what makes the Level 1 generalization-across-seeds study
    meaningful.

    Parameters
    ----------
    seed : int
        Seed for the environment RNG.
    success_prob : float, optional
        Probability that an action is executed as intended, by default 0.8.
    rewards : list of float, optional
        Reward per cell, by default [1, 0, 0, 0, 10].
    horizon : int, optional
        Episode length, by default 10.

    Returns
    -------
    MarsRover
        The seeded environment.
    """
    transition_probabilities = np.full((len(rewards), 2), success_prob)
    return MarsRover(
        transition_probabilities=transition_probabilities,
        rewards=rewards,
        horizon=horizon,
        seed=seed,
    )


def make_contextual(
    contexts: list[dict[str, float]],
    seed: int,
    horizon: int = 10,
) -> ContextualMarsRover:
    """Create a ContextualMarsRover restricted to the given contexts.

    Passing a single context isolates one MDP; passing several enables the
    round-robin context switching used for the "global" configuration.

    Parameters
    ----------
    contexts : list of dict
        Context list, each with integer ``goal_pos`` (so the reward fires) and
        a ``friction`` value.
    seed : int
        Seed for the environment RNG.
    horizon : int, optional
        Episode length, by default 10.

    Returns
    -------
    ContextualMarsRover
        The seeded environment.
    """
    env = ContextualMarsRover(horizon=horizon, seed=seed, contexts=contexts)
    reseed_env(env, seed)
    return env


def train_q_agent(
    env: gym.Env,
    alpha: float,
    epsilon: float,
    gamma: float,
    n_steps: int,
    seed: int,
    algorithm: Literal["sarsa", "qlearning"] = "qlearning",
) -> TDAgent:
    """Train a tabular TD agent for a fixed number of environment steps.

    Mirrors the week 3 training loop: predict an action, step the environment,
    and apply a single-transition TD update.

    Parameters
    ----------
    env : gym.Env
        Environment to train on.
    alpha : float
        Learning rate.
    epsilon : float
        Exploration rate for the epsilon-greedy policy.
    gamma : float
        Discount factor.
    n_steps : int
        Number of environment steps to train for.
    seed : int
        Seed for the exploration policy.
    algorithm : Literal["sarsa", "qlearning"], optional
        TD update rule, by default "qlearning".

    Returns
    -------
    TDAgent
        The trained agent.
    """
    policy = EpsilonGreedyPolicy(env, epsilon=epsilon, seed=seed)
    agent = TDAgent(env, policy, alpha=alpha, gamma=gamma, algorithm=algorithm)

    state, info = env.reset()
    for _ in range(n_steps):
        action, info = agent.predict_action(state, info)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        agent.update_agent([(state, action, reward, next_state, done, {})])
        state = next_state
        if done:
            state, info = env.reset()

    return agent


def evaluate_agent(
    env: gym.Env, agent: TDAgent, episodes: int = 50, seed: int = 0
) -> float:
    """Evaluate a trained agent greedily and return the mean episode return.

    Parameters
    ----------
    env : gym.Env
        Environment to evaluate on.
    agent : TDAgent
        Trained agent.
    episodes : int, optional
        Number of evaluation episodes, by default 50.
    seed : int, optional
        Seed used to reset the environment RNG before evaluation, by default 0.

    Returns
    -------
    float
        Mean return over the evaluation episodes.
    """
    reseed_env(env, seed)
    returns: list[float] = []
    for _ in range(episodes):
        state, info = env.reset()
        done = False
        total = 0.0
        while not done:
            action, _ = agent.predict_action(state, info, evaluate=True)
            state, reward, terminated, truncated, info = env.step(action)
            total += float(reward)
            done = terminated or truncated
        returns.append(total)
    return float(np.mean(returns))


def train_and_evaluate(
    make_env: Any,
    alpha: float,
    epsilon: float,
    gamma: float,
    n_steps: int,
    train_seed: int,
    eval_seed: int,
    eval_episodes: int = 50,
) -> float:
    """Convenience wrapper: build an env, train an agent and evaluate it.

    Uses a fresh environment for training and evaluation to keep the two RNG
    streams independent.

    Parameters
    ----------
    make_env : Callable[[int], gym.Env]
        Factory taking a seed and returning an environment.
    alpha, epsilon, gamma : float
        Hyperparameters.
    n_steps : int
        Training budget in environment steps.
    train_seed : int
        Seed for training environment and policy.
    eval_seed : int
        Seed for the evaluation environment.
    eval_episodes : int, optional
        Number of evaluation episodes, by default 50.

    Returns
    -------
    float
        Mean evaluation return.
    """
    train_env = make_env(train_seed)
    agent = train_q_agent(
        train_env, alpha, epsilon, gamma, n_steps=n_steps, seed=train_seed
    )
    eval_env = make_env(eval_seed)
    return evaluate_agent(eval_env, agent, episodes=eval_episodes, seed=eval_seed)
