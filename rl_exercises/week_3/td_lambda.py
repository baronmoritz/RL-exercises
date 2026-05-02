from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from rl_exercises.agent import AbstractAgent

State = Any


class TDLambda(AbstractAgent):
    """TDLambda agent"""

    def __init__(
        self,
        env: gym.Env,
        alpha: float = 0.5,  # Learning rate
        gamma: float = 1.0,  # Discount factor
        lamda: float = 0.5,  # Trace decay
    ) -> None:
        """Initialize the TD agent

        Parameters
        ----------
        env : gym.Env
            Environment for the agent
        alpha : float, optional
            Learning Rate, by default 0.5
        gamma : float, optional
            Discount Factor , by default 1.0
        lamda : float, optional
            Trace decay, by default 0.5
        """
        # Check hyperparameter boundaries
        assert 0 <= gamma <= 1, "Gamma should be in [0, 1]"
        assert alpha > 0, "Learning rate has to be greater than 0"
        assert 0 <= lamda <= 1, "Lamda should be in [0, 1]"

        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.lambda_factor = lamda

        # # number of actions → used by Q’s default factory
        # self.n_actions = env.action_space.n

        # # Build Q so that unseen states map to zero‐vectors
        # self.Q: DefaultDict[Any, np.ndarray] = defaultdict(
        #     lambda: np.zeros(self.n_actions, dtype=float)
        # )

        # self.policy = policy

    def predict_action(
        self, state: np.array, info: dict = {}, evaluate: bool = False
    ) -> Any:  # type: ignore # noqa
        """Predict the action for a given state"""
        raise NotImplementedError

    def save(self, path: str) -> Any:  # type: ignore
        """Save the Q table

        Parameters
        ----------
        path :
            Path to save the Q table

        """
        # np.save(path, dict(self.Q))  # type: ignore

    def load(self, path) -> Any:  # type: ignore
        """Load the Q table

        Parameters
        ----------
        path :
            Path to saved the Q table

        """
        # loaded_q = np.load(path, allow_pickle=True).item()
        # self.Q = defaultdict(
        #     lambda: np.zeros(self.n_actions, dtype=float),
        #     loaded_q,
        # )

    def update_agent(self, batch) -> float:  # type: ignore
        """Unpack a batch from SimpleBuffer and route to the appropriate TD update.

        Parameters
        ----------
        batch : list
            List of (state, action, reward, next_state, done, info) tuples

        Returns
        -------
        float
            New Q value for the state action pair
        """
        state, action, reward, next_state, done, _ = batch[0]
        # if self.algorithm == "sarsa":
        #     # TODO: Get the next action for the lookahead in SARSA using the policy of this agent.

        #     next_action = self.policy(
        #         self.Q, next_state, evaluate=False
        #     )  # Use the policy to select the next action
        #     return self.SARSA(state, action, reward, next_state, next_action, done)
        # else:
        #     return self.Q_Learning(state, action, reward, next_state, done)
