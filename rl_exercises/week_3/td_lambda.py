from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from rl_exercises.agent import AbstractAgent

State = Any


class TDLambdaAgent(AbstractAgent):
    """TDLambda agent"""

    def __init__(
        self,
        env: gym.Env,
        alpha: float = 0.5,  # Learning rate
        gamma: float = 1.0,  # Discount factor
        lambd: float = 0.5,  # Trace decay
        initial_value: float = 0.5,  # Initial value to init V as described in the paper on page 22 (PDF page 14)
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
        lamd : float, optional
            Trace decay, by default 0.5
        initial_value : float, optional
            Initial value to initialize the weights, by default 0.5
        """
        # Check hyperparameter boundaries
        assert 0 <= gamma <= 1, "Gamma should be in [0, 1]"
        assert alpha > 0, "Learning rate has to be greater than 0"
        assert 0 <= lambd <= 1, "Lamda should be in [0, 1]"
        assert 0 <= initial_value <= 1, "Initial value should be in [0, 1]"

        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.lambd = lambd
        self.initial_value = initial_value

        # number of states → used by V’s default factory
        self.n_states = env.observation_space.n

        # Build V and initialize it with the initial_value
        self.V = np.full(self.n_states, self.initial_value, dtype=float)

        # Eligibility traces as on page 16 of the paper (PDF page 8)
        self.e_traces = np.zeros(self.n_states, dtype=float)

        # For environments with terminal states (like RandomWalk)
        if hasattr(env, "terminal_states"):
            for state in env.terminal_states:
                self.V[state] = 0.0  # Terminal states have the value 0

    def predict_action(
        self, state: np.ndarray, info: dict = {}, evaluate: bool = False
    ) -> Any:  # type: ignore # noqa
        """Predict the action for a given state.
        However, in this scenario we do not predict actions.
        Therefore, we just return 0 here and this is just for completeness.
        """
        return 0, info

    def save(self, path: str) -> Any:  # type: ignore
        """Save the value table V.

        Parameters
        ----------
        path :
            Path to save the value table V

        """
        np.save(path, self.V)  # type: ignore

    def load(self, path) -> Any:  # type: ignore
        """Load the value table V and reset the eligibility traces.

        Parameters
        ----------
        path :
            Path to saved the value table V

        """
        self.V = np.load(path, allow_pickle=True).item()
        self.e_traces = np.zeros(self.n_states, dtype=float)

    def update_agent(self, batch: list) -> float:  # type: ignore
        """Unpack a batch from SimpleBuffer and then update the agent.

        Parameters
        ----------
        batch : list
            List of (state, action, reward, next_state, done, info) tuples

        Returns
        -------
        float
            New value V(s) for the state
        """

        state, _action, reward, next_state, done, _ = batch[0]

        # Convert to the correct type (if not already)
        state = int(state)
        next_state = int(next_state)
        reward = float(reward)

        # Now we update the value according to formula (4)
        # in the paper on page 15 (PDF page 7)
        # For terminal states, V(next_state) = 0
        next_V = 0.0 if done else self.V[next_state]
        delta = reward + self.gamma * next_V - self.V[state]

        # Update eligibility traces (accumulating traces) as described
        # in the paper on page 16 (PDF page 8)
        self.e_traces *= self.gamma * self.lambd
        self.e_traces[state] += 1.0

        # Update all state values
        self.V += self.alpha * delta * self.e_traces

        # Reset eligibility traces at the end of episode
        if done:
            self.e_traces.fill(0.0)

        return float(self.V[state])
