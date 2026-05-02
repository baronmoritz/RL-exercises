from __future__ import annotations

from typing import Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class BoundedRandomWalkEnv(gym.Env):
    """Bounded Random Walk environment from section 3.2 of the paper.

    This environment implements the bounded random walk task used to
    demonstrate TD(lambda) in the paper. The walk starts at state D
    (3) and can move left or right with equal probability until it
    reaches a terminal state (A=0 or G=6).

    States: A (0), B (1), C (2), D (3), E (4), F (5), G (6)
    Terminal states: A (0) and G (6)
    Rewards: 0 for A, 1 for G
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self):
        super().__init__()

        # Define the states: A, B, C, D, E, F, G
        self.n_states = 7
        self.states = list(range(self.n_states))

        # Terminal states (A=0, G=6)
        self.terminal_states = {0, 6}

        # Non-terminal states (B, C, D, E, F)
        self.nonterminal_states = [1, 2, 3, 4, 5]

        # Start state (D=3)
        self.start_state = 3

        # Define action space (left, right) even though we do not
        # use it in the task since we don't have Q values
        self.action_space = spaces.Discrete(2)

        # Define observation space (discrete states)
        self.observation_space = spaces.Discrete(self.n_states)

        # Set the start state as the current state
        self.state = self.start_state

        # Define the names of the states for rendering
        self.state_names = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G"}

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> Tuple[int, dict]:
        """Reset the environment to the start state."""

        super().reset(seed=seed)

        # Reset to start state
        self.state = self.start_state

        # Return observation and info
        return self.state, {}

    def step(self, action: int) -> Tuple[int, float, bool, bool, dict]:
        """Take a random step left or right.

        Note: In the prediction task, actions are not used. The walk moves
        randomly regardless of the action.
        """

        # Random walk -> either right or left with a 50% chance each
        if np.random.random() < 0.5:
            self.state -= 1  # Move left
        else:
            self.state += 1  # Move right

        # Check if we have reached a terminal state
        done = self.state in self.terminal_states

        # Reward: 1 for G, 0 otherwise
        reward = 1.0 if self.state == 6 else 0.0

        return self.state, reward, done, False, {}

    def render(self, mode: str = "human") -> None:
        """Render the current state of the environment."""

        if mode == "human":  # Just print the state
            print("Current state:", self.state_names[self.state])
        elif mode == "ansi":  # Print the state and use escape sequences
            print(f"\033[2J\033[HState: {self.state_names[self.state]}")
        else:  # Not implemented alternative mode
            super().render()
