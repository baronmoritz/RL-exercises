from typing import Any, Dict, List, Tuple

import random

import numpy as np
from rl_exercises.agent import AbstractBuffer


class SumTree:
    """
    Sum Tree data structure for efficient sampling and updates.

    Used in Prioritized Experience Replay to sample transitions with probability
    proportional to their priority.

    Tree structure: leaves [capacity-1: 2*capacity-2] store priorities,
    internal nodes [0: capacity-2] store sums of children.
    """

    def __init__(self, capacity: int) -> None:
        """
        Parameters
        ----------
        capacity : int
            Maximum number of transitions to store.
        """
        self.capacity = capacity
        # Tree has 2 * capacity - 1 nodes: internal + leaves
        self.tree = np.zeros(2 * capacity - 1)
        # Leaves start at index capacity - 1
        self.leaf_start = capacity - 1
        # Data stored at each leaf
        self.data = [None] * capacity
        # Current write position (circular)
        self.write = 0
        # Number of entries added so far
        self.n_entries = 0

    def _propagate(self, idx: int, change: float) -> None:
        """Propagate priority change up the tree."""
        while idx > 0:
            idx = (idx - 1) // 2  # Move to parent
            self.tree[idx] += change

    def _retrieve(self, idx: int, cumulative_sum: float) -> int:
        """
        Retrieve the leaf index where the cumulative sum falls.

        Parameters
        ----------
        idx : int
            Current node index.
        cumulative_sum : float
            Target cumulative sum.

        Returns
        -------
        int
            Leaf index where the cumulative sum is located.
        """
        # If we're at a leaf, return it
        if idx >= self.leaf_start:
            return idx

        left = 2 * idx + 1
        right = left + 1

        if cumulative_sum <= self.tree[left]:
            return self._retrieve(left, cumulative_sum)
        else:
            return self._retrieve(right, cumulative_sum - self.tree[left])

    def total(self) -> float:
        """Return the sum of all priorities (root of the tree)."""
        return self.tree[0]

    def add(self, p: float, data: Any) -> None:
        """
        Add a new element with priority p.

        Parameters
        ----------
        p : float
            Priority of the new element.
        data : Any
            Data to store.
        """
        # Leaf index for this write position
        idx = self.leaf_start + self.write

        # Store data
        self.data[self.write] = data

        # Set priority and propagate
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

        # Update write position
        self.write = (self.write + 1) % self.capacity

        # Update entry count
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, pos: int, p: float) -> None:
        """
        Update the priority of an element at position pos.

        Parameters
        ----------
        pos : int
            Circular buffer position (0 to capacity-1).
        p : float
            New priority.
        """
        idx = self.leaf_start + pos
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, cumulative_sum: float) -> Tuple[int, float, Any]:
        """
        Get the element at the given cumulative sum.

        Parameters
        ----------
        cumulative_sum : float
            Value in [0, total()] to sample.

        Returns
        -------
        pos : int
            Circular buffer position (0 to capacity-1).
        priority : float
            Priority of the retrieved element.
        data : Any
            The stored data.
        """
        idx = self._retrieve(0, cumulative_sum)
        pos = idx - self.leaf_start
        priority = self.tree[idx]
        data = self.data[pos]
        return pos, priority, data


class ReplayBuffer(AbstractBuffer):
    """
    Simple FIFO replay buffer.

    Stores tuples of (state, action, reward, next_state, done, info),
    and evicts the oldest when capacity is exceeded.
    """

    def __init__(self, capacity: int) -> None:
        """
        Parameters
        ----------
        capacity : int
            Maximum number of transitions to store.
        """
        super().__init__()
        self.capacity = capacity
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.next_states: List[np.ndarray] = []
        self.dones: List[bool] = []
        self.infos: List[Dict] = []

    def add(
        self,
        state: np.ndarray,
        action: int | float,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        info: dict,
    ) -> None:
        """
        Add a single transition to the buffer.

        If the buffer is full, the oldest transition is removed.

        Parameters
        ----------
        state : np.ndarray
            Observation before action.
        action : int or float
            Action taken.
        reward : float
            Reward received.
        next_state : np.ndarray
            Observation after action.
        done : bool
            Whether episode terminated/truncated.
        info : dict
            Gym info dict (can store extras).
        """
        if len(self.states) >= self.capacity:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.next_states.pop(0)
            self.dones.pop(0)
            self.infos.pop(0)

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.infos.append(info)

    def sample(
        self, batch_size: int = 32
    ) -> List[Tuple[Any, Any, float, Any, bool, Dict]]:
        """
        Uniformly sample a batch of transitions.

        Parameters
        ----------
        batch_size : int
            Number of transitions to sample.

        Returns
        -------
        List of transitions as (state, action, reward, next_state, done, info).
        """
        size = min(
            len(self.states), batch_size
        )  # ensure we dont sample more then the number of experiences we have
        idxs = random.sample(range(len(self.states)), size)  # sample

        return [
            (
                self.states[i],
                self.actions[i],
                self.rewards[i],
                self.next_states[i],
                self.dones[i],
                self.infos[i],
            )
            for i in idxs
        ]

    def __len__(self) -> int:
        """Current number of stored transitions."""
        return len(self.states)


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay Buffer.

    Extends ReplayBuffer to sample transitions with probability proportional
    to their TD error. Uses a SumTree for efficient sampling and priority updates.

    Implements importance sampling correction to account for the bias introduced
    by prioritized sampling.

    Note: This class overrides the parent's storage mechanism (which uses separate
    lists for states, actions, etc.) with a SumTree-based approach for prioritized
    sampling.
    """

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increase: float = 0.001,
        epsilon: float = 1e-6,
    ) -> None:
        """
        Parameters
        ----------
        capacity : int
            Maximum number of transitions to store.
        alpha : float
            How much prioritization is used (0 = uniform, 1 = full prioritization).
        beta : float
            Importance sampling correction exponent (0 = no correction, 1 = full correction).
        beta_increase : float
            How much beta increases each sampling step.
        epsilon : float
            Small constant to ensure all transitions have non-zero priority.
        """
        super().__init__(capacity)
        self.alpha = alpha
        self.beta = beta
        self.beta_increase = beta_increase
        self.epsilon = epsilon

        # Override parent's storage with SumTree-based approach
        self.tree = SumTree(capacity)
        # Current position in the circular buffer
        self.pos = 0
        # Number of transitions stored
        self.size = 0
        # Small value for new transitions
        self.max_priority = 1.0

    def _get_priority(self, td_error: float) -> float:
        """
        Convert TD error to priority.

        Parameters
        ----------
        td_error : float
            The TD error for the transition.

        Returns
        -------
        float
            Priority = (|TD error| + epsilon) ^ alpha
        """
        return (abs(td_error) + self.epsilon) ** self.alpha

    def add(
        self,
        state: np.ndarray,
        action: int | float,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        info: dict,
    ) -> None:
        """
        Add a single transition to the buffer.

        New transitions get the maximum priority initially.

        Parameters
        ----------
        state : np.ndarray
            Observation before action.
        action : int or float
            Action taken.
        reward : float
            Reward received.
        next_state : np.ndarray
            Observation after action.
        done : bool
            Whether episode terminated/truncated.
        info : dict
            Gym info dict (can store extras).
        """
        transition = (state, action, reward, next_state, done, info)

        # Add to tree with max priority
        self.tree.add(self.max_priority, transition)

        # Update position and size
        self.pos = (self.pos + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def update_priority(self, pos: int, td_error: float) -> None:
        """
        Update priority for a transition at position pos.

        Parameters
        ----------
        pos : int
            Position in the circular buffer (0 to capacity-1).
        td_error : float
            TD error for this transition.
        """
        priority = self._get_priority(td_error)
        self.tree.update(pos, priority)
        # Update max priority for new transitions
        self.max_priority = max(self.max_priority, priority)

    def sample(
        self, batch_size: int = 32
    ) -> Tuple[List[Tuple[Any, Any, float, Any, bool, Dict]], List[float], List[int]]:
        """
        Sample a batch of transitions with probability proportional to priority.
        Uses stratified sampling to ensure good coverage across the priority range.

        Returns importance sampling weights for bias correction.

        Parameters
        ----------
        batch_size : int
            Number of transitions to sample.

        Returns
        -------
        batch : List of transitions
            List of (state, action, reward, next_state, done, info) tuples.
        weights : List[float]
            Importance sampling weights for each transition in the batch.
        indices : List[int]
            Positions of sampled transitions in the circular buffer.
        """
        batch = []
        weights = []
        indices = []

        batch_size = min(batch_size, self.size)

        # Stratified sampling: divide the priority range into batch_size segments
        # and sample one transition from each segment. This reduces variance and
        # ensures better coverage across the priority distribution.
        total = max(self.tree.total(), 1e-8)  # Protect against division by zero
        segment_size = total / batch_size

        for i in range(batch_size):
            # Sample uniformly from the i-th segment
            segment_start = i * segment_size
            segment_end = (i + 1) * segment_size
            s = random.uniform(segment_start, segment_end)

            # Get the transition
            pos, priority, transition = self.tree.get(s)

            # Calculate importance sampling weight
            # w_i = (N * p_i)^(-beta) where p_i = priority / total_priority
            prob = priority / total
            weight = (self.size * prob) ** (-self.beta)

            batch.append(transition)
            weights.append(weight)
            indices.append(pos)

        # Normalize weights for numerical stability
        max_weight = max(weights) if weights else 1.0
        weights = [w / max_weight for w in weights]

        # Increase beta over time (up to 1.0)
        self.beta = min(1.0, self.beta + self.beta_increase)

        return batch, weights, indices

    def __len__(self) -> int:
        """Current number of stored transitions."""
        return self.size
