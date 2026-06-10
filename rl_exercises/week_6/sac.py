from typing import Any, Tuple

import copy
import random

import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import DictConfig
from rl_exercises.agent import AbstractAgent
from torch.distributions import Normal


# Neural Network Definitions
class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(sa))
        x = F.relu(self.fc2(x))
        return self.out(x)


class ValueNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_size: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.out(x)


class GaussianPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.mean_head = nn.Linear(hidden_size, action_dim)
        self.log_std_head = nn.Linear(hidden_size, action_dim)
        self.action_dim = action_dim

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        return mean, log_std

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(state)
        # Clamp log_std to prevent numerical instability
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        # Sample unbounded u
        u = dist.rsample()
        # Apply tanh squashing to constrain actions to (-1, 1)
        action = torch.tanh(u)

        log_prob = dist.log_prob(u) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob


# SAC Agent
class SACAgent(AbstractAgent):
    def __init__(
        self,
        env: gym.Env,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        hidden_size: int = 256,
        buffer_size: int = int(1e6),
        batch_size: int = 256,
        seed: int = 0,
    ) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        env.reset(seed=seed)
        if hasattr(env.action_space, "seed"):
            env.action_space.seed(seed)
        if hasattr(env.observation_space, "seed"):
            env.observation_space.seed(seed)

        self.env = env
        self.seed = seed
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size

        # Assert continuous action space. We want to make sure that a box space is used
        assert isinstance(env.action_space, gym.spaces.Box), (
            "SAC only supports continuous action spaces (gym.spaces.Box)"
        )

        # Dimensions
        self.state_dim = np.prod(env.observation_space.shape)
        self.action_dim = np.prod(env.action_space.shape)

        # Two independent Q networks
        self.q1_net = QNetwork(self.state_dim, self.action_dim, hidden_size)
        self.q2_net = QNetwork(self.state_dim, self.action_dim, hidden_size)

        # Value network
        self.v_net = ValueNetwork(self.state_dim, hidden_size)
        self.v_target = copy.deepcopy(self.v_net)

        # Policy network
        self.policy = GaussianPolicy(self.state_dim, self.action_dim, hidden_size)

        # Separate optimizers for each network
        self.q1_optimizer = optim.Adam(self.q1_net.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.q2_net.parameters(), lr=lr)
        self.v_optimizer = optim.Adam(self.v_net.parameters(), lr=lr)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Replay buffer as numpy arrays for O(1) sampling
        self.buffer_size = buffer_size
        self.replay_buffer = {
            "states": np.zeros((buffer_size, self.state_dim), dtype=np.float32),
            "actions": np.zeros((buffer_size, self.action_dim), dtype=np.float32),
            "rewards": np.zeros(buffer_size, dtype=np.float32),
            "next_states": np.zeros((buffer_size, self.state_dim), dtype=np.float32),
            "dones": np.zeros(buffer_size, dtype=np.float32),
        }
        self.buffer_index = 0
        self.buffer_count = 0

        # Tracking
        self.returns = {}
        self.steps = []

    def predict(self, state: np.ndarray, deterministic: bool = False) -> Any:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            mean, _ = self.policy.forward(state_tensor)
            if deterministic:
                return torch.tanh(mean).squeeze(0).numpy()

            action, _ = self.policy.sample(state_tensor)
            return action.squeeze(0).numpy()

    def update_targets(self) -> None:
        for target, source in zip(self.v_target.parameters(), self.v_net.parameters()):
            target.data.copy_(self.tau * source.data + (1 - self.tau) * target.data)

    def update(self) -> Tuple[float, float, float]:
        if self.buffer_count < self.batch_size:
            return 0.0, 0.0, 0.0

        # Sample batch from replay buffer
        indices = np.random.choice(self.buffer_count, self.batch_size, replace=False)
        states = torch.FloatTensor(self.replay_buffer["states"][indices])
        actions = torch.FloatTensor(self.replay_buffer["actions"][indices])
        rewards = torch.FloatTensor(self.replay_buffer["rewards"][indices]).unsqueeze(1)
        next_states = torch.FloatTensor(self.replay_buffer["next_states"][indices])
        dones = torch.FloatTensor(self.replay_buffer["dones"][indices]).unsqueeze(1)

        # Update V function first
        with torch.no_grad():
            # Sample new actions from current policy for s_t
            new_actions, new_log_pi = self.policy.sample(states)
            # Compute Q values using both networks
            new_q1 = self.q1_net(states, new_actions)
            new_q2 = self.q2_net(states, new_actions)
            # Use minimum of both Q functions
            min_q = torch.min(new_q1, new_q2)
            # Soft value target
            v_target = min_q.detach() - self.alpha * new_log_pi.detach()

        # Value function loss
        v_loss = F.mse_loss(self.v_net(states), v_target)
        # Update V network
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()

        # Update Q-functions
        with torch.no_grad():
            # Compute target Q values
            target_q = rewards + self.gamma * (1 - dones) * self.v_target(next_states)

        # Get current Q estimates from both networks
        current_q1 = self.q1_net(states, actions)
        current_q2 = self.q2_net(states, actions)

        # Q-function losses
        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)

        # Update Q1 network
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        # Update Q2 network
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # Update Policy
        # Freeze Q network gradients for policy update to prevent unnecessary computation
        for p in self.q1_net.parameters():
            p.requires_grad = False
        for p in self.q2_net.parameters():
            p.requires_grad = False

        # Sample actions again
        new_actions, log_pi = self.policy.sample(states)
        new_q1 = self.q1_net(states, new_actions)
        new_q2 = self.q2_net(states, new_actions)
        # Use minimum of both Q functions for policy gradient
        min_q = torch.min(new_q1, new_q2)

        # Policy loss
        policy_loss = (self.alpha * log_pi - min_q).mean()

        # Update policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Unfreeze Q network gradients
        for p in self.q1_net.parameters():
            p.requires_grad = True
        for p in self.q2_net.parameters():
            p.requires_grad = True

        # Update target networks
        self.update_targets()

        return q1_loss.item() + q2_loss.item(), v_loss.item(), policy_loss.item()

    def train(
        self,
        total_steps: int,
        eval_interval: int = 10000,
        eval_episodes: int = 5,
    ) -> None:
        eval_env = gym.make(self.env.spec.id)
        step_count = 0

        state, _ = self.env.reset(seed=self.seed)
        while step_count < total_steps:
            # Collect experience
            action, _ = self.policy.sample(torch.FloatTensor(state).unsqueeze(0))
            action = action.squeeze(0).detach().numpy()  # Remove batch dim

            action = np.clip(
                action, self.env.action_space.low, self.env.action_space.high
            )

            next_state, reward, term, trunc, _ = self.env.step(action)

            # Store in replay buffer
            self.replay_buffer["states"][self.buffer_index] = state
            self.replay_buffer["actions"][self.buffer_index] = action
            self.replay_buffer["rewards"][self.buffer_index] = reward
            self.replay_buffer["next_states"][self.buffer_index] = next_state
            self.replay_buffer["dones"][self.buffer_index] = float(term)
            self.buffer_index = (self.buffer_index + 1) % self.buffer_size
            self.buffer_count = min(self.buffer_count + 1, self.buffer_size)

            # Reset environment if episode ended
            if term or trunc:
                state, _ = self.env.reset()
            else:
                state = next_state
            step_count += 1

            # Update networks
            q_loss, v_loss, p_loss = self.update()

            # Periodic evaluation
            if step_count % eval_interval == 0:
                mean_r, std_r = self.evaluate(eval_env, num_episodes=eval_episodes)
                self.returns[step_count] = mean_r
                self.steps.append(step_count)
                print(
                    f"[Eval ] Step {step_count:6d} AvgReturn {mean_r:5.1f} ± {std_r:4.1f}"
                )

            # Log training progress
            if step_count % 1000 == 0:
                print(
                    f"[Train] Step {step_count:6d} Q Loss {q_loss:.3f} V Loss {v_loss:.3f} Policy Loss {p_loss:.3f}"
                )

        print("Training complete.")

    def evaluate(
        self, eval_env: gym.Env, num_episodes: int = 10
    ) -> Tuple[float, float]:
        returns = []
        for _ in range(num_episodes):
            state, _ = eval_env.reset(seed=self.seed)
            done = False
            total_r = 0.0
            while not done:
                with torch.no_grad():
                    action = self.predict(state, deterministic=True)
                action = np.clip(
                    action, self.env.action_space.low, self.env.action_space.high
                )
                state, r, term, trunc, _ = eval_env.step(action)
                done = term or trunc
                total_r += r
            returns.append(total_r)
        return float(np.mean(returns)), float(np.std(returns))


@hydra.main(config_path="../configs/agent/", config_name="sac", version_base="1.1")
def main(cfg: DictConfig) -> None:
    env = gym.make(cfg.env.name)
    agent = SACAgent(
        env,
        lr=cfg.agent.lr,
        gamma=cfg.agent.gamma,
        tau=cfg.agent.tau,
        alpha=cfg.agent.alpha,
        hidden_size=cfg.agent.hidden_size,
        buffer_size=cfg.agent.buffer_size,
        batch_size=cfg.agent.batch_size,
        seed=cfg.seed,
    )
    agent.train(
        cfg.train.total_steps,
        cfg.train.eval_interval,
        cfg.train.eval_episodes,
    )


if __name__ == "__main__":
    main()
