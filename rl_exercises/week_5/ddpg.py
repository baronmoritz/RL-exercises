from typing import List, Tuple

import collections
import csv
import random

import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig


def set_seed(env: gym.Env, seed: int = 0) -> None:
    """Seed random number generators for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.reset(seed=seed)
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    if hasattr(env.observation_space, "seed"):
        env.observation_space.seed(seed)


class Actor(nn.Module):
    """
    Deterministic Policy Network mapping states to continuous actions.
    """

    def __init__(
        self, state_dim: int, action_dim: int, hidden_size: int, max_action: float
    ):
        super().__init__()
        self.max_action = max_action

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Tanh(),  # scale to [-1, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # flatten input
        if x.dim() == 1:
            x = x.unsqueeze(0)
        elif x.dim() > 2:
            x = torch.flatten(x, start_dim=1)

        # re-scale output for environment
        return self.max_action * self.net(x)


class Critic(nn.Module):
    """
    Q-Value Network mapping (state, action) pairs to a scalar Q-value.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_size: int):
        super().__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

        self.relu = nn.ReLU()

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)

        # concat state and action before first layer
        xu = torch.cat([state, action], dim=-1)

        x = self.relu(self.fc1(xu))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    """Experience Replay Buffer for Off-Policy learning."""

    def __init__(self, capacity: int = 1_000_000):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)

        return (
            torch.tensor(np.array(state), dtype=torch.float32),
            torch.tensor(np.array(action), dtype=torch.float32),
            torch.tensor(np.array(reward), dtype=torch.float32).unsqueeze(1),
            torch.tensor(np.array(next_state), dtype=torch.float32),
            torch.tensor(np.array(done), dtype=torch.float32).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)


class DDPGAgent:
    """
    DDPG Agent implementing deep deterministic policy gradients with target networks.
    """

    def __init__(
        self,
        env: gym.Env,
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.001,  # Soft-Update Parameter
        hidden_size: int = 256,
        batch_size: int = 64,
        expl_noise: float = 0.1,  # standard deviation for exploration noise
        seed: int = 0,
    ) -> None:
        set_seed(env, seed)
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.expl_noise = expl_noise

        self.state_dim = int(np.prod(env.observation_space.shape))
        self.action_dim = int(np.prod(env.action_space.shape))
        self.max_action = float(env.action_space.high[0])

        # Main-networks for actor and critic
        self.actor = Actor(
            self.state_dim, self.action_dim, hidden_size, self.max_action
        )
        self.critic = Critic(self.state_dim, self.action_dim, hidden_size)

        # Target-networks (copies with frozen weights)
        self.actor_target = Actor(
            self.state_dim, self.action_dim, hidden_size, self.max_action
        )
        self.critic_target = Critic(self.state_dim, self.action_dim, hidden_size)

        # initial sync of weights
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizer adagrad + momentun
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Off-Policy replay buffer
        self.replay_buffer = ReplayBuffer()
        self.total_episodes = 0

    def predict_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """Select action deterministically, optionally adding Gaussian exploration noise."""
        state_t = torch.from_numpy(state).float()

        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_t).cpu().data.numpy().flatten()
        self.actor.train()

        if not evaluate:
            # Add noise so that the agent is forced to explore from time to time
            noise = np.random.normal(
                0, self.max_action * self.expl_noise, size=self.action_dim
            )
            action = (action + noise).clip(-self.max_action, self.max_action)

        return action

    def update_agent(self) -> float:
        """Perform a single step optimization step for Actor and Critic."""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        # 1. Sample a random Mini-Batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # ------------------ Update Critic ------------------
        with torch.no_grad():
            # Calculate next action using the target actor
            next_actions = self.actor_target(next_states)
            # Calculate next Q-value using the target critic
            target_Q = self.critic_target(next_states, next_actions)
            # Use Bellman Equation for TD-target
            target_Q = rewards + ((1.0 - dones) * self.gamma * target_Q)

        # current Q-value from critic
        current_Q = self.critic(states, actions)

        # MSE-Loss for critic
        critic_loss = nn.functional.mse_loss(current_Q, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ------------------ Update Actor ------------------
        # The actor tries to maximize the calculated Q-value -> negative loss
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ------------------ Soft Update of Targets ------------------
        # θ_target = τ * θ + (1 - τ) * θ_target
        for param, target_param in zip(
            self.critic.parameters(), self.critic_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

        for param, target_param in zip(
            self.actor.parameters(), self.actor_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

        return float(critic_loss.item())

    def evaluate(self, eval_env: gym.Env, num_episodes: int = 5) -> Tuple[float, float]:
        """Evaluate policy deterministically over multiple episodes."""
        returns: List[float] = []

        for _ in range(num_episodes):
            state, _ = eval_env.reset()
            episode_return = 0.0
            done = False

            while not done:
                action = self.predict_action(state, evaluate=True)
                next_state, reward, terminated, truncated, _ = eval_env.step(action)
                episode_return += reward
                done = terminated or truncated
                state = next_state

            returns.append(episode_return)

        return float(np.mean(returns)), float(np.std(returns))

    def train(
        self, num_episodes: int, eval_interval: int = 20, eval_episodes: int = 5
    ) -> None:
        """Train the off-policy DDPG agent."""
        # continuous has to be True for continuous action space
        eval_env = gym.make(self.env.spec.id, continuous=True)

        # Log everything into csv for plotting
        csv_filename = "ddpg3.csv"
        csv_headers = [
            "episode",
            "return",
            "mean_critic_loss",
            "mean_return",
            "std_return",
        ]
        with open(csv_filename, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(csv_headers)

        for ep in range(1, num_episodes + 1):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0.0
            critic_losses = []

            while not done:
                # select action using exploration noise
                action = self.predict_action(state, evaluate=False)
                next_state, reward, term, trunc, _ = self.env.step(action)
                done = term or trunc

                # write experience to replay buffer
                self.replay_buffer.push(
                    state, action, float(reward), next_state, float(done)
                )

                # learn after every step
                loss = self.update_agent()
                if loss > 0:
                    critic_losses.append(loss)

                state = next_state
                episode_reward += reward

            self.total_episodes += 1
            mean_loss = np.mean(critic_losses) if critic_losses else 0.0

            eval_mean, eval_std = "", ""

            if ep % 10 == 0:
                print(
                    f"[Train] Ep {ep:3d} Return {episode_reward:5.1f} Critic-Loss {mean_loss:.4f}"
                )

            if ep % eval_interval == 0:
                eval_mean, eval_std = self.evaluate(
                    eval_env, num_episodes=eval_episodes
                )
                print(
                    f"[Eval ] Ep {ep:3d} AvgReturn {eval_mean:5.1f} ± {eval_std:4.1f}"
                )

            with open(csv_filename, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([ep, episode_reward, mean_loss, eval_mean, eval_std])


@hydra.main(config_path="../configs/agent/", config_name="ddpg", version_base="1.1")
def main(cfg: DictConfig) -> None:
    # continuous has to be True, because normally LunarLander is a discrete environment
    env = gym.make(cfg.env.name, continuous=True)
    set_seed(env, cfg.seed)

    agent = DDPGAgent(
        env=env,
        lr_actor=cfg.agent.lr_actor,
        lr_critic=cfg.agent.lr_critic,
        gamma=cfg.agent.gamma,
        tau=cfg.agent.tau,
        hidden_size=cfg.agent.hidden_size,
        batch_size=cfg.agent.batch_size,
        expl_noise=cfg.agent.expl_noise,
        seed=cfg.seed,
    )

    agent.train(
        num_episodes=cfg.train.episodes,
        eval_interval=cfg.train.eval_interval,
        eval_episodes=cfg.train.eval_episodes,
    )


if __name__ == "__main__":
    main()
