# ppo.py
"""
On-policy Proximal Policy Optimization (PPO) with GAE, clipped surrogate objective,
value-loss coefficient, and entropy bonus, trained for a total number of environment steps.

Implemented enhancements for task 2:
   - KL divergence early stopping as described in the blog post: https://github.com/openai/spinningup/blob/038665d62d569055401d91856abb287263096178/spinup/algos/pytorch/ppo/ppo.py#L269-L271
   - Adam Learning Rate Annealing as described in the blog post: https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/ppo2.py#L133-L135
"""

from typing import Any, List, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

import os  # noqa: E402
import random  # noqa: E402

import hydra  # noqa: E402
from omegaconf import DictConfig  # noqa: E402
from rl_exercises.agent import AbstractAgent  # noqa: E402
from rl_exercises.week_6.networks import (  # noqa: E402
    Policy,
    ValueNetwork,
)


def set_seed(env: gym.Env, seed: int = 0) -> None:
    env.reset(seed=seed)
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    if hasattr(env.observation_space, "seed"):
        env.observation_space.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class PPOAgent(AbstractAgent):
    def __init__(
        self,
        env: gym.Env,
        lr_actor: float = 5e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        epochs: int = 4,
        batch_size: int = 64,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        seed: int = 0,
        hidden_size: int = 128,
        kl_early_stopping_threshold: float = 0.01,
        enable_kl_early_stopping: bool = True,
        enable_lr_annealing: bool = True,
    ) -> None:
        set_seed(env, seed)
        self.seed = seed
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.kl_early_stopping_threshold = kl_early_stopping_threshold
        self.enable_kl_early_stopping = enable_kl_early_stopping
        self.enable_lr_annealing = enable_lr_annealing
        self.lr_actor_initial = lr_actor
        self.lr_critic_initial = lr_critic

        # networks
        self.policy = Policy(env.observation_space, env.action_space, hidden_size)
        self.value_fn = ValueNetwork(env.observation_space, hidden_size)

        # combined optimizer with separate lr for actor and critic
        self.optimizer = optim.Adam(
            [
                {"params": self.policy.parameters(), "lr": lr_actor},
                {"params": self.value_fn.parameters(), "lr": lr_critic},
            ]
        )

    def predict(
        self, state: np.ndarray
    ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        t = torch.from_numpy(state).float()
        probs = self.policy(t).squeeze(0)
        dist = Categorical(probs)
        action = dist.sample().item()
        return (
            action,
            dist.log_prob(torch.tensor(action)),
            dist.entropy(),
            self.value_fn(t),
        )

    def compute_gae(
        self,
        rewards: List[float],
        values: torch.Tensor,
        next_values: torch.Tensor,  # noqa: F841
        dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: compute advantages using GAE (Hint: replicate the GAE formula from actor critic)
        # Compute deltas: one-step TD errors
        deltas = []
        for t in range(len(rewards)):
            done = dones[t]
            delta = rewards[t] + self.gamma * next_values[t] * (1 - done) - values[t]
            deltas.append(delta)
        deltas = torch.stack(deltas)

        # Accumulate GAE advantages backwards
        advantages = torch.zeros_like(deltas)
        last_advantage = torch.tensor(0.0)
        for t in reversed(range(len(rewards))):
            done = dones[t]
            advantages[t] = (
                deltas[t] + self.gamma * self.gae_lambda * (1 - done) * last_advantage
            )
            last_advantage = advantages[t]

        # Compute returns using advantages and values
        returns = advantages + values.squeeze()

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages.detach(), returns.detach()

    def update(self, trajectory: List[Any]) -> None:
        # unpack trajectory
        states = torch.stack([torch.from_numpy(t[0]).float() for t in trajectory])
        actions = torch.tensor([t[1] for t in trajectory])
        old_logps = torch.stack([t[2] for t in trajectory]).detach()
        entropies = torch.stack([t[3] for t in trajectory]).detach()  # noqa: F841
        rewards = [t[4] for t in trajectory]
        dones = torch.tensor([t[5] for t in trajectory], dtype=torch.float32)

        # TODO: compute values and next_values without gradients
        with torch.no_grad():
            values = self.value_fn(states).squeeze()
            next_values = self.value_fn(
                torch.stack([torch.from_numpy(t[6]).float() for t in trajectory])
            ).squeeze()

        # TODO: compute advantages and returns
        advantages = None  # template placeholder
        returns = None  # template placeholder

        advantages, returns = self.compute_gae(rewards, values, next_values, dones)

        dataset = torch.utils.data.TensorDataset(
            states, actions, old_logps, advantages, returns
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        for _ in range(self.epochs):
            for b_states, b_actions, b_oldlogp, b_adv, b_ret in loader:
                # TODO: compute policy loss, value loss, and entropy loss

                # TODO: compute new log probabilities by sampling actions from the policy distribution
                dist = Categorical(self.policy(b_states))
                new_logp = dist.log_prob(b_actions)

                # Enhancement 1: KL divergence early stopping
                # Early stopping based on approximate KL divergence between old and new policy.
                # Justification: Prevents excessively large policy updates that could destabilize learning.
                # Large KL divergence indicates the new policy is too far from the old one, which may
                # lead to performance collapse. Early stopping improves training stability.
                # Reference: https://github.com/openai/spinningup/blob/038665d62d569055401d91856abb287263096178/spinup/algos/pytorch/ppo/ppo.py#L269-L271
                if self.enable_kl_early_stopping:
                    approx_kl = (
                        b_oldlogp - new_logp
                    ).mean()  # Approximate KL divergence
                    if approx_kl > self.kl_early_stopping_threshold:
                        print(
                            f"[KL Early Stopping] KL={approx_kl:.4f} > target={self.kl_early_stopping_threshold:.4f}"
                        )
                        break

                # TODO: compute the ratio of new log probabilities to old log probabilities
                ratio = torch.exp(new_logp - b_oldlogp)

                # TODO: compute the clipped surrogate loss using the clipped objective
                surrogate1 = ratio * b_adv
                surrogate2 = (
                    torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_adv
                )
                policy_loss = -torch.min(surrogate1, surrogate2).mean()

                # TODO: compute value loss using mean squared error
                values = self.value_fn(b_states).squeeze()
                value_loss = torch.nn.functional.mse_loss(values, b_ret)

                # TODO: compute entropy loss using the distribution's entropy
                entropy_loss = -dist.entropy().mean()

                loss = (
                    policy_loss
                    + self.vf_coef * value_loss
                    + self.ent_coef * entropy_loss
                )
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return policy_loss.item(), value_loss.item(), entropy_loss.item()

    def train(
        self,
        total_steps: int,
        eval_interval: int = 10000,
        eval_episodes: int = 5,
    ) -> None:
        eval_env = gym.make(self.env.spec.id)
        step_count = 0
        while step_count < total_steps:
            state, _ = self.env.reset(seed=self.seed)
            done = False
            trajectory: List[Any] = []

            while not done and step_count < total_steps:
                action, logp, ent, val = self.predict(state)
                next_state, reward, term, trunc, _ = self.env.step(action)
                done = term or trunc
                trajectory.append(
                    (state, action, logp, ent, reward, float(done), next_state)
                )
                state = next_state
                step_count += 1

                if step_count % eval_interval == 0:
                    mean_r, std_r = self.evaluate(eval_env, num_episodes=eval_episodes)
                    print(
                        f"[Eval ] Step {step_count:6d} AvgReturn {mean_r:5.1f} ± {std_r:4.1f}"
                    )

            # Enhancement 2: Adam Learning Rate Annealing
            # Linearly decay learning rate from initial value to 0 over total_steps.
            # Justification: Helps fine-tune the policy as training progresses and improves
            # final performance by reducing step sizes near convergence.
            # Reference: https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/ppo2.py#L133-L135
            if self.enable_lr_annealing:
                progress = step_count / total_steps
                current_lr_actor = self.lr_actor_initial * (1 - progress)
                current_lr_critic = self.lr_critic_initial * (1 - progress)
                for param_group in self.optimizer.param_groups:
                    if "policy" in str(param_group["params"]):
                        param_group["lr"] = current_lr_actor
                    else:
                        param_group["lr"] = current_lr_critic

            # PPO update
            policy_loss, value_loss, entropy_loss = self.update(trajectory)
            total_return = sum(t[4] for t in trajectory)
            print(
                f"[Train] Step {step_count:6d} Return {total_return:5.1f} Policy Loss {policy_loss:.3f} Value Loss {value_loss:.3f} Entropy Loss {entropy_loss:.3f}"
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
                action, _, _, _ = self.predict(state)
                state, r, term, trunc, _ = eval_env.step(action)
                done = term or trunc
                total_r += r
            returns.append(total_r)
        return float(np.mean(returns)), float(np.std(returns))


@hydra.main(config_path="../configs/agent/", config_name="ppo", version_base="1.1")
def main(cfg: DictConfig) -> None:
    env = gym.make(cfg.env.name)
    set_seed(env, cfg.seed)
    agent = PPOAgent(
        env,
        lr_actor=cfg.agent.lr_actor,
        lr_critic=cfg.agent.lr_critic,
        gamma=cfg.agent.gamma,
        gae_lambda=cfg.agent.gae_lambda,
        clip_eps=cfg.agent.clip_eps,
        epochs=cfg.agent.epochs,
        batch_size=cfg.agent.batch_size,
        ent_coef=cfg.agent.ent_coef,
        vf_coef=cfg.agent.vf_coef,
        seed=cfg.seed,
        hidden_size=cfg.agent.hidden_size,
        kl_early_stopping_threshold=cfg.agent.get("kl_early_stopping_threshold", 0.01),
        enable_kl_early_stopping=cfg.agent.get("enable_kl_early_stopping", True),
        enable_lr_annealing=cfg.agent.get("enable_lr_annealing", True),
    )
    agent.train(
        cfg.train.total_steps,
        cfg.train.eval_interval,
        cfg.train.eval_episodes,
    )


if __name__ == "__main__":
    main()
