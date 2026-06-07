# rl_exercises/week_6/train_ppo.py
import os
import pickle

import gymnasium as gym
import hydra
from omegaconf import DictConfig
from rl_exercises.week_6.ppo import PPOAgent


@hydra.main(config_path="../configs/agent/", config_name="ppo", version_base="1.1")
def train_ppo(cfg: DictConfig) -> None:
    seeds = [0, 892735897, 142312]
    env_name = cfg.env.name
    RESULTS_DIR = os.path.join(hydra.utils.get_original_cwd(), "results", "ppo")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for seed in seeds:
        # PPO enhanced
        cfg.seed = seed
        env = gym.make(env_name)
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
            seed=seed,
            hidden_size=cfg.agent.hidden_size,
            enable_kl_early_stopping=True,
            enable_lr_annealing=True,
        )
        agent.train(
            cfg.train.total_steps,
            eval_interval=cfg.train.eval_interval,
            eval_episodes=cfg.train.eval_episodes,
        )
        save_path = os.path.join(RESULTS_DIR, f"{env_name}_ppo_enhanced_seed{seed}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump({"returns": agent.returns, "steps": agent.steps}, f)

        # PPO vanilla
        agent_vanilla = PPOAgent(
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
            seed=seed,
            hidden_size=cfg.agent.hidden_size,
            enable_kl_early_stopping=False,
            enable_lr_annealing=False,
        )
        agent_vanilla.train(
            cfg.train.total_steps,
            eval_interval=cfg.train.eval_interval,
            eval_episodes=cfg.train.eval_episodes,
        )

        save_path_vanilla = os.path.join(
            RESULTS_DIR, f"{env_name}_ppo_vanilla_seed{seed}.pkl"
        )

        with open(save_path_vanilla, "wb") as f:
            pickle.dump(
                {"returns": agent_vanilla.returns, "steps": agent_vanilla.steps}, f
            )


if __name__ == "__main__":
    train_ppo()
