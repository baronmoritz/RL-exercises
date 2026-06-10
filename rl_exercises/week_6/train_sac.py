import os
import pickle

import gymnasium as gym
import hydra
from omegaconf import DictConfig
from rl_exercises.week_6.sac import SACAgent


@hydra.main(config_path="../configs/agent/", config_name="sac", version_base="1.1")
def train_sac(cfg: DictConfig) -> None:
    RESULTS_DIR = os.path.join(hydra.utils.get_original_cwd(), "results", "sac")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    seeds = [0, 892735897, 142312]  # Same seeds as previous tasks
    env_name = cfg.env.name

    for seed in seeds:
        cfg.seed = seed
        env = gym.make(env_name)
        agent = SACAgent(
            env,
            lr=cfg.agent.lr,
            gamma=cfg.agent.gamma,
            tau=cfg.agent.tau,
            alpha=cfg.agent.alpha,
            hidden_size=cfg.agent.hidden_size,
            buffer_size=cfg.agent.buffer_size,
            batch_size=cfg.agent.batch_size,
            seed=seed,
        )
        agent.train(
            cfg.train.total_steps,
            cfg.train.eval_interval,
            cfg.train.eval_episodes,
        )
        # Save results
        save_path = os.path.join(RESULTS_DIR, f"{env_name}_sac_seed{seed}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump({"returns": agent.returns, "steps": agent.steps}, f)


if __name__ == "__main__":
    train_sac()
