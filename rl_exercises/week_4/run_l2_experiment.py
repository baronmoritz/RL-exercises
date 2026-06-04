import os
import pickle

import gymnasium as gym
from rl_exercises.week_4.dqn import DQNAgent, set_seed


def run_dqn_with_seed(seed: int, num_frames: int = 20000) -> dict:
    """Run DQN with a specific seed and return results."""
    print(f"Running DQN with seed {seed}...")

    env = gym.make("CartPole-v1")
    set_seed(env, seed)

    agent = DQNAgent(
        env=env,
        buffer_capacity=10000,
        batch_size=32,
        lr=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_final=0.01,
        epsilon_decay=500,
        target_update_freq=1000,
        seed=seed,
        hidden_dim=64,
        use_double_dqn=False,
        use_prioritized_replay=False,
    )

    frames, rewards = agent.train(num_frames)

    print(f"  Completed seed {seed}, final reward: {rewards[-1] if rewards else 0:.2f}")

    return {"frames": frames, "rewards": rewards}


def main():
    """Run DQN with 5 different seeds and save results."""
    num_seeds = 5
    num_frames = 20000

    all_results = {}

    for seed in range(num_seeds):
        results = run_dqn_with_seed(seed, num_frames)
        all_results[f"seed_{seed}"] = results

    # Save results
    os.makedirs("rl_exercises/week_4/l2_results", exist_ok=True)

    # Save individual seed results
    for seed_name, results in all_results.items():
        with open(f"rl_exercises/week_4/l2_results/{seed_name}.pkl", "wb") as f:
            pickle.dump(results, f)

    # Save combined results
    with open("rl_exercises/week_4/l2_results/all_seed_reward.pkl", "wb") as f:
        pickle.dump(all_results, f)

    print("\nResults saved to rl_exercises/week_4/l2_results/")
    print(f"Individual seeds: rl_exercises/week_4/l2_results/seed_{seed}.pkl")
    print("Combined: rl_exercises/week_4/l2_results/all_seed_reward.pkl")


if __name__ == "__main__":
    main()
