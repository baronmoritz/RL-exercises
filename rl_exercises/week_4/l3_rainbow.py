from typing import Dict, List, Tuple

import os
import pickle

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from rl_exercises.week_4.dqn import DQNAgent, set_seed
from rliable import library as rly
from rliable import metrics, plot_utils


def run_experiment(
    config_name: str,
    num_seeds: int = 5,
    num_frames: int = 50000,
    env_name: str = "CartPole-v1",
    use_double_dqn: bool = False,
    use_prioritized_replay: bool = False,
    per_alpha: float = 0.6,
    per_beta: float = 0.4,
) -> Tuple[List[float], List[float]]:
    """
    Run an experiment with the given configuration across multiple seeds.

    Parameters
    ----------
    config_name : str
        Name of the configuration (for identification).
    num_seeds : int
        Number of random seeds to run.
    num_frames : int
        Number of training frames per run.
    env_name : str
        Gym environment name.
    use_double_dqn : bool
        Whether to use Double DQN.
    use_prioritized_replay : bool
        Whether to use Prioritized Experience Replay.
    per_alpha : float
        Alpha parameter for prioritized replay.
    per_beta : float
        Beta parameter for prioritized replay.

    Returns
    -------
    all_frames : List[float]
        Frames at evaluation points (average across seeds).
    all_rewards : List[float]
        Mean rewards at evaluation points (across seeds).
    """
    all_results = {"frames": [], "rewards": []}

    for seed in range(num_seeds):
        print(f"\nRunning {config_name} with seed {seed}...")

        # Create environment
        env = gym.make(env_name)
        set_seed(env, seed)

        # Create agent
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
            use_double_dqn=use_double_dqn,
            use_prioritized_replay=use_prioritized_replay,
            per_alpha=per_alpha,
            per_beta=per_beta,
        )

        # Train
        frames, rewards = agent.train(num_frames)

        all_results["frames"].append(frames)
        all_results["rewards"].append(rewards)

        print(
            f"  Completed seed {seed}, final avg reward: {rewards[-1] if rewards else 0:.2f}"
        )

    return all_results


def prepare_for_rliable(
    all_configs: Dict[str, Dict[str, List]],
) -> Dict[str, np.ndarray]:
    """
    Prepare data for Rliable analysis.

    Parameters
    ----------
    all_configs : Dict[str, Dict[str, List]]
        Dictionary mapping config names to their results.

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping config names to reward arrays (num_seeds x num_points).
    """
    rliable_data = {}

    # Find the maximum number of evaluation points across all configs
    max_points = 0
    for config_name, results in all_configs.items():
        for seed_data in results["rewards"]:
            max_points = max(max_points, len(seed_data))

    # Pad all reward arrays to the same length
    for config_name, results in all_configs.items():
        padded_rewards = []
        for seed_data in results["rewards"]:
            # Convert to list and ensure it's a flat list
            seed_list = (
                list(seed_data)
                if isinstance(seed_data, (list, tuple, np.ndarray))
                else [seed_data]
            )
            # Pad with the last value if needed
            if len(seed_list) < max_points:
                last_val = seed_list[-1] if seed_list else 0
                padded = seed_list + [last_val] * (max_points - len(seed_list))
            else:
                padded = seed_list[:max_points]
            padded_rewards.append(padded)

        rliable_data[config_name] = np.array(padded_rewards)

    return rliable_data


def main():
    """Run all 4 configurations and generate Rliable plots."""

    # Define configurations
    configs = [
        {"name": "Base DQN", "use_double_dqn": False, "use_prioritized_replay": False},
        {
            "name": "DQN + Prioritized Replay",
            "use_double_dqn": False,
            "use_prioritized_replay": True,
        },
        {
            "name": "DQN + Double DQN",
            "use_double_dqn": True,
            "use_prioritized_replay": False,
        },
        {
            "name": "DQN + Prioritized Replay + Double DQN",
            "use_double_dqn": True,
            "use_prioritized_replay": True,
        },
    ]

    # Run experiments
    all_configs = {}

    for config in configs:
        print(f"\n{'=' * 60}")
        print(f"Running: {config['name']}")
        print(f"{'=' * 60}")

        results = run_experiment(
            config_name=config["name"],
            num_seeds=5,
            num_frames=50000,
            env_name="CartPole-v1",
            use_double_dqn=config["use_double_dqn"],
            use_prioritized_replay=config["use_prioritized_replay"],
        )

        all_configs[config["name"]] = results

    # Save raw data
    os.makedirs("l3_results", exist_ok=True)
    with open("l3_results/all_configs.pkl", "wb") as f:
        pickle.dump(all_configs, f)

    # Prepare data for Rliable
    rliable_data = prepare_for_rliable(all_configs)

    # Define aggregate functions
    aggregate_func = lambda x: np.array(
        [
            metrics.aggregate_median(x),
            metrics.aggregate_iqm(x),
            metrics.aggregate_mean(x),
        ]
    )

    # Compute interval estimates
    aggregate_scores, aggregate_score_cis = rly.get_interval_estimates(
        rliable_data, aggregate_func, reps=50000
    )

    # Plot
    fig, axes = plot_utils.plot_interval_estimates(
        aggregate_scores,
        aggregate_score_cis,
        metric_names=["Median", "IQM", "Mean"],
        algorithms=list(rliable_data.keys()),
        xlabel="Average Reward (10 episodes)",
    )

    plt.tight_layout()
    plt.savefig("l3_results/rainbow_comparison.png")
    plt.close()

    print("\nPlots saved to l3_results/rainbow_comparison.png")

    # Also plot training curves for each configuration
    for config_name, results in all_configs.items():
        plt.figure(figsize=(10, 6))

        # Find max length for this config
        max_len = max(len(r) for r in results["rewards"])

        for seed in range(len(results["frames"])):
            frames = results["frames"][seed]
            rewards = results["rewards"][seed]
            plt.plot(frames, rewards, alpha=0.3, color="blue")

        # Plot mean - handle variable length arrays
        padded_rewards = []
        for r in results["rewards"]:
            if len(r) < max_len:
                padded_rewards.append(r + [r[-1]] * (max_len - len(r)))
            else:
                padded_rewards.append(r[:max_len])
        all_rewards = np.array(padded_rewards)
        mean_rewards = np.mean(all_rewards, axis=0)
        frames = results["frames"][0]
        if len(frames) < max_len:
            frames = frames + [frames[-1] + 100 * (max_len - len(frames))] * (
                max_len - len(frames)
            )
        plt.plot(frames[:max_len], mean_rewards, color="red", linewidth=2, label="Mean")

        plt.xlabel("Frames")
        plt.ylabel("Average Reward (10 episodes)")
        plt.title(f"Training Curve: {config_name}")
        plt.legend()
        plt.grid(True)
        plt.savefig(
            f"l3_results/training_curve_{config_name.replace(' ', '_').replace('+', 'plus')}.png"
        )
        plt.close()

    print("Training curve plots saved to l3_results/")

    # Print final results summary
    print("\n" + "=" * 60)
    print("Summary of final results (mean across seeds)")
    print("=" * 60)
    for config_name, results in all_configs.items():
        final_rewards = [r[-1] if r else 0 for r in results["rewards"]]
        mean_final = np.mean(final_rewards)
        std_final = np.std(final_rewards)
        print(f"{config_name:45s}: {mean_final:6.2f} ± {std_final:5.2f}")


if __name__ == "__main__":
    main()
