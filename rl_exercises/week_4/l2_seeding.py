import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from rliable import library as rly
from rliable import metrics, plot_utils


def load_results(data_dir: str = "rl_exercises/week_4/l2_results") -> dict:
    """Load results from saved pickle files."""
    all_results = {}

    # Try to load from all_seed_reward.pkl first
    all_file = os.path.join(data_dir, "all_seed_reward.pkl")
    if os.path.exists(all_file):
        with open(all_file, "rb") as f:
            all_results = pickle.load(f)
        print(f"Loaded combined results from {all_file}")
        return all_results

    # Otherwise, try to load individual seed files
    for seed in range(5):
        file_path = os.path.join(data_dir, f"seed_{seed}.pkl")
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                data = pickle.load(f)
            all_results[f"Seed_{seed}"] = data
        else:
            print(f"Warning: {file_path} not found")

    return all_results


def prepare_rliable_data(all_results: dict) -> dict:
    """Prepare data for RLiable analysis.

    RLiable expects data in format: {algorithm_name: array(num_seeds, num_points)}
    """
    # Find the maximum number of evaluation points
    max_points = 0
    for config_name, results in all_results.items():
        if "rewards" in results:
            max_points = max(max_points, len(results["rewards"]))

    # Pad all reward arrays to the same length
    rliable_data = {}
    for config_name, results in all_results.items():
        if "rewards" in results:
            rewards = results["rewards"]
            if len(rewards) < max_points:
                padded = rewards + [rewards[-1]] * (max_points - len(rewards))
            else:
                padded = rewards[:max_points]
            rliable_data[config_name] = np.array([padded])

    return rliable_data


def plot_rliable_analysis(
    rliable_data: dict, output_dir: str = "rl_exercises/week_4/l2_results"
) -> None:
    """Plot RLiable analysis of the results."""
    os.makedirs(output_dir, exist_ok=True)

    # Define aggregate functions
    aggregate_func = lambda x: np.array(
        [
            metrics.aggregate_median(x),
            metrics.aggregate_iqm(x),
            metrics.aggregate_mean(x),
            metrics.aggregate_optimality_gap(x),
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
        metric_names=["Median", "IQM", "Mean", "Optimality Gap"],
        algorithms=list(rliable_data.keys()),
        xlabel="Average Reward",
    )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dqn_rliable_analysis.png"))
    plt.close()

    print(f"RLiable analysis plot saved to {output_dir}/dqn_rliable_analysis.png")


def plot_training_curves(
    all_results: dict, output_dir: str = "rl_exercises/week_4/l2_results"
) -> None:
    """Plot individual training curves for each seed."""
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(12, 6))

    for seed_name, results in all_results.items():
        if "frames" in results and "rewards" in results:
            frames = results["frames"]
            rewards = results["rewards"]
            plt.plot(frames, rewards, alpha=0.7, label=seed_name)

    plt.xlabel("Frames")
    plt.ylabel("Average Reward (10 episodes)")
    plt.title("DQN Training Curves Across 5 Seeds")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "dqn_training_curves.png"))
    plt.close()

    print(f"Training curves plot saved to {output_dir}/dqn_training_curves.png")


def plot_mean_with_std(
    all_results: dict, output_dir: str = "rl_exercises/week_4/l2_results"
) -> None:
    """Plot mean training curve with standard deviation."""
    os.makedirs(output_dir, exist_ok=True)

    # Collect all rewards
    all_rewards = []
    max_len = 0
    for seed_name, results in all_results.items():
        if "rewards" in results:
            rewards = results["rewards"]
            all_rewards.append(rewards)
            max_len = max(max_len, len(rewards))

    # Pad rewards
    padded_rewards = []
    for rewards in all_rewards:
        if len(rewards) < max_len:
            padded_rewards.append(rewards + [rewards[-1]] * (max_len - len(rewards)))
        else:
            padded_rewards.append(rewards[:max_len])

    # Compute mean and std
    mean_rewards = np.mean(padded_rewards, axis=0)
    std_rewards = np.std(padded_rewards, axis=0)

    # Use frames from first seed
    frames = all_results[list(all_results.keys())[0]]["frames"]
    if len(frames) < max_len:
        frames = frames + [frames[-1] + 100 * (max_len - len(frames))] * (
            max_len - len(frames)
        )

    plt.figure(figsize=(12, 6))
    plt.fill_between(
        frames[:max_len],
        mean_rewards - std_rewards,
        mean_rewards + std_rewards,
        alpha=0.2,
    )
    plt.plot(frames[:max_len], mean_rewards, linewidth=2, color="red", label="Mean")
    plt.xlabel("Frames")
    plt.ylabel("Average Reward (10 episodes)")
    plt.title("DQN Training: Mean ± Std Dev Across 5 Seeds")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "dqn_mean_std.png"))
    plt.close()

    print(f"Mean ± Std plot saved to {output_dir}/dqn_mean_std.png")


def main():
    """Main function to load data, compute metrics, and plot results."""
    print("Loading results...")
    all_results = load_results()

    if not all_results:
        print("No results found. Please run run_l2_experiment.py first.")
        return

    print(f"Loaded {len(all_results)} seed results")

    # Prepare data for RLiable
    rliable_data = prepare_rliable_data(all_results)

    # Plot RLiable analysis
    plot_rliable_analysis(rliable_data)

    # Plot training curves
    plot_training_curves(all_results)

    # Plot mean with std
    plot_mean_with_std(all_results)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY OF FINAL RESULTS")
    print("=" * 60)
    final_rewards = []
    for seed_name, results in all_results.items():
        if "rewards" in results and results["rewards"]:
            final_rewards.append(results["rewards"][-1])

    if final_rewards:
        mean_final = np.mean(final_rewards)
        std_final = np.std(final_rewards)
        min_final = np.min(final_rewards)
        max_final = np.max(final_rewards)

        print(f"Mean final reward: {mean_final:.2f}")
        print(f"Std dev: {std_final:.2f}")
        print(f"Min: {min_final:.2f}")
        print(f"Max: {max_final:.2f}")
        print(f"Range: {max_final - min_final:.2f}")

    print("\nPlots saved to rl_exercises/week_4/l2_results/")
    print(" - dqn_rliable_analysis.png (RLiable metrics)")
    print(" - dqn_training_curves.png (Individual curves)")
    print(" - dqn_mean_std.png (Mean with standard deviation)")


if __name__ == "__main__":
    main()
