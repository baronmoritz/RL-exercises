import matplotlib.pyplot as plt
import numpy as np
from rl_exercises.week_3.bounded_randomwalk import BoundedRandomWalkEnv
from rl_exercises.week_3.td_lambda import TDLambdaAgent

# Define the configurations as described in the paper
lambda_values = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]


def generate_training_set(env, n_sequences=10):
    """Generate a training set of n_sequences observation-outcome sequences.
    This matches the paper's description: "100 training sets, each consisting
    of 10 sequences" as described on page 20 (PDF page 12).
    """

    training_set = []
    for _ in range(n_sequences):
        # Generate one sequence (episode)

        # Reset the environment
        state, _ = env.reset()
        done = False

        sequence = []
        # Random-walk until done
        while not done:
            # Sample an action since we do not control our actions
            action = env.action_space.sample()

            # Perform a step in the environment
            next_state, reward, done, _, _ = env.step(action)

            # Update the TD-lamda agent with this new state
            sequence.append((state, action, reward, next_state, done, {}))

            # Change the state to the next state
            state = next_state

        training_set.append(sequence)

    return training_set


def experiment1_repeated_presentations(
    lambda_values,
    n_training_sets=100,
    n_sequences_per_set=10,
    max_iterations=1000,
    alpha=0.01,
):
    """Run the experiment with a different value for lambda each
    (figure 3 in the paper).
    """

    rms_errors = np.zeros(len(lambda_values))  # Init the errors with zeros

    # Define the true probabilities for each state as described on
    # page 20 (PDF page 12) of the paper (described for B,C,D,E,F)
    true_probabilities = np.array([0.0, 1 / 6, 1 / 3, 1 / 2, 2 / 3, 5 / 6, 1.0])

    # Loop over the possible values for lambda
    for i, lambd in enumerate(lambda_values):
        trial_errors = np.zeros(n_training_sets)

        # Train as often as there are trainings sets (100 as in the paper)
        for trial in range(n_training_sets):
            # Create the bounded random-walk environment
            env = BoundedRandomWalkEnv()

            # Create a new agent with the lamda value
            agent = TDLambdaAgent(
                env, alpha=alpha, gamma=1.0, lambd=lambd, initial_value=0.5
            )

            # Create a training set
            training_set = generate_training_set(env, n_sequences=n_sequences_per_set)

            # Repeated until convergence
            for _ in range(max_iterations):
                # Reset the accumulated updates
                agent.reset_accumulated_updates()
                converged = True

                # Run all the sequences of the training set
                for sequence in training_set:
                    # The Egilibility Traces have to be reset to 0
                    # for each sequence. Otherwise, the errors explode
                    agent.e_traces.fill(0.0)

                    for transition in sequence:
                        # Batch update
                        agent.update_agent([transition], accumulate=True)

                # Check for convergence (delta w is nearly 0)
                if np.max(np.abs(agent.accumulated_delta_w)) > 1e-4:
                    agent.apply_accumulated_updates()
                    converged = False

                if converged:
                    break

            # Calculate the RMS error
            trial_errors[trial] = agent.rms_error(true_probabilities)

        # Mean RMS error over all trials
        rms_errors[i] = np.mean(trial_errors)

    return rms_errors


def experiment2_single_presentation(
    lambda_values, alpha_values, n_training_sets=100, n_sequences_per_set=10
):
    """
    Reproduces experiment 2: Single Presentation (Figure 4).
    """

    true_probabilities = np.array([0.0, 1 / 6, 1 / 3, 1 / 2, 2 / 3, 5 / 6, 1.0])

    # Result matrix: rows -> lambdas, columns -> alphas
    results = np.full((len(lambda_values), len(alpha_values)), np.nan)

    for i, lambd in enumerate(lambda_values):
        for j, alpha in enumerate(alpha_values):
            trial_errors = []

            for _ in range(n_training_sets):
                # Generate the environment and the agent
                env = BoundedRandomWalkEnv()
                agent = TDLambdaAgent(
                    env, alpha=alpha, gamma=1.0, lambd=lambd, initial_value=0.5
                )

                training_set = generate_training_set(
                    env, n_sequences=n_sequences_per_set
                )

                # Loop over the sequences
                for sequence in training_set:
                    agent.e_traces.fill(0.0)
                    for transition in sequence:
                        agent.update_agent([transition], accumulate=False)

                # Calculate the mean squared error
                rms_error = agent.rms_error(true_probabilities)
                trial_errors.append(rms_error)

            # Average the rms errors over the number of training sets
            results[i, j] = np.mean(trial_errors)

    return results


# Experiment 1: Repeated Presentations
print("Experiment 1: Repeated Presentations (Figure 3)")
rms_errors_repeated = experiment1_repeated_presentations(
    lambda_values=lambda_values,
    n_training_sets=100,
    n_sequences_per_set=10,
    max_iterations=1000,
    alpha=0.01,
)
for lambd, error in zip(lambda_values, rms_errors_repeated):
    print(f"λ={lambd:.1f}: RMS Error = {error:.4f}")

# Now, we recreate the plot from the paper
plt.figure(figsize=(10, 6))
plt.plot(
    lambda_values, rms_errors_repeated, "o-", linewidth=2, markersize=8, label="TD(λ)"
)

# Special mark for λ=1 (Widrow-Hoff)
plt.plot(1.0, rms_errors_repeated[-1], "ro", markersize=10, label="Widrow-Hoff (λ=1)")
plt.xlabel("λ", fontsize=12)
plt.ylabel(f"Error using best alpha ({0.01})", fontsize=12)
plt.title("Reproduction Sutton (1988) - Figure 3", fontsize=14)
plt.xticks(lambda_values)
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend()

# Save as PDF
plt.tight_layout()
plt.savefig("rl_exercises/week_3/Sutton_experiment1_figure3.pdf")


# Experiment 2: Single Representation
print("\nExperiment 2: Single Representation (Figure 4)")
alpha_values = np.linspace(0.00001, 0.6, 13)
rms_errors_repeated = experiment2_single_presentation(
    lambda_values=lambda_values,
    alpha_values=alpha_values,
    n_training_sets=100,
    n_sequences_per_set=10,
)

plt.figure(figsize=(10, 7))

for i, lambd in enumerate(lambda_values):
    # Remove nan values (divergence)
    valid_indices = ~np.isnan(rms_errors_repeated[i])
    plt.plot(
        alpha_values[valid_indices],
        rms_errors_repeated[i][valid_indices],
        "o-",
        label=f"λ = {lambd}",
    )

plt.ylim(0.0, 0.7)
plt.xlim(0.0, 0.6)
plt.xlabel("alpha")
plt.ylabel("ERROR")
plt.title("Reproduction Sutton (1988) - Figure 4")
plt.legend(loc="upper left")
plt.grid(True, alpha=0.3)
plt.savefig("rl_exercises/week_3/Sutton_experiment1_figure4.pdf")
