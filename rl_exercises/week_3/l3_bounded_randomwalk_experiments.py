import numpy as np
from rl_exercises.week_3.bounded_randomwalk import BoundedRandomWalkEnv
from rl_exercises.week_3.td_lambda import TDLambdaAgent

# Create the bounded random-walk environment
env = BoundedRandomWalkEnv()

# Create the agent
agent = TDLambdaAgent(env, alpha=0.1, gamma=1.0, lambd=0.3, initial_value=0.5)

# Train
n_episodes = 1000
for episode in range(n_episodes):
    # Reset the state of the environment
    state, _ = env.reset()
    done = False

    # Perform the random-walk until an end state is reached
    while not done:
        # Sample an action since we do not control our actions
        action = env.action_space.sample()

        # Perform a step in the environment
        next_state, reward, done, _, _ = env.step(action)

        # Update the TD-lamda agent with this new state
        batch = [(state, 0, reward, next_state, done, {})]
        agent.update_agent(batch)

        # Change the state to the next state
        state = next_state

# Define the true probabilities for each state as described on
# page 20 (PDF page 12) of the paper (described for B,C,D,E,F)
true_probabilities = np.array([0.0, 1 / 6, 1 / 3, 1 / 2, 2 / 3, 5 / 6, 1.0])

# Print the true and the learned probability for each state
print("State | True P(G) | Learned P(G)")
for s in range(7):
    print(f"{env.state_names[s]:5} | {true_probabilities[s]:9.3f} | {agent.V[s]:9.3f}")
