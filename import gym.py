import gym
from gym import spaces
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class UFSCEvironment(gym.Env):
    def __init__(self):
        super(UFSCEvironment, self).__init__()

        # Define a 2D observation space
        self.observation_space = spaces.Discrete(10*10)

        # Define an action space ranging from 0 to 3: up, right, down, left
        self.action_space = spaces.Discrete(4)

        # Define the initial state
        self.current_state = 0

        # Define the initial step
        self.current_step = 0

        # Define the maximum number of steps
        self.max_steps = 500

        # Import the grid from a csv file
        self.grid = pd.read_csv('board.csv').values.tolist()

    def step(self, action):
        # Check if max steps reached
        if self.current_step >= self.max_steps:
            return self.current_state, -10, True, {}
        self.current_step += 1

        # Define how the agent will step
        x, y = self.current_state // 10, self.current_state % 10

        # Store current position for possible invalid move
        old_x, old_y = x, y

        if action == 0:  # up
            x = max(x - 1, 0)
        elif action == 1:  # right
            y = min(y + 1, 9)
        elif action == 2:  # down
            x = min(x + 1, 9)
        elif action == 3:  # left
            y = max(y - 1, 0)

        # If we hit the wall, stay in place
        if self.grid[x][y] == -50:
            x, y = old_x, old_y

        self.current_state = x + y

        # Calculate reward
        if self.grid[x][y] == 'Aula de IA':
            reward = 50
            done = True
        elif self.grid[x][y] == 'Start':
            reward = -1
            done = False
        elif type(self.grid[x][y]) is int:
            reward = self.grid[x][y]
            done = False
        else:
            reward = -10
            done = False

        return self.current_state, reward, done, {}


    def reset(self):
        # Reset the state to the starting point and current step
        self.current_state = 0
        self.current_step = 0
        return self.current_state

    
    def render(self):
        for row in self.grid:
            print(' | '.join(map(str, row)))
            print('-' * 41)


# Testing the environment
env = UFSCEvironment()
env.render()

# Define the search space for alpha, gamma, and epsilon
alphas = np.linspace(0.1, 1.0, 3)
gammas = np.linspace(0.1, 1.0, 3)
epsilons = np.linspace(0.1, 1.0, 3)

num_episodes = 1000

# Prepare a dictionary to store the total rewards for each parameter combination
results = {}

for alpha in alphas:
    for gamma in gammas:
        for epsilon in epsilons:
            Q_table = np.zeros([env.observation_space.n, env.action_space.n])
            total_rewards = []

            for i_episode in range(num_episodes):
                # Reset state
                state = env.reset()
                total_reward = 0  # Reset the total reward per episode

                for t in range(500):
                    # Choose action. Either explore randomly, or exploit knowledge from Q-table
                    if np.random.uniform(0, 1) < epsilon:
                        action = env.action_space.sample()
                    else:
                        action = np.argmax(Q_table[state]) 

                    next_state, reward, done, info = env.step(action) 

                    old_value = Q_table[state, action]
                    next_max = np.max(Q_table[next_state])

                    # Update Q-value for the current state-action pair
                    new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                    Q_table[state, action] = new_value

                    state = next_state
                    total_reward += reward  # Add reward to total reward

                    if done:
                        break

                total_rewards.append(total_reward)  # Append total reward of the episode to the total_rewards list

            # Store the total rewards for this parameter combination
            results[(alpha, gamma, epsilon)] = total_rewards

# Find the best parameters
best_params = max(results, key=lambda x: sum(results[x])/num_episodes)

print(f"Best parameters are alpha: {best_params[0]}, gamma: {best_params[1]}, epsilon: {best_params[2]}")

# Plot total rewards per episode for the best parameters
plt.plot(results[best_params])
plt.title(f'Total rewards per episode in Q-learning with alpha={best_params[0]}, gamma={best_params[1]}, epsilon={best_params[2]}')
plt.xlabel('Episode')
plt.ylabel('Total reward')
plt.show()
