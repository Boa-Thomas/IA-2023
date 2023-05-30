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

        # define the initial step
        self.current_step = 0

        # Define the maximum number of steps
        self.max_steps = 500

        # Import the grid from a csv file
        self.grid = pd.read_csv('board.csv').values.tolist()

        # Get the dimensions of your grid
        self.n = len(self.grid)

        # Define a 2D observation space
        self.observation_space = spaces.Discrete(self.n * self.n)

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
            y = min(y + 1, self.n - 1)
        elif action == 2:  # down
            x = min(x + 1, self.n - 1)
        elif action == 3:  # left
            y = max(y - 1, 0)

        # If we hit the wall, stay in place
        if self.grid[x][y] == -50:
            x, y = old_x, old_y

        self.current_state = x * (self.n-1) + y  # Update current_state using 2D to 1D conversion

        # Calculate reward
        if self.grid[x][y] == 'End':
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

import numpy as np

# Parameters
alpha_values = np.arange(0, 2, 0.001)
gamma_values = np.arange(0, 2, 0.001)
epsilon_values = np.arange(0, 2, 0.001)
num_episodes = 1500

best_avg_reward = -np.inf
best_params = None

for alpha in alpha_values:
    for gamma in gamma_values:
        for epsilon in epsilon_values:
            Q_table = np.zeros([env.observation_space.n, env.action_space.n])
            total_rewards = []

            for i_episode in range(num_episodes):
                state = env.reset()
                total_reward = 0

                for t in range(500):
                    if np.random.uniform(0, 1) < epsilon:
                        action = env.action_space.sample()
                    else:
                        action = np.argmax(Q_table[state])

                    next_state, reward, done, info = env.step(action)
                    old_value = Q_table[state, action]
                    next_max = np.max(Q_table[next_state])
                    new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                    Q_table[state, action] = new_value
                    state = next_state
                    total_reward += reward

                    if done:
                        break

                total_rewards.append(total_reward)

            avg_reward = np.mean(total_rewards[-50:]) # average of last 50 rewards
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                best_params = {'alpha': alpha, 'gamma': gamma, 'epsilon': epsilon}

print(f'Best average reward: {best_avg_reward} with parameters: {best_params}')

# Now you can run the model again with the best parameters
