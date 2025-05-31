# Deep Reinforcement Learning for Lunar Landing
# In this project, we will train an agent to land a spacecraft on the moon using Deep Q-Learning, Monte Carlo Tree Search, and Model Predictive Control. 
# We will compare the performance of these agents in the Lunar Lander environment from OpenAI Gym.

# Installing Gymnasium

# !pip install gymnasium
# !pip install "gymnasium[atari, accept-rom-license]"
# !apt-get install -y swig
# !pip install gymnasium[box2d]

# Importing the libraries

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
from collections import deque, namedtuple
import gymnasium as gym
from agents import DQNAgent, MCTSAgent, MPCAgent
from utils import show_video_of_model, show_video, deepcopy_env
from constants import *


# Set up the Lunar Lander environment
env = gym.make('LunarLander-v3')
state_shape = env.observation_space.shape
state_size = env.observation_space.shape[0]
number_actions = env.action_space.n
print('State shape: ', state_shape)
print('State size: ', state_size)
print('Number of actions: ', number_actions)

# Choose an agent
agent_type = input('Which agent should run the lunar landing? (DQN, MCTS, MPC): ')

# Mapping agent types to their respective classes
agent_mapping = {
    'DQN': DQNAgent,
    'MCTS': MCTSAgent,
    'MPC': MPCAgent
}

# Initialize the selected agent
if agent_type in agent_mapping:
    if agent_type == 'MCTS':  # MCTS agent requires a copy of the environment for simulations
        agent = agent_mapping[agent_type](state_size, number_actions, env)
    else:
        agent = agent_mapping[agent_type](state_size, number_actions)
else:
    print("Invalid agent type.")
    raise ValueError

## TRAINING THE AGENTS
epsilon = EPSILON_STARTING_VALUE
scores_on_100_episodes = deque(maxlen = 100)

# Global variable to track actions of the most recent episode
last_episode_actions = []

def train_agent(agent, env, agent_type, epsilon, scores_on_100_episodes):
    """Train the agent with the given environment and parameters."""
    global last_episode_actions
    for episode in range(1, NUMBER_EPISODES + 1):
        # Reset environment and score for new episode
        state, _ = env.reset()
        score = 0
        last_episode_actions = []

        # For MCTS agent, reset the environment for simulations and track actions
        if agent_type == 'MCTS':
            agent.env = deepcopy_env(env)
            actions_taken = []

        for t in range(MAXIMUM_NUMBER_TIMESTEPS_PER_EPISODE):
            if agent_type == 'MCTS':
                if actions_taken:
                    action = actions_taken.pop(0)
                else:
                    action, actions_taken = agent.act(state, epsilon)
            else:
                action = agent.act(state, epsilon)

            # Track the action taken
            last_episode_actions.append(action)

            # Step both the environment and the agent, returning the next state, reward, and done flag
            next_state, reward, done, _, _ = env.step(action)
            agent.step(state, action, reward, next_state, done, env)
            state = next_state
            score += reward
            if done:
                break

        scores_on_100_episodes.append(score)
        epsilon = max(EPSILON_ENDING_VALUE, EPSILON_DECAY_VALUE * epsilon)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_on_100_episodes)), end="")
        if episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_on_100_episodes)))
        if np.mean(scores_on_100_episodes) >= 200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode - 100, np.mean(scores_on_100_episodes)))
            torch.save(agent.local_qnetwork.state_dict(), 'checkpoint.pth')
            break

# Train the selected agent
train_agent(agent, env, agent_type, epsilon, scores_on_100_episodes)

# Visualize the results. Deep Learning agents can dynamically run their learned actions, 
# while MCTS agents can simply rerun the best action sequence it found so far.
if agent_type == 'MCTS':
    show_video_of_model(agent, 'LunarLander-v3', last_episode_actions=last_episode_actions)
else:
    show_video_of_model(agent, 'LunarLander-v3')

show_video()