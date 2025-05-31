import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
import gymnasium as gym
from torch.autograd import Variable
from collections import deque, namedtuple
from utils import ReplayMemory, deepcopy_env
from concurrent.futures import ThreadPoolExecutor
from constants import *
from networks import *


class BaseAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayMemory(REPLAY_BUFFER_SIZE)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done, env):
        raise NotImplementedError

    def act(self, state, epsilon=0.):
        raise NotImplementedError

    def learn(self, experiences, discount_factor):
        raise NotImplementedError
  
  
# Deep Q-Network Agent
class DQNAgent(BaseAgent):

  def __init__(self, state_size, action_size):
    super(DQNAgent, self).__init__(state_size, action_size)
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.local_qnetwork = DQNNetwork(state_size, action_size).to(self.device)
    self.target_qnetwork = DQNNetwork(state_size, action_size).to(self.device)
    self.optimizer = optim.Adam(self.local_qnetwork.parameters(), lr = LEARNING_RATE)

  def step(self, state, action, reward, next_state, done, _):
    self.memory.push((state, action, reward, next_state, done))
    self.t_step = (self.t_step + 1) % 4
    if self.t_step == 0:
      if len(self.memory.memory) > MINIBATCH_SIZE:
        experiences = self.memory.sample(100)
        self.learn(experiences, DISCOUNT_FACTOR)

  def act(self, state, epsilon = 0.):
    state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
    self.local_qnetwork.eval()
    with torch.no_grad():
      action_values = self.local_qnetwork(state)
    self.local_qnetwork.train()
    if random.random() > epsilon:
      return np.argmax(action_values.cpu().data.numpy())
    else:
      return random.choice(np.arange(self.action_size))

  def learn(self, experiences, discount_factor):
    states, next_states, actions, rewards, dones = experiences
    next_q_targets = self.target_qnetwork(next_states).detach().max(1)[0].unsqueeze(1)
    q_targets = rewards + discount_factor * next_q_targets * (1 - dones)
    q_expected = self.local_qnetwork(states).gather(1, actions)
    loss = F.mse_loss(q_expected, q_targets)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    self.soft_update(self.local_qnetwork, self.target_qnetwork, INTERPOLATION_PARAMETER)

  def soft_update(self, local_model, target_model, interpolation_parameter):
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
      target_param.data.copy_(interpolation_parameter * local_param.data + (1.0 - interpolation_parameter) * target_param.data)


# Monte Carlo Search Tree Agent
class MCTSAgent(BaseAgent):

    def __init__(self, state_size, action_size, env, num_simulations=100):
        super(MCTSAgent, self).__init__(state_size, action_size)
        self.num_simulations = num_simulations
        self.env = deepcopy_env(env)

    def step(self, state, action, reward, next_state, done, env):
        self.memory.push((state, action, reward, next_state, done))
        self.env = deepcopy_env(env)

    def act(self, state, epsilon=0.):
        if random.random() > epsilon:
            return self.monte_carlo_tree_search(state)
        else:
            return random.choice(np.arange(self.action_size)), []

    def monte_carlo_tree_search(self, state):
        root = self.MCT_Node(state=state)

        # Expand the root node and get environment copies
        env_copies = self.expand(root)

        # Run N simulations for each child node
        for action, child in root.children.items():

            def simulate_and_accumulate(_):
                # Create a new environment instance for each thread to avoid race conditions between threads
                env_copy = env_copies[action]
                thread_env = deepcopy_env(env_copy)
                try:
                    return self.simulate(child.state, thread_env)
                finally:
                    # Explicitly delete the environment copies to free memory
                    del thread_env, env_copy

            # Run simulations in parallel for a performance boost
            with ThreadPoolExecutor() as executor:
                results = list(executor.map(simulate_and_accumulate, range(self.num_simulations)))

            # Separate rewards and actions_taken from the results
            rewards, actions_taken_list = zip(*results)

            # If a valid solution was found, return the entire corresponding action list
            for reward, actions_taken in zip(rewards, actions_taken_list):
                if reward > 200:
                    return action, actions_taken

            # Otherwise, backpropagate the result to give the children scores
            self.backprop(child, rewards)

        # If a solution wasn't found, just select the action from the child node with the highest average reward/utility
        max_child = max(root.children.values(), key=lambda c: c.U)
        return max_child.action, []

    def expand(self, node):
        env_copies = {}
        
        for action in range(self.action_size):
            # Create a copy of the environment for each action
            env_copy = deepcopy_env(self.env)
            next_state, _, _ = self.env_step(node.state, action, env_copy)
            child_node = self.MCT_Node(parent=node, state=next_state, action=action)
            node.children[action] = child_node
            env_copies[action] = env_copy  # Store the environment copy
        
        return env_copies

    def simulate(self, state, env):
        current_state = state
        total_reward = 0
        actions_taken = []  # List to store actions taken during the simulation
        for _ in range(self.num_simulations):
            action = random.choice(np.arange(self.action_size))
            actions_taken.append(action)  # Record the action
            next_state, reward, done = self.env_step(current_state, action, env)
            total_reward += reward
            if done:
                break
            current_state = next_state
        return total_reward, actions_taken  # Return total reward and sequence of actions taken

    def backprop(self, node, rewards):
        node.U = np.mean(rewards)

    def env_step(self, state, action, env):
        next_state, reward, done, _, _ = env.step(action)
        return next_state, reward, done

    class MCT_Node:
        def __init__(self, parent=None, state=None, action=None, U=0, N=0):
            self.parent = parent
            self.state = state
            self.action = action
            self.U = U
            self.N = N
            self.children = {}


# Model Predictive Control Agent
class MPCAgent(BaseAgent):
   
    def __init__(self, state_size, action_size):
        super(MPCAgent, self).__init__(state_size, action_size)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = DynamicsNetwork(state_size, action_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        
        self.plan_horizon = MIN_PLAN_HORIZON  # Start with a small plan horizon
        self.total_steps = 0  # Track total steps for horizon updates

    def step(self, state, action, reward, next_state, done, _):
        self.memory.push((state, action, reward, next_state, done))
        self.t_step = (self.t_step + 1) % 4
        if self.t_step == 0:
            if len(self.memory.memory) > MINIBATCH_SIZE:
                experiences = self.memory.sample(100)
                self.learn(experiences, DISCOUNT_FACTOR)
        
        # The horizon should update gradually, upto a maximum value
        if self.plan_horizon < MAX_PLAN_HORIZON:
            self.total_steps += 1
            if self.total_steps % HORIZON_UPDATE_INTERVAL == 0:
                self.plan_horizon = self.plan_horizon + HORIZON_INCREMENT

    def act(self, state, epsilon=0.):
        # Set the NN model to evaluation mode
        self.model.eval()

        # Convert the state to a PyTorch tensor
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        # Initialize action probabilities for each time step in the planning horizon
        action_seqs = []
        rewards = []

        # Turn off gradients calculations to save time and memory (only needed for training)
        with torch.no_grad():
            # Sample action sequences and evaluate their rewards (cross-entropy method)
            for _ in range(HORIZON_SAMPLES):
                # Sample an action sequence based on the current plan horizon
                action_seq = [np.random.choice(self.action_size) for _ in range(self.plan_horizon)]
                action_seqs.append(action_seq)

                total_reward = 0.0
                sim_state = state_tensor.clone()  # Clone the state for simulation

                # Simulate the environment using the sampled action sequence. For each action in the sequence,
                # predict the next state and reward using the dynamics model and accumulate the total reward.
                for a in action_seq:
                    a_tensor = torch.tensor([a], device=self.device)
                    sim_state, r = self.model(sim_state, a_tensor)
                    total_reward += r.item()

                rewards.append(total_reward)

        # Select the first action from the best-performing action sequence
        elite_idx = np.argmax(rewards)
        elite_seq = action_seqs[elite_idx]
        return elite_seq[0]

    def learn(self, experiences, _):
        # Set the NN model to training mode
        self.model.train()

        # Unpack experiences into individual components
        states, next_states, actions, rewards, dones = experiences

        # Convert data to PyTorch tensors for processing
        obs_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
        next_obs_tensor = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.int64, device=self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)

        # Initialize loss function (mean squared error)
        loss_fn = nn.MSELoss()

        # Predict the next states and rewards using the dynamics model
        pred_next, pred_rewards = self.model(obs_tensor, actions_tensor)

        # Compute the loss as the sum of state prediction loss and reward prediction loss
        reward_weight = 5  # reward prediction should be weighted more than state prediction
        loss = loss_fn(pred_next, next_obs_tensor.view_as(pred_next)) + reward_weight * loss_fn(pred_rewards, rewards_tensor.view_as(pred_rewards))

        # Perform backpropagation to update the model parameters
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
