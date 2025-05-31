import torch
import torch.nn as nn
import torch.nn.functional as F


# Neural Network for DQN
class DQNNetwork(nn.Module):

  def __init__(self, state_size, action_size, seed = 42):
    super(DQNNetwork, self).__init__()
    self.seed = torch.manual_seed(seed)
    self.fc1 = nn.Linear(state_size, 64)
    self.fc2 = nn.Linear(64, 64)
    self.fc3 = nn.Linear(64, action_size)

  def forward(self, state):
    x = self.fc1(state)
    x = F.relu(x)
    x = self.fc2(x)
    x = F.relu(x)
    return self.fc3(x)


# Neural Network for MPC
class DynamicsNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(DynamicsNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.delta_state_layer = nn.Linear(256, obs_dim)
        self.reward_layer = nn.Linear(256, 1)
        self.ln1 = nn.LayerNorm(256)
        self.ln2 = nn.LayerNorm(256)

    def forward(self, state, action):
        # Ensure action is a 2D tensor for one-hot encoding
        if action.dim() == 1:  # If action is a 1D tensor (batch of actions)
            action = action.unsqueeze(-1)  # Add a dimension to make it (batch_size, 1)
        action_onehot = F.one_hot(action.squeeze(-1), num_classes=4).float()
        x = torch.cat([state, action_onehot], dim=-1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.ln1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.ln2(x)
        delta_state = self.delta_state_layer(x)
        reward = self.reward_layer(x)
        next_state = state + delta_state
        return next_state, reward.squeeze(-1)
