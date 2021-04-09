import torch
import torch.nn as nn
import torch.nn.functional as F


class DDPG_Policy_Network(nn.Module):
    """
    Simple FC network using only a linear feedforward network with Relu activation
    with tanh activation after the final layer.
    Linear layers to be defined as input
    """
    def __init__(self, obs_space_size, action_space_size, seed, hidden_layers):
        super(DDPG_Policy_Network, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.layers = nn.ModuleList([nn.Linear(obs_space_size, hidden_layers[0])])
        self.layers.extend([nn.Linear(hidden_layers[i-1], hidden_layers[i]) for i in range(1, len(hidden_layers))])
        self.layers.append(nn.Linear(hidden_layers[-1], action_space_size))

    def forward(self, state):
        x = F.relu(self.layers[0](state))
        for i in range(1, len(self.layers)-1):
            x = F.relu(self.layers[i](x))
        return F.tanh(self.layers[-1](x))


class DDPG_Value_Network(nn.Module):
    """
    Simple FC network using only a linear feedforward network with Relu activation
    with tanh activation after the final layer.
    Linear layers to be defined as input
    """
    def __init__(self, obs_space_size, action_space_size, seed, hidden_layers):
        super(DDPG_Value_Network, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.layers = nn.ModuleList([nn.Linear(obs_space_size, hidden_layers[0])])
        hidden_layers[0] += action_space_size
        self.layers.extend([nn.Linear(hidden_layers[i-1], hidden_layers[i]) for i in range(1, len(hidden_layers))])
        self.layers.append(nn.Linear(hidden_layers[-1], 1))

    def forward(self, state, action):
        x = F.relu(self.layers[0](state))
        for i in range(1, len(self.layers)-1):
            if i == 1:
                x = F.relu(self.layers[i](torch.cat([x, action], dim=1)))
            else:
                x = F.relu(self.layers[i](x))
        return self.layers[-1](x)
