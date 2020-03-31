from torch import nn
import torch
from torch.nn import functional as F
import numpy as np


def init_weights_biases(size):
    v = 1.0 / np.sqrt(size[0])
    return torch.FloatTensor(size).uniform_(-v, v)


class Actor(nn.Module):
    def __init__(self, n_states, n_actions, n_goals, n_hidden1=256, n_hidden2=256, n_hidden3=256, initial_w=3e-3):
        self.n_states = n_states[0]
        self.n_actions = n_actions
        self.n_goals = n_goals
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.n_hidden3 = n_hidden3
        self.initial_w = initial_w
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(in_features=self.n_states + self.n_goals, out_features=self.n_hidden1)
        self.fc2 = nn.Linear(in_features=self.n_hidden1, out_features=self.n_hidden2)
        self.fc3 = nn.Linear(in_features=self.n_hidden2, out_features=self.n_hidden3)
        self.output = nn.Linear(in_features=self.n_hidden3, out_features=self.n_actions)
        self.tanh = nn.Tanh()

        # self.fc1.weight.data = init_weights_biases(self.fc1.weight.data.size())
        # self.fc1.bias.data = init_weights_biases(self.fc1.bias.data.size())
        # self.fc2.weight.data = init_weights_biases(self.fc2.weight.data.size())
        # self.fc2.bias.data = init_weights_biases(self.fc2.bias.data.size())
        # self.fc3.weight.data = init_weights_biases(self.fc3.weight.data.size())
        # self.fc3.bias.data = init_weights_biases(self.fc3.bias.data.size())
        #
        # self.output.weight.data.uniform_(-self.initial_w, self.initial_w)
        # self.output.bias.data.uniform_(-self.initial_w, self.initial_w)

    def forward(self, x, g):
        x = F.relu(self.fc1(torch.cat([x, g], dim=-1)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        output = self.tanh(self.output(x))  # TODO add scale of the action

        return output


class Critic(nn.Module):
    def __init__(self, n_states, n_goals, n_hidden1=256, n_hidden2=256, n_hidden3=256, initial_w=3e-3, action_size=1):
        self.n_states = n_states[0]
        self.n_goals = n_goals
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.n_hidden3 = n_hidden3
        self.initial_w = initial_w
        self.action_size = action_size
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(in_features=self.n_states + self.n_goals + self.action_size, out_features=self.n_hidden1)
        self.fc2 = nn.Linear(in_features=self.n_hidden1 , out_features=self.n_hidden2)
        self.fc3 = nn.Linear(in_features=self.n_hidden2, out_features=self.n_hidden3)
        self.output = nn.Linear(in_features=self.n_hidden3, out_features=1)

        # self.fc1.weight.data = init_weights_biases(self.fc1.weight.data.size())
        # self.fc1.bias.data = init_weights_biases(self.fc1.bias.data.size())
        # self.fc2.weight.data = init_weights_biases(self.fc2.weight.data.size())
        # self.fc2.bias.data = init_weights_biases(self.fc2.bias.data.size())
        # self.fc3.weight.data = init_weights_biases(self.fc3.weight.data.size())
        # self.fc3.bias.data = init_weights_biases(self.fc3.bias.data.size())

        # self.output.weight.data.uniform_(-self.initial_w, self.initial_w)
        # self.output.bias.data.uniform_(-self.initial_w, self.initial_w)

    def forward(self, x, g, a):
        x = F.relu(self.fc1(torch.cat([x, g, a], dim=-1)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        output = self.output(x)

        return output
