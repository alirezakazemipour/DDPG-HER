import torch
from torch import from_numpy, device
import numpy as np
from models import Actor, Critic
from random_process import OrnsteinUhlenbeckProcess


## Dont forget L2 decay !!!

class Agent:
    def __init__(self, n_states, action_bounds):
        self.device = device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_states = n_states
        self.action_bounds = action_bounds
        self.actor = Actor(self.n_states)
        self.training_mode = True

        self.epsilon = 1
        self.epsilon_decay = 0.05
        self.random_process = OrnsteinUhlenbeckProcess()

    def choose_action(self, state):
        state = np.expand_dims(state, axis=0)
        state = from_numpy(state).to(self.device)

        action = self.actor(state)[0]

        if self.training_mode:
            action += max(self.epsilon, 0) * self.random_process.sample()
            self.epsilon -= self.epsilon_decay

        action = torch.clamp(action, self.action_bounds[0, self.action_bounds[1]])

        return action

    def reset_randomness(self):
        self.random_process.reset_states()
