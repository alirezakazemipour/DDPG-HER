import torch
from torch import from_numpy, device
import numpy as np
from models import Actor, Critic
from random_process import OrnsteinUhlenbeckProcess
from memory import Meomory, Transition
from torch.optim import Adam


class Agent:
    def __init__(self, n_states, action_bounds, capacity,
                 batch_size,
                 action_size=1,
                 tau=0.001,
                 actor_lr=1e-4,
                 critic_lr=1e-3,
                 gamma=0.99):
        self.device = device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_states = n_states
        self.action_bounds = action_bounds
        self.action_size = action_size

        self.actor = Actor(self.n_states).to(self.device)
        self.critic = Critic(self.n_states, action_size=self.action_size).to(self.device)
        self.actor_target = Actor(self.n_states).to(self.device)
        self.critic_target = Critic(self.n_states, action_size=self.action_size).to(self.device)
        # self.actor_target.eval()
        # self.critic_target.eval()
        self.init_target_networks()
        self.training_mode = 1
        self.tau = tau
        self.gamma = gamma

        self.epsilon = 1
        self.epsilon_decay = 0.5
        self.random_process = OrnsteinUhlenbeckProcess()
        self.capacity = capacity
        self.memory = Meomory(self.capacity)

        self.batch_size = batch_size
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.actor_optim = Adam(self.actor.parameters(), self.actor_lr)
        self.critic_optim = Adam(self.critic.parameters(), self.critic_lr, weight_decay=1e-2)

        self.critic_loss_fn = torch.nn.MSELoss()

    def choose_action(self, state):
        state = np.expand_dims(state, axis=0)
        state = from_numpy(state).float().to(self.device)

        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state)[0].cpu().data.numpy()
        self.actor.train()

        if self.training_mode:
            action += max(self.epsilon, 0) * self.random_process.sample()
            self.epsilon -= self.epsilon_decay

        action = np.clip(action, self.action_bounds[0], self.action_bounds[1])

        return action

    def reset_randomness(self):
        self.random_process.reset_states()

    def store(self, state, reward, done, action, next_state):
        state = from_numpy(state).float().to("cpu")
        reward = torch.FloatTensor([reward])
        done = torch.Tensor([done])
        next_state = from_numpy(next_state).float().to("cpu")
        action = from_numpy(action)

        self.memory.add(state, reward, done, action, next_state)

    def init_target_networks(self):
        self.hard_update_networks(self.actor, self.actor_target)
        self.hard_update_networks(self.critic, self.critic_target)

    @staticmethod
    def hard_update_networks(local_model, target_model):
        target_model.load_state_dict(local_model.state_dict())
        target_model.eval()

    @staticmethod
    def soft_update_networks(local_model, target_model, tau=0.01):
        for t_params, e_params in zip(target_model.parameters(), local_model.parameters()):
            t_params.data.copy_(tau * e_params.data + (1 - tau) * t_params.data)

    def unpack_batch(self, batch):

        batch = Transition(*zip(*batch))

        states = torch.cat(batch.state).to(self.device).view(self.batch_size, *self.n_states)
        actions = torch.cat(batch.action).to(self.device).view((-1, 1))
        rewards = torch.cat(batch.reward).to(self.device).view(self.batch_size, 1)
        dones = torch.cat(batch.done).to(self.device).view(self.batch_size, 1)
        next_states = torch.cat(batch.next_state).to(self.device).view(self.batch_size, *self.n_states)

        return states, actions, rewards, dones, next_states

    def train(self):
        if len(self.memory) < self.batch_size:
            return 0, 0
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, dones, next_states = self.unpack_batch(batch)
        with torch.no_grad():
            target_q = self.critic_target(next_states, self.actor_target(next_states))
            target_returns = rewards + self.gamma * target_q * (1.0 - dones)

        q_eval = self.critic(states, actions)
        critic_loss = self.critic_loss_fn(target_returns.view(self.batch_size, 1), q_eval)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        actor_loss = -self.critic(states, self.actor(states))
        actor_loss = actor_loss.mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        self.soft_update_networks(self.actor, self.actor_target)
        self.soft_update_networks(self.critic, self.critic_target)

        return actor_loss, critic_loss
