import torch
from torch import from_numpy, device
import numpy as np
from models import Actor, Critic
from random_process import OrnsteinUhlenbeckProcess
from memory import Memory
from torch.optim import Adam
from mpi4py import MPI


class Agent:
    def __init__(self, n_states, n_actions, n_goals, action_bounds, capacity, env,
                 k_future,
                 batch_size,
                 action_size=1,
                 tau=0.001,
                 actor_lr=1e-4,
                 critic_lr=1e-3,
                 gamma=0.99):
        self.device = device("cpu")
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_goals = n_goals
        self.k_future = k_future
        self.action_bounds = action_bounds
        self.action_size = action_size
        self.env = env

        self.actor = Actor(self.n_states, n_actions=self.n_actions, n_goals=self.n_goals).to(self.device)
        self.critic = Critic(self.n_states, action_size=self.action_size, n_goals=self.n_goals).to(self.device)
        self.sync_networks(self.actor)
        self.sync_networks(self.critic)
        self.actor_target = Actor(self.n_states, n_actions=self.n_actions, n_goals=self.n_goals).to(self.device)
        self.critic_target = Critic(self.n_states, action_size=self.action_size, n_goals=self.n_goals).to(self.device)
        self.init_target_networks()
        self.training_mode = 1
        self.tau = tau
        self.gamma = gamma

        self.epsilon = 1
        self.epsilon_decay = 0.05
        self.random_process = OrnsteinUhlenbeckProcess()
        self.capacity = capacity
        self.memory = Memory(self.capacity, self.k_future, self.env)

        self.batch_size = batch_size
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.actor_optim = Adam(self.actor.parameters(), self.actor_lr)
        self.critic_optim = Adam(self.critic.parameters(), self.critic_lr)

        self.critic_loss_fn = torch.nn.MSELoss()

    def choose_action(self, state, goal):
        state = np.expand_dims(state, axis=0)
        state = from_numpy(state).float().to(self.device)
        goal = np.expand_dims(goal, axis=0)
        goal = from_numpy(goal).float().to(self.device)

        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state, goal)[0].cpu().data.numpy()
        self.actor.train()

        if self.training_mode:
            action += max(self.epsilon, 0) * self.random_process.sample()
            self.epsilon -= self.epsilon_decay

        action = np.clip(action, self.action_bounds[0], self.action_bounds[1])

        return action

    def reset_randomness(self):
        self.random_process.reset_states()

    def store(self, episode_dict):

        self.memory.add(**episode_dict)

    def init_target_networks(self):
        self.hard_update_networks(self.actor, self.actor_target)
        self.hard_update_networks(self.critic, self.critic_target)

    @staticmethod
    def hard_update_networks(local_model, target_model):
        target_model.load_state_dict(local_model.state_dict())
        target_model.eval()

    @staticmethod
    def soft_update_networks(local_model, target_model, tau=0.05):
        for t_params, e_params in zip(target_model.parameters(), local_model.parameters()):
            t_params.data.copy_(tau * e_params.data + (1 - tau) * t_params.data)

    def train(self):

        states, actions, rewards, next_states, goals = self.memory.sample(self.batch_size)
        states = torch.Tensor(states).to(self.device)
        rewards = torch.Tensor(rewards).to(self.device)
        next_states = torch.Tensor(next_states).to(self.device)
        actions = torch.Tensor(actions).to(self.device)
        goals = torch.Tensor(goals).to(self.device)

        with torch.no_grad():
            target_q = self.critic_target(next_states, goals, self.actor_target(next_states, goals))
            target_returns = rewards + self.gamma * target_q
            target_returns = torch.clamp(target_returns, -1 / (1 - self.gamma), 0).detach()

        q_eval = self.critic(states, goals, actions)
        critic_loss = self.critic_loss_fn(target_returns.view(self.batch_size, 1), q_eval)

        a = self.actor(states, goals)
        actor_loss = -self.critic(states, goals, a)
        actor_loss = actor_loss.mean() + a.pow(2).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.sync_grads(self.actor)
        self.actor_optim.step()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.sync_grads(self.critic)
        self.critic_optim.step()

        return actor_loss, critic_loss

    def save_weights(self):
        torch.save(self.actor.state_dict(), "./actor_weights.pth")
        torch.save(self.critic.state_dict(), "./critic_weights.pth")

    def load_weights(self):
        self.actor.load_state_dict(torch.load("./actor_weights.pth"))
        self.critic.load_state_dict(torch.load("./critic_weights.pth"))

    def set_to_eval_mode(self):
        self.actor.eval()
        self.critic.eval()

    def update_networks(self):
        self.soft_update_networks(self.actor, self.actor_target, self.tau)
        self.soft_update_networks(self.critic, self.critic_target, self.tau)

    @staticmethod
    def sync_networks(network):
        comm = MPI.COMM_WORLD
        flat_params = _get_flat_params_or_grads(network, mode='params')
        comm.Bcast(flat_params, root=0)
        _set_flat_params_or_grads(network, flat_params, mode='params')

    @staticmethod
    def sync_grads(network):
        flat_grads = _get_flat_params_or_grads(network, mode='grads')
        comm = MPI.COMM_WORLD
        global_grads = np.zeros_like(flat_grads)
        comm.Allreduce(flat_grads, global_grads, op=MPI.SUM)
        _set_flat_params_or_grads(network, global_grads, mode='grads')


def _get_flat_params_or_grads(network, mode='params'):
    attr = 'data' if mode == 'params' else 'grad'
    return np.concatenate([getattr(param, attr).cpu().numpy().flatten() for param in network.parameters()])


def _set_flat_params_or_grads(network, flat_params, mode='params'):

    attr = 'data' if mode == 'params' else 'grad'
    pointer = 0
    for param in network.parameters():
        getattr(param, attr).copy_(
            torch.tensor(flat_params[pointer:pointer + param.data.numel()]).view_as(param.data))
        pointer += param.data.numel()
