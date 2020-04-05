import numpy as np
from copy import deepcopy as dc
import random


class Memory:
    def __init__(self, capacity, k_future, env):
        self.capacity = capacity
        self.memory = {"state": [],
                       "action": [],
                       "info": [],
                       "achieved_goal": [],
                       "desired_goal": [],
                       "next_state": [],
                       "next_achieved_goal": []
                       }
        self.memory_counter = 0
        self.memory_length = 0
        self.env = env

        self.future_p = 1 - (1. / (1 + k_future))

    def sample(self, batch_size):
        indices = np.random.randint(0, len(self.memory), batch_size)
        p = np.random.uniform(size=batch_size)
        her_indices = indices[p <= self.future_p]
        total_episode_steps = [len(self.memory[idx]["state"]) for idx in her_indices]
        her_timesteps = [np.random.randint(0, timestep) for timestep in total_episode_steps]
        future_offsets = np.random.uniform(size=len(her_indices)) * \
                         (np.array(total_episode_steps) - np.array(her_timesteps))
        future_offsets = future_offsets.astype(np.int)

        regular_indices = indices[p > self.future_p]
        total_episode_steps = [len(self.memory[idx]["state"]) for idx in regular_indices]
        regular_timesteps = [np.random.randint(0, timestep) for timestep in total_episode_steps]

        states = []
        actions = []
        rewards = []
        next_states = []
        goals = []
        for ep_idx, timestep, f_offset in zip(her_indices, her_timesteps, future_offsets):
            desired_goal = dc(self.memory[ep_idx]["achieved_goal"][timestep + f_offset])
            reward = self.env.compute_reward(self.memory[ep_idx]["next_achieved_goal"][timestep].copy(), desired_goal,
                                             self.memory[ep_idx]["info"][timestep].copy())

            states.append(self.memory[ep_idx]["state"][timestep].copy())
            actions.append(self.memory[ep_idx]["action"][timestep].copy())
            rewards.append(reward)
            next_states.append(self.memory[ep_idx]["next_state"][timestep].copy())
            goals.append(desired_goal)

        for ep_idx, timestep in zip(regular_indices, regular_timesteps):
            reward = self.env.compute_reward(self.memory[ep_idx]["next_achieved_goal"][timestep],
                                             self.memory[ep_idx]["desired_goal"][timestep],
                                             self.memory[ep_idx]["info"][timestep].copy())

            states.append(self.memory[ep_idx]["state"][timestep].copy())
            actions.append(self.memory[ep_idx]["action"][timestep].copy())
            rewards.append(reward)
            next_states.append(self.memory[ep_idx]["next_state"][timestep].copy())
            goals.append(self.memory[ep_idx]["desired_goal"][timestep].copy())

        return self.clip_obs(np.vstack(states)), np.vstack(actions), np.vstack(rewards), \
               self.clip_obs(np.vstack(next_states)), self.clip_obs(np.vstack(goals))

    def add(self, transition):
        self.memory["state"].append(transition["state"])
        self.memory["action"].append(transition["action"])
        self.memory["info"].append(transition["info"])
        self.memory["achieved_goal"].append(transition["achieved_goal"])
        self.memory["desired_goal"].append(transition["desired_goal"])
        self.memory["next_state"].append(transition["next_state"])
        self.memory["next_achieved_goal"].append(transition["next_achieved_goal"])

        if len(self.memory["state"][0]) > self.capacity:
            self.memory.pop(0)
        assert len(self.memory) <= self.capacity

    def __len__(self):
        return len(self.memory)

    @staticmethod
    def clip_obs(x):
        return np.clip(x, -200, 200)
