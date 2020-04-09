import numpy as np
from copy import deepcopy as dc
import random


class Memory:
    def __init__(self, capacity, k_future, env):
        self.capacity = capacity
        self.memory = []
        self.memory_counter = 0
        self.memory_length = 0
        self.env = env

        self.future_p = 1 - (1. / (1 + k_future))

    def sample(self, batch_size):

        ep_indices = np.random.randint(0, len(self.memory), batch_size)
        time_indices = np.random.randint(0, len(self.memory[0]["state"]), batch_size)
        p = np.random.uniform(size=batch_size)
        her_indices = np.where(p < self.future_p)
        regular_indices = np.where(p >= self.future_p)

        future_offset = np.random.uniform(size=batch_size) * (len(self.memory[0]["state"]) - time_indices)
        future_offset = future_offset.astype(int)

        her_episodes = ep_indices[her_indices]
        her_timesteps = time_indices[her_indices]
        her_f_timesteps = (time_indices[her_indices] + future_offset[her_indices])

        states = []
        actions = []
        rewards = []
        next_states = []
        goals = []
        for ep_idx, time_idx, f_offset in zip(her_episodes, her_timesteps, her_f_timesteps):
            desired_goal = dc(self.memory[ep_idx]["achieved_goal"][f_offset])
            reward = self.env.compute_reward(self.memory[ep_idx]["next_achieved_goal"][time_idx], desired_goal,
                                             None)

            states.append(dc(self.memory[ep_idx]["state"][time_idx]))
            actions.append(dc(self.memory[ep_idx]["action"][time_idx]))
            rewards.append(reward)
            next_states.append(dc(self.memory[ep_idx]["next_state"][time_idx]))
            goals.append(desired_goal)

        reg_episodes = ep_indices[regular_indices]
        reg_timesteps = time_indices[regular_indices]
        for ep_idx, time_idx in zip(reg_episodes, reg_timesteps):
            reward = self.env.compute_reward(self.memory[ep_idx]["next_achieved_goal"][time_idx],
                                             self.memory[ep_idx]["desired_goal"][time_idx], None)

            states.append(dc(self.memory[ep_idx]["state"][time_idx]))
            actions.append(dc(self.memory[ep_idx]["action"][time_idx]))
            rewards.append(reward)
            next_states.append(dc(self.memory[ep_idx]["next_state"][time_idx]))
            goals.append(dc(self.memory[ep_idx]["desired_goal"][time_idx]))

        return self.clip_obs(np.vstack(states)), np.vstack(actions), np.vstack(rewards), \
               self.clip_obs(np.vstack(next_states)), self.clip_obs(np.vstack(goals))

    def add(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            self.memory.pop(0)
        assert len(self.memory) <= self.capacity

    def __len__(self):
        return len(self.memory)

    @staticmethod
    def clip_obs(x):
        return np.clip(x, -200, 200)

    def sample_for_normalization(self, batch):
        size = len(batch[0]["state"])
        ep_indices = np.random.randint(0, len(batch), size)
        time_indices = np.random.randint(0, len(batch[0]["state"]), size)
        p = np.random.uniform(size=size)
        her_indices = np.where(p < self.future_p)
        regular_indices = np.where(p >= self.future_p)

        future_offset = np.random.uniform(size=size) * (len(batch[0]["state"]) - time_indices)
        future_offset = future_offset.astype(int)

        her_episodes = ep_indices[her_indices]
        her_timesteps = time_indices[her_indices]
        her_f_timesteps = time_indices[her_indices] + future_offset[her_indices]

        states = []
        goals = []
        for ep_idx, time_idx, f_offset in zip(her_episodes, her_timesteps, her_f_timesteps):
            desired_goal = dc(batch[ep_idx]["achieved_goal"][f_offset])
            states.append(dc(batch[ep_idx]["state"][time_idx]))
            goals.append(desired_goal)

        reg_episodes = ep_indices[regular_indices]
        reg_timesteps = time_indices[regular_indices]
        for ep_idx, time_idx in zip(reg_episodes, reg_timesteps):
            states.append(dc(batch[ep_idx]["state"][time_idx]))
            goals.append(dc(batch[ep_idx]["desired_goal"][time_idx]))

        return self.clip_obs(np.vstack(states)), self.clip_obs(np.vstack(goals))

