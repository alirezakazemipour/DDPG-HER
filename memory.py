import random
import numpy as np
from collections import namedtuple

Transition = namedtuple("Transition",
                        ("ep_state", "ep_reward", "ep_done", "ep_action", "ep_next_state", "ep_agoal", "ep_dgoal"))


class Memory:
    def __init__(self, capacity, k_future):
        self.capacity = capacity
        # self.batch_size = batch_size
        self.memory = []
        self.memory_counter = 0
        self.__memory_length = 0

        self.future_p = 1 - (1. / (1 + k_future))

    def sample(self, batch_size):
        indices = np.random.randint(0, len(self.memory), batch_size)
        p = np.random.uniform(size=batch_size)
        her_indices = indices[p <= self.future_p]
        total_episode_steps = [len(self.memory[idx][0]) for idx in her_indices]
        her_timesteps = [np.random.randint(0, timestep) for timestep in total_episode_steps]
        future_offsets = np.random.uniform(size=len(her_indices)) * (
                    np.array(total_episode_steps) - np.array(her_timesteps))
        future_offsets = future_offsets.astype(np.int)

        regular_indices = indices[p > self.future_p]
        total_episode_steps = [len(self.memory[idx][0]) for idx in regular_indices]
        regular_timesteps = [np.random.randint(0, timestep) for timestep in total_episode_steps]

        states = []
        actions = []
        rewards = []
        dones = []
        next_states = []
        goals = []
        for ep_idx, timestep, f_offset in zip(her_indices, her_timesteps, future_offsets):
            self.memory[ep_idx][-3][timestep] = self.memory[ep_idx][-4][timestep + f_offset]
            if np.linalg.norm(self.memory[ep_idx][-3][timestep] - self.memory[ep_idx][-1][timestep]) <= 0.05:
                self.memory[ep_idx][2][timestep] = 0
                # print("HER affected")
            else:
                self.memory[ep_idx][2][timestep] = -1
            states.append(self.memory[ep_idx][0][timestep])
            actions.append(self.memory[ep_idx][1][timestep])
            rewards.append(self.memory[ep_idx][2][timestep])
            dones.append(self.memory[ep_idx][3][timestep])
            next_states.append(self.memory[ep_idx][-2][timestep])
            goals.append(self.memory[ep_idx][-3][timestep])

        for ep_idx, timestep in zip(regular_indices, regular_timesteps):
            states.append(self.memory[ep_idx][0][timestep])
            actions.append(self.memory[ep_idx][1][timestep])
            rewards.append(self.memory[ep_idx][2][timestep])
            dones.append(self.memory[ep_idx][3][timestep])
            next_states.append(self.memory[ep_idx][-2][timestep])
            goals.append(self.memory[ep_idx][-3][timestep])

        return np.vstack(states), np.vstack(actions), np.vstack(rewards), np.vstack(dones), np.vstack(next_states), np.vstack(goals)

    def add(self, *transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            self.memory.pop(0)
        assert len(self.memory) <= self.capacity
        self.__update_length__(transition)

    def __len__(self):
        return self.__memory_length

    def __update_length__(self, transition):
        self.__memory_length += len(transition[0][0])
