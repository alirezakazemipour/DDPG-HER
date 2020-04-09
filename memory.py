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

        transition = {"states": np.empty((batch_size, self.memory[0]["state"][0].shape[0]), dtype=np.float64),
                      "actions": np.empty((batch_size, self.memory[0]["action"][0].shape[0]), dtype=np.float64),
                      "rewards": np.empty((batch_size, 1), dtype=np.float64),
                      "achieved_goals": np.empty((batch_size, self.memory[0]["achieved_goal"][0].shape[0]),
                                                 dtype=np.float64),
                      "next_states": np.empty((batch_size, self.memory[0]["next_state"][0].shape[0]), dtype=np.float64),
                      "next_achieved_goals": np.empty((batch_size, self.memory[0]["next_achieved_goal"][0].shape[0]),
                                                      dtype=np.float64),
                      "desired_goals": np.empty((batch_size, self.memory[0]["desired_goal"][0].shape[0]),
                                                dtype=np.float64)}
        batch_indices = np.arange(0, batch_size)
        for batch_idx, ep_idx, time_idx in zip(batch_indices, ep_indices, time_indices):
            transition["states"][batch_idx] = dc(self.memory[ep_idx]["state"][time_idx])
            transition["actions"][batch_idx] = dc(self.memory[ep_idx]["action"][time_idx])
            # rewards would be calculated later!
            transition["next_states"][batch_idx] = dc(self.memory[ep_idx]["next_state"][time_idx])
            transition["achieved_goals"][batch_idx] = dc(self.memory[ep_idx]["achieved_goal"][time_idx])
            transition["desired_goals"][batch_idx] = dc(self.memory[ep_idx]["desired_goal"][time_idx])
            transition["next_achieved_goals"][batch_idx] = dc(self.memory[ep_idx]["next_achieved_goal"][time_idx])

        her_indices = np.where(np.random.uniform(size=batch_size) < self.future_p)
        future_offset = np.random.uniform(size=batch_size) * (len(self.memory[0]["state"]) - time_indices)
        future_offset = future_offset.astype(int)
        future_t = (time_indices + future_offset)[her_indices]

        future_achieved_goals = np.empty(
            (ep_indices[her_indices].shape[0], self.memory[0]["next_achieved_goal"][0].shape[0]), dtype=np.float64)
        future_indices = np.arange(0, ep_indices[her_indices].shape[0])
        for f_idx, ep_idx, time_idx in zip(future_indices, ep_indices[her_indices], future_t):
            future_achieved_goals[f_idx] = dc(self.memory[ep_idx]["achieved_goal"][time_idx])
        transition['desired_goals'][her_indices] = future_achieved_goals

        transition['rewards'] = np.expand_dims(self.env.compute_reward(transition['next_achieved_goals'],
                                                                       transition['desired_goals'], None), 1)

        return self.clip_obs(transition["states"]), transition["actions"], transition["rewards"], \
               self.clip_obs(transition["next_states"]), self.clip_obs(transition["desired_goals"])

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

        transition = {"states": np.empty((size, batch[0]["state"][0].shape[0]), dtype=np.float64),
                      "achieved_goals": np.empty((size, batch[0]["achieved_goal"][0].shape[0]), dtype=np.float64),
                      "next_achieved_goals": np.empty((size, batch[0]["next_achieved_goal"][0].shape[0]),
                                                      dtype=np.float64),
                      "desired_goals": np.empty((size, batch[0]["desired_goal"][0].shape[0]), dtype=np.float64)}
        batch_indices = np.arange(0, size)
        for batch_idx, ep_idx, time_idx in zip(batch_indices, ep_indices, time_indices):
            transition["states"][batch_idx] = dc(batch[ep_idx]["state"][time_idx])
            transition["achieved_goals"][batch_idx] = dc(batch[ep_idx]["achieved_goal"][time_idx])
            transition["desired_goals"][batch_idx] = dc(batch[ep_idx]["desired_goal"][time_idx])
            transition["next_achieved_goals"][batch_idx] = dc(batch[ep_idx]["next_achieved_goal"][time_idx])

        her_indices = np.where(np.random.uniform(size=size) < self.future_p)
        future_offset = np.random.uniform(size=size) * (len(batch[0]["state"]) - time_indices)
        future_offset = future_offset.astype(int)
        future_t = (time_indices + future_offset)[her_indices]

        future_achieved_goals = np.empty(
            (ep_indices[her_indices].shape[0], batch[0]["next_achieved_goal"][0].shape[0]), dtype=np.float64)
        future_indices = np.arange(0, ep_indices[her_indices].shape[0])
        for f_idx, ep_idx, time_idx in zip(future_indices, ep_indices[her_indices], future_t):
            future_achieved_goals[f_idx] = dc(batch[ep_idx]["achieved_goal"][time_idx])
        transition['desired_goals'][her_indices] = future_achieved_goals

        return self.clip_obs(np.vstack(transition["states"])), self.clip_obs(np.vstack(transition["desired_goals"]))
