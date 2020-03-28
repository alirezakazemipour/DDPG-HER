import random

from collections import namedtuple

Transition = namedtuple("Transition",
                        ("state", "reward", "done", "action", "next_state", "goal"))


class Meomory:
    def __init__(self, capacity):
        self.capacity = capacity
        # self.batch_size = batch_size
        self.memory = []
        self.memory_counter = 0

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def add(self, *transition):
        self.memory.append(Transition(*transition))
        if len(self.memory) > self.capacity:
            self.memory.pop(0)
        assert len(self.memory) <= self.capacity

    def __len__(self):
        return len(self.memory)
