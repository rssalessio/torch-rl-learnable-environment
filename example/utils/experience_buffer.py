# Load packages
from typing import Tuple
import numpy as np
from collections import deque, namedtuple
import random
from numpy.typing import NDArray


Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ExperienceBuffer(object):
    """ Experience replay buffer object """
    capacity: int
    buffer: deque

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def append(self, experience: Experience):
        self.buffer.append(experience)

    def pop(self) -> Experience:
        return self.buffer.pop()

    def sample_all(self) -> Tuple[NDArray[np.float64], NDArray, NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
        return self.sample(len(self.buffer))

    def sample(self, batch_size: int) -> Tuple[NDArray[np.float64], NDArray, NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
        if batch_size > len(self.buffer):
            raise MemoryError('Requested more data than what is available from the buffer!')

        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))

        return states, actions, rewards, next_states, dones
