"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
from gym import spaces, logger
import numpy as np
from learnable_environment.learnable_environment import LearnableEnvironment, StateType, ActionType


class MountainCarContinuousLearnableEnvironment(LearnableEnvironment):
    goal_position: float = 0.45
    goal_velocity: float = 0
    min_position: float = -1.2
    max_position: float = 0.6
    max_speed: float = 0.07
    min_action: float = -1.0
    max_action: float = 1.0
    low = np.array([min_position, -max_speed], dtype=np.float32)
    high = np.array([max_position, max_speed], dtype=np.float32)
    action_space = spaces.Box(low=min_action, high=max_action, shape=(1,), dtype=np.float32)
    observation_space = spaces.Box(low, high, dtype=np.float32)


    def __init__(self, goal_velocity: int = 0, *args, **kwargs):
        super(MountainCarContinuousLearnableEnvironment, self).__init__(*args, **kwargs)
        self.goal_velocity = goal_velocity

    def _termination_fn(self, state: StateType, action: ActionType, next_state: StateType) -> bool:
        assert len(state) == 2
        position, velocity = next_state[0], next_state[1]
        return bool(
            position >= self.goal_position and velocity >= self.goal_velocity
        )

    def _reward_fn(self, state: StateType, action: ActionType, next_state: StateType, done: bool) -> float:
        reward = 0
        if done:
            reward = 100.0
        reward -= math.pow(action[0], 2) * 0.1
        return reward

    def _reset_fn(self) -> StateType:
        self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
        return self.state

    def close(self):
        pass
