from gym import spaces
import numpy as np
from learnable_environment.learnable_environment import LearnableEnvironment, StateType, ActionType

class InvertedPendulumLearnableEnvironment(LearnableEnvironment):
    action_space = spaces.Box(low=-3.0, high=3.0, shape=(1,), dtype=np.float32)
    observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64)

    def __init__(self, *args, **kwargs):
        super(InvertedPendulumLearnableEnvironment, self).__init__(*args, **kwargs)

    def _termination_fn(self, state: StateType, action: ActionType, next_state: StateType) -> bool:
        return bool(not np.isfinite(state).all() or (np.abs(state[1]) > 0.2))

    def _reward_fn(self, state: StateType, action: ActionType, next_state: StateType, done: bool) -> float:
        return 1.0

    def _reset_fn(self) -> StateType:
        return self.np_random.uniform(size=4, low=-0.01, high=0.01)

    def close(self):
        pass
