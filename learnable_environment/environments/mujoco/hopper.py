from gym import spaces
import numpy as np
from learnable_environment.learnable_environment import LearnableEnvironment, StateType, ActionType

class HopperLearnableEnvironment(LearnableEnvironment):
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
    observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float64)

    def __init__(self, *args, **kwargs):
        super(HopperLearnableEnvironment, self).__init__(*args, **kwargs)
        # For hopper the reward function depends on a hidden variable, therefore we can't use it
        self._reward_fn = None

    def _termination_fn(self, state: StateType, action: ActionType, next_state: StateType) -> bool:
        height = next_state[:, 0]
        angle = next_state[:, 1]
        terminated = not (
            np.isfinite(next_state).all()
            and (np.abs(next_state[:, 1:]) < 100).all()
            and (height > 0.7)
            and (abs(angle) < 0.2)
        )

        return terminated

    def _reset_fn(self) -> StateType:
        init = np.array([(0.0, 1.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)])
        return self.np_random.uniform(size=len(init), low=-5e-3, high=5e-3)

    def close(self):
        pass
