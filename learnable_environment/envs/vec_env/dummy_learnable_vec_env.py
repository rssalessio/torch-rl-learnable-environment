from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from typing import Callable, List, Tuple, Dict, Union
from learnable_environment.learnable_environment import LearnableEnvironment
from numpy.typing import NDArray

import numpy as np

from learnable_environment.learnable_environment import StateType, ActionType


class DummyLearnableVecEnv(DummyVecEnv):
    def __init__(self, env_fns: List[Callable[[], LearnableEnvironment]]):
        super(DummyLearnableVecEnv, self).__init__(env_fns)

    @property
    def learnable_env(self) -> LearnableEnvironment:
        return self.envs[0]

    def _step(self, state: StateType, action: ActionType) -> Tuple[StateType, float, bool, Dict[str, Union[StateType, NDArray[np.float64]]]]:
        for env_idx in range(self.num_envs):
            obs, self.buf_rews[env_idx], self.buf_dones[env_idx], self.buf_infos[env_idx] = self.envs[env_idx]._step(
                self.actions[env_idx]
            )
            if self.buf_dones[env_idx]:
                # save final observation where user can get it, then reset
                self.buf_infos[env_idx]["terminal_observation"] = obs
                obs = self.envs[env_idx].reset()
            self._save_obs(env_idx, obs)
        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), deepcopy(self.buf_infos))