import gym
import numpy as np
from numpy.typing import NDArray
from typing import Dict, Optional, Tuple, Union
from gym.utils import seeding
from ensemble_model.gaussian_ensemble import GaussianEnsemble
from abc import ABC, abstractmethod


StateType = NDArray[np.float64]
ActionType = Union[float, int, NDArray[Union[np.float64, np.int64]]]

class LearnableEnvironment(gym.Env, ABC):
    """
    Description:
        An environment that is learnt using a neural network
    """

    metadata = {
        'render.modes': [],
    }

    state: np.array
    n_steps: int
    action_space: gym.Space
    observation_space: gym.Space


    def __init__(self, model: GaussianEnsemble, use_learnt_reward_fn: bool = False, seed: Optional[int]  = None):
        self.model = model
        self.state = None
        self.n_steps = 0
        self.action_space = None
        self.observation_space = None
        self.use_learnt_reward_fn = use_learnt_reward_fn
        self.seed(seed)

    @abstractmethod
    def _termination_fn(self, state: StateType, action: ActionType, next_state: StateType) -> bool:
        return NotImplemented

    @abstractmethod
    def _reward_fn(self, state: StateType, action: ActionType, next_state: StateType, done: bool) -> float:
        return NotImplemented

    @abstractmethod
    def _reset_fn(self) -> StateType:
        return NotImplemented

    def seed(self, seed: Optional[int] = None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: ActionType) -> Tuple[StateType, float, bool, Dict[str, Union[StateType, NDArray[np.float64]]]]:
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        inputs = np.array([np.concatenate((self.state, [action]), axis=-1)])
        ensemble_means, ensemble_vars = self.model.predict(inputs)
        ensemble_stds = np.sqrt(ensemble_vars)
        ensemble_means[:, :, :-1] += self.state

        ensemble_samples = ensemble_means + np.random.normal(size=ensemble_means.shape) * ensemble_stds
        
        model_idx = np.random.choice(self.model.elite_models_idxs)

        samples = ensemble_samples[model_idx, 0, :]
        samples_means = ensemble_means[model_idx, 0, :]
        samples_std = ensemble_stds[model_idx, 0, :]

        next_state = samples[:-1]
        
        done = self._termination_fn(self.state, action, next_state)
        reward = samples[-1] if self.use_learnt_reward_fn else self._reward_fn(self.state, action, next_state, done)
        self.state = next_state

        info = {'mean': samples_means, 'std': samples_std}

        if done:
            info['terminal_observation'] = next_state
        return self.state, reward, done, info

    def reset(self) -> StateType:
        self.n_steps = 0
        return self._reset_fn()

    def render(self, mode='human'):
        raise Exception('Not implemented')

    def close(self):
        pass
    