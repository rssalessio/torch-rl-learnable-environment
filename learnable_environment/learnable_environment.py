import gym
import numpy as np
from numpy.typing import NDArray
from typing import Callable, Dict, Optional, Tuple, Union
from gym.utils import seeding
from learnable_environment.ensemble_model import GaussianEnsembleModel
from learnable_environment.ensemble_model import EnsembleModel
from abc import ABC, abstractmethod


StateType = NDArray[np.float64]
ActionType = Union[float, int, NDArray[Union[np.float64, np.int64]]]

class LearnableEnvironment(gym.Env):
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

    def __init__(self,
            model: EnsembleModel,
            use_learnt_reward_fn: bool = False,
            custom_reward_fn: Optional[Callable[[StateType, ActionType, StateType, bool], float]] = None,
            seed: Optional[int]  = None):
        self.model = model
        self.state = None
        self.n_steps = 0
        self.use_learnt_reward_fn = use_learnt_reward_fn
        self.custom_reward_fn = custom_reward_fn
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

    def _step(self, state: StateType, action: ActionType) -> Tuple[StateType, float, bool, Dict[str, Union[StateType, NDArray[np.float64]]]]:
        if not self.action_space.contains(action):
            err_msg = "%r (%s) invalid" % (action, type(action))
            raise Exception(err_msg)

        # Reshape according to state
        _action = np.array(action)
        action_reshaped = np.expand_dims(_action, axis=tuple(range(len(_action.shape), len(state.shape))))
        inputs = np.concatenate((state, action_reshaped), axis=-1)[None, :]
        info = {}
        if isinstance(self.model, GaussianEnsembleModel):
            ensemble_means, ensemble_vars = self.model.predict(inputs)
            ensemble_stds = np.sqrt(ensemble_vars)
            ensemble_means[:, :, :-1] += state

            ensemble_samples = ensemble_means + np.random.normal(size=ensemble_means.shape) * ensemble_stds
            
            model_idx = np.random.choice(self.model.elite_models_idxs)

            samples = ensemble_samples[model_idx, 0, :]
            samples_means = ensemble_means[model_idx, 0, :]
            samples_std = ensemble_stds[model_idx, 0, :]
            info = {'mean': samples_means, 'std': samples_std}

        else:
            raise Exception('Not implemented!')

        next_state = samples[:-1]
        
        done = self._termination_fn(state, action, next_state)
        reward = samples[-1] if self.use_learnt_reward_fn else self._reward_fn(state, action, next_state, done)
        if done:
            info['terminal_observation'] = next_state
        return next_state, reward, done, info


    def step(self, action: ActionType) -> Tuple[StateType, float, bool, Dict[str, Union[StateType, NDArray[np.float64]]]]:
        next_state, reward, done, info = self._step(self.state, action)
        self.state = next_state
        return next_state, reward, done, info

    def reset(self) -> StateType:
        self.n_steps = 0
        return self._reset_fn()

    def render(self, mode='human'):
        raise Exception('Not implemented')

    def close(self):
        pass
    