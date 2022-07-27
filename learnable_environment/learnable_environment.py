from __future__ import annotations
import gym
import numpy as np
from numpy.typing import NDArray
from typing import Callable, Dict, List, Optional, Tuple, Union
from gym.utils import seeding
from learnable_environment.ensemble_model import GaussianEnsembleModel
from learnable_environment.ensemble_model import EnsembleModel
from abc import ABC, abstractmethod
import torch

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
        self._model = model
        self.state = None
        self.n_steps = 0
        self.use_learnt_reward_fn = use_learnt_reward_fn
        self.custom_reward_fn = custom_reward_fn
        self.seed(seed)
        self.vectorized_check_action = np.vectorize(self.action_space.contains)

    @property
    def model(self) -> EnsembleModel:
        return self._model

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
        # Reshape according to state
        _action = np.array(action)
        action_reshaped = np.expand_dims(_action, axis=tuple(range(len(_action.shape), len(state.shape))))
        if len(state.shape) == 1:
            assert self.action_space.contains(action)
        else:
            assert np.all(self.vectorized_check_action(action_reshaped))

        inputs = np.concatenate((state, action_reshaped), axis=-1)       

        if len(inputs.shape) < 2:
            inputs = inputs[None, :]

        info = {}
        batch_size = inputs.shape[0]
        if isinstance(self._model, GaussianEnsembleModel):
            ensemble_means, ensemble_vars = self._model.predict(inputs)
            ensemble_stds = np.sqrt(ensemble_vars)
            ensemble_means[:, :, :-1] += state

            ensemble_samples = ensemble_means + np.random.normal(size=ensemble_means.shape) * ensemble_stds
            
            model_idxs = np.random.choice(self._model.elite_models_idxs, size=batch_size)

            batch_idxs = np.arange(0, batch_size)
            samples = ensemble_samples[model_idxs, batch_idxs]
            samples_means = ensemble_means[model_idxs, batch_idxs]
            samples_std = ensemble_stds[model_idxs, batch_idxs]
            info = {'mean': samples_means, 'std': samples_std}

        else:
            raise Exception('Not implemented!')

        if batch_size == 1:
            next_state = samples[0, :-1]
            done = self._termination_fn(state, action, next_state)
            reward = samples[:, -1] if self.use_learnt_reward_fn else self._reward_fn(state, action, next_state, done)
            if done:
                info['terminal_observation'] = next_state
        else:
            next_state = samples[:, :-1]
            done = np.array([self._termination_fn(state[x], action[x], next_state[x]) for x in range(batch_size)])
            reward = np.array(
                [samples[x, -1] if self.use_learnt_reward_fn and not self._reward_fn else self._reward_fn(state[x], action[x], next_state[x], done[x]) for x in range(batch_size)])

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

    def compute_log_prob_batch(self, states: NDArray, actions: NDArray) -> torch.Tensor:
        if isinstance(self._model, GaussianEnsembleModel):
            _action = np.array(actions)
            actions_reshaped = np.expand_dims(_action, axis=tuple(range(len(_action.shape), len(states.shape))))
            inputs = np.concatenate((states, actions_reshaped), axis=-1)    
            return self.model.compute_log_prob_batch(inputs)
        else:
            raise Exception('Not implemented')

    def compute_kl_divergence_over_batch(self,
            state: NDArray,
            action: NDArray,
            modelB: LearnableEnvironment,
            mask: NDArray[np.bool_] = None,
            num_avg_over_ensembles: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        assert num_avg_over_ensembles > 0
        data = np.concatenate((state, action[:, None] if len(action.shape) < len(state.shape) else action), axis=-1)
        if isinstance(self._model, GaussianEnsembleModel):
            return self.model.compute_kl_divergence_over_batch(data, modelB.model, mask, num_avg_over_ensembles)
        else:
            raise Exception('Not implemented')

    def train(self, 
            state: NDArray,
            action: NDArray,
            reward: NDArray,
            next_state: NDArray,
            batch_size: int,
            holdout_ratio: float = 0.2,
            max_epochs: int = 1000,
            max_epochs_since_update: int = 5,
            use_decay: bool = False,
            variance_regularizer_factor: float = 1e-2,
            decay_regularizer_factor: float = 1e-3) -> Tuple[List[float], List[float]]:

        assert len(state) == len(action) == len(reward) == len(next_state)

        delta_state = next_state - state
        data = np.concatenate((state, action[:, None] if len(action.shape) < len(state.shape) else action), axis=-1)
        target = np.concatenate((delta_state, reward[:, None]), axis=-1)

        return self._model.train(data, target,
            batch_size, holdout_ratio, max_epochs, max_epochs_since_update,
            use_decay, variance_regularizer_factor, decay_regularizer_factor)

    def train_on_batch(self, 
            data: Tuple[List[StateType], List[ActionType], List[float], List[StateType]],
            batch_size: int,
            holdout_ratio: float = 0.2,
            max_epochs: int = 1000,
            max_epochs_since_update: int = 5,
            use_decay: bool = False,
            variance_regularizer_factor: float = 1e-2,
            decay_regularizer_factor: float = 1e-3) -> Tuple[List[float], List[float]]:
        assert len(data) == 4
        return self.train(data[0], data[1], data[2], data[3],
            batch_size, holdout_ratio, max_epochs, max_epochs_since_update,
            use_decay, variance_regularizer_factor, decay_regularizer_factor)

    