from typing import Dict, List
import numpy as np
import torch
from learnable_environment.ensemble_model.ensemble_network import EnsembleNetwork
from numpy.typing import NDArray
from abc import ABC, abstractmethod

class EnsembleModel(ABC):
    """Abstract class that represents a generic ensemble model"""
    elite_models_idxs: List[int]
    network_size: int
    device: torch.device
    learning_rate: float
    model: torch.nn.Module

    def __init__(self,
            ensemble_model: EnsembleNetwork,
            lr: float = 1e-2,
            elite_proportion: float = 0.2,
            device: torch.device = torch.device('cpu')):
        assert elite_proportion > 0. and elite_proportion <= 1.
        self.ensemble_model = ensemble_model
        self.network_size = ensemble_model.ensemble_size
        self.elite_proportion = elite_proportion
        self.elite_models_idxs = [i for i in range(self.network_size)]
        self.device = device
        self.learning_rate = lr
        self._optimizer = torch.optim.Adam(self.ensemble_model.parameters(), lr=lr)

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        """ Returns the ensemble optimizer """
        return self._optimizer

    @abstractmethod
    def predict(self, input: NDArray[np.float64]):
        return NotImplemented

    @abstractmethod
    def train(self):
        return NotImplemented

    @abstractmethod
    def train_on_trajectories(self):
        return NotImplemented

    @abstractmethod
    def reset(self):
        return NotImplemented

