from functools import reduce
from typing import Dict, List, Tuple
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from learnable_environment.ensemble_model.ensemble_network import EnsembleNetwork
from learnable_environment.ensemble_model.gaussian_ensemble_network import GaussianEnsembleNetwork
from learnable_environment.ensemble_model.ensemble_utils import save_best_result
from numpy.typing import NDArray
from abc import ABC, abstractmethod

class EnsembleModel(ABC):
    """Abstract class that represents a generic ensemble model"""
    elite_models_idxs: List[int]
    network_size: int

    def __init__(self,
            ensemble_model: EnsembleNetwork,
            elite_proportion: float = 0.2):
        assert elite_proportion > 0. and elite_proportion <= 1.
        self.ensemble_model = ensemble_model
        self.network_size = ensemble_model.ensemble_size
        self.elite_proportion = elite_proportion
        self.elite_models_idxs = [i for i in range(self.network_size)]

    @abstractmethod
    def predict(self, input: NDArray[np.float64]):
        return NotImplemented

    @abstractmethod
    def train(self):
        return NotImplemented
