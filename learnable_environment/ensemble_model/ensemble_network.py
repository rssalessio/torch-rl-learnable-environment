import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class EnsembleNetwork(nn.Module, ABC):
    """ Abstract class that represents an ensemble of neural networks"""
    ensemble_size: int
    device: torch.device

    def __init__(self,
            ensemble_size: int,
            device: torch.device = torch.device('cpu')):
        super(EnsembleNetwork, self).__init__()
        assert ensemble_size > 0
        self.ensemble_size = ensemble_size
        self.device = device

    @abstractmethod
    def forward(self, x: torch.Tensor):
        return NotImplemented

    @abstractmethod
    def get_decay_loss(self, exponent: float = 2) -> float:
        return NotImplemented

    @abstractmethod
    def reset(self):
        return NotImplemented

