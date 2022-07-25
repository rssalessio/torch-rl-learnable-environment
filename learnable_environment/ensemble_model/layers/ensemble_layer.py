from typing import Callable, List, Optional
from pydantic import BaseModel
import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class EnsembleLayerInfo(BaseModel):
    input_size: int
    output_size: int
    weight_decay: float = 0.
    bias: bool = True
    activation_function: Optional[Callable[[],nn.Module]] = lambda: nn.SiLU()
    device: torch.device = torch.device('cpu')

    class Config:
        arbitrary_types_allowed = True


class EnsembleLayer(ABC):
    in_features: int
    out_features: int
    ensemble_size: int
    weight_decay: float
    weight: torch.Tensor
    device: torch.device

    @abstractmethod
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        raise NotImplemented

    @abstractmethod
    def reset(self):
       raise NotImplemented

    @abstractmethod
    def extra_repr(self) -> str:
        raise NotImplemented

    @abstractmethod
    def get_decay_loss(self, exponent: float = 2) -> float:
        raise NotImplemented

