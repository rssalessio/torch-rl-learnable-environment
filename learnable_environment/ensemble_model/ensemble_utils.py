import torch
import torch.nn as nn
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, validator
from typing import List, Optional, Tuple, Dict
from learnable_environment.ensemble_model.ensemble_linear_layer import EnsembleLinear

class LayerInfo(BaseModel):
    input_size: int
    output_size: int
    weight_decay: float = 0.
    bias: bool = True
    activation_function: Optional[nn.Module] = nn.ReLU()
    device: torch.device = torch.device('cpu')

    class Config:
        arbitrary_types_allowed = True

def create_ensemble_network_body(ensemble_size: int, layers: List[LayerInfo]) -> nn.Module:
    assert ensemble_size > 0
    _layers = []
    for idx, layer in enumerate(layers):
        _layers.append(EnsembleLinear(layer.input_size, layer.output_size, ensemble_size,  layer.weight_decay, layer.bias, layer.device))
        if layer.activation_function:
            _layers.append(layer.activation_function)

    return nn.Sequential(*_layers)

def save_best_result(
        epoch: int,
        snapshots: Dict[int, Tuple[int, float]],
        holdout_losses: NDArray[np.float64],
        rel_tol: float = 1e-2) -> Tuple[bool, Dict[int, Tuple[int, float]]]:
    updated = False
    for i, current in enumerate(holdout_losses):
        _, best = snapshots[i]
        if 1 - (current / best) > rel_tol:
            snapshots[i] = (epoch, current)
            updated = True
    return updated, snapshots
