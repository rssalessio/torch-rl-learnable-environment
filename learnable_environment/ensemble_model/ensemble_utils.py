import torch.nn as nn
import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple, Dict, Union, NamedTuple
from learnable_environment.ensemble_model.layers import EnsembleGRU, EnsembleGRULayerInfo
from learnable_environment.ensemble_model.layers import EnsembleLinear, EnsembleLinearLayerInfo

def create_ensemble_network_body(ensemble_size: int, layers: List[Union[EnsembleGRULayerInfo,EnsembleLinearLayerInfo]]) -> nn.Module:
    assert ensemble_size > 0
    _layers = []
    for idx, layer in enumerate(layers):
        # @TODO isinstance does not seem to work
        if 'EnsembleLinearLayerInfo' in str(layer.__class__):
            _layers.append(EnsembleLinear(layer.input_size, layer.output_size, ensemble_size,  layer.weight_decay, layer.bias, layer.device))
        elif 'EnsembleGRULayerInfo' in str(layer.__class__):
            _layers.append(EnsembleGRU(layer.input_size, layer.hidden_features, layer.num_layers, ensemble_size, layer.weight_decay, layer.bias, layer.device))
        if layer.activation_function:
            _layers.append(layer.activation_function())

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

