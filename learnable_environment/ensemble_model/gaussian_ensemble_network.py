from functools import reduce
from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers.ensemble_gru_layer import EnsembleGRU, EnsembleGRULayerInfo
from .layers.ensemble_linear_layer import EnsembleLinear, EnsembleLinearLayerInfo
from .ensemble_network import EnsembleNetwork
from .ensemble_utils import create_ensemble_network_body

class GaussianEnsembleNetwork(EnsembleNetwork):
    """ Ensemble network that consists of fully connected layers. Outputs mean and variance """
    max_logvar: torch.Tensor
    min_logvar: torch.Tensor

    def __init__(self,
            ensemble_size: int,
            layers: List[Union[EnsembleGRULayerInfo,EnsembleLinearLayerInfo]],
            max_logvar: float = 0.5,
            min_logvar: float = -10,
            device: torch.device = torch.device('cpu')):
        super(GaussianEnsembleNetwork, self).__init__(ensemble_size, device)
        assert ensemble_size > 0
        self.model = create_ensemble_network_body(ensemble_size, layers[:-1]).to(self.device)

        # Add last layer (and add variance output)
        self.mean_layer = EnsembleLinear(
            layers[-1].input_size, layers[-1].output_size, ensemble_size,
            layers[-1].weight_decay, layers[-1].bias, layers[-1].device).to(self.device)
        self.var_layer = EnsembleLinear(
            layers[-1].input_size, layers[-1].output_size, ensemble_size,
            layers[-1].weight_decay, layers[-1].bias, layers[-1].device).to(self.device)

        # Initialize max logvar and min logvar
        self.max_logvar = nn.Parameter(torch.ones((1, layers[-1].output_size)).float() * max_logvar, requires_grad=True).to(self.device)
        self.min_logvar = nn.Parameter(torch.ones((1, layers[-1].output_size)).float() * min_logvar, requires_grad=True).to(self.device)

    def forward(self, x: torch.Tensor, hidden_state: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, Union[torch.Tensor, None]]:
        new_hidden_state = None

        if hidden_state is None:
            output = self.model(x)
        else:
            output = x
            for idx, layer in enumerate(self.model):
                if isinstance(layer, EnsembleGRU):
                    output, hx = layer(output, hidden_state)
                    new_hidden_state = hx

        mean = self.mean_layer(output)
        var = self.var_layer(output)

        logvar = self.max_logvar - F.softplus(self.max_logvar - var)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        return mean, logvar, new_hidden_state

    def reset(self):
        for layer in self.children():
            if isinstance(layer, EnsembleLinear) or isinstance(layer, EnsembleGRU):
                layer.reset()

    def get_decay_loss(self, exponent: float = 2) -> float:
        return reduce(lambda x, y: x + y.get_decay_loss(exponent) if isinstance(y, EnsembleLinear) else 0, self.children(), 0)

    def get_variance_regularizer(self) -> torch.Tensor:
        return (self.max_logvar - self.min_logvar).sum()
