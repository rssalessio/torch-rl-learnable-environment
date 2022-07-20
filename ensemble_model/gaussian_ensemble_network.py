from functools import reduce
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .ensemble_linear_layer import EnsembleLinear
from .ensemble_utils import LayerInfo, create_ensemble_network_body

class GaussianEnsembleNetwork(nn.Module):
    """ Ensemble network that consists of fully connected layers. Outputs mean and variance """
    ensemble_size: int
    max_logvar: torch.Tensor
    min_logvar: torch.Tensor
    device: torch.device

    def __init__(self,
            ensemble_size: int,
            layers: List[LayerInfo],
            max_logvar: float = 0.5,
            min_logvar: float = -10,
            device: torch.device = torch.device('cpu')):
        super(GaussianEnsembleNetwork, self).__init__()
        assert max_logvar > min_logvar
        assert ensemble_size > 0

        self.ensemble_size = ensemble_size
        self.device = device
        self.model = create_ensemble_network_body(ensemble_size, layers[:-1]).to(self.device)

        # Add last layer (and add variance output)
        self.mean_layer = EnsembleLinear(
            layers[-1].input_size, layers[-1].output_size, ensemble_size,
            layers[-1].weight_decay, layers[-1].bias, layers[-1].device).to(self.device)
        self.var_layer = EnsembleLinear(
            layers[-1].input_size, layers[-1].output_size, ensemble_size,
            layers[-1].weight_decay, layers[-1].bias, layers[-1].device).to(self.device)

        # Initialize max logvar and min logvar
        self.max_logvar = nn.Parameter(torch.ones((1, layers[-1].output_size)).float() * max_logvar, requires_grad=False).to(self.device)
        self.min_logvar = nn.Parameter(torch.ones((1, layers[-1].output_size)).float() * min_logvar, requires_grad=False).to(self.device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self.model(x)
        mean = self.mean_layer(output)
        var = self.var_layer(output)

        logvar = self.max_logvar - F.softplus(self.max_logvar - var)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        return mean, logvar

    def get_decay_loss(self, exponent: float = 2) -> float:
        return reduce(lambda x, y: x + y.get_decay_loss(exponent) if isinstance(y, EnsembleLinear) else 0, self.children())

    def get_variance_regularizer(self) -> float:
        return (self.max_logvar - self.min_logvar).sum()
