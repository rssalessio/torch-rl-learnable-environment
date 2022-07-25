from typing import Callable, List, Optional, Type
import torch
import torch.nn as nn
from learnable_environment.ensemble_model.layers import EnsembleLayer, EnsembleLayerInfo

class EnsembleGRULayerInfo(EnsembleLayerInfo):
    hidden_features: int
    num_layers: int
    output_size: int = None

class EnsembleGRU(EnsembleLayer, nn.Module):
    """
        Fully connected layers for ensemble models
    """
    __constants__ = ['ensemble_size', 'in_features', 'hidden_features', 'num_layers']
    in_features: int
    hidden_features: int
    num_layers: int
    ensemble_size: int
    gru_ensemble: List[Type[nn.GRU]]
    device: torch.device
    bias: bool
    weight_decay: float

    def __init__(self, 
            in_features: int,
            hidden_features: int,
            num_layers: int,
            ensemble_size: int,
            weight_decay: float = 0,
            bias: bool = True,
            device: torch.device = torch.device('cpu')) -> None:
        super(EnsembleGRU, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.num_layers = num_layers
        self.ensemble_size = ensemble_size
        self.device = device
        self.bias = bias
        self.weight_decay = weight_decay

        self.gru_ensemble = [nn.GRU(
            input_size = self.in_features,
            hidden_size = self.hidden_features,
            num_layers = self.num_layers,
            bias = self.bias,
            batch_first = False,
            dropout = 0,
            bidirectional = False).to(self.device) for _ in range(self.ensemble_size)]

        self.hidden_state = torch.zeros((self.ensemble_size, self.num_layers, self.hidden_features))

    def forward(self, input: torch.Tensor, hidden_state: torch.Tensor = None) -> torch.Tensor:
        hidden_state = self.hidden_state if hidden_state is None else hidden_state
        output = [self.gru_ensemble[idx](input[None,idx], hidden_state)for idx in range(self.ensemble_size)]
        output, new_hidden_state = map(torch.stack, zip(*output))
        # @TODO to fix
        self.hidden_state = new_hidden_state[0]

        return output[:,0,:,:], new_hidden_state

    def reset(self):
        self.hidden_state = torch.zeros((self.num_layers, self.hidden_size))

    def extra_repr(self) -> str:
        return f'ensemble_size={self.ensemble_size}, in_features={self.in_features}, hidden_features={self.hidden_size}, num_layers={self.num_layers}, bias={self.bias}'

    def get_decay_loss(self, exponent: float = 2) -> float:
        # @TODO
        return 0.


if __name__ == "__main__":
    layer = EnsembleGRU(10, 20, 2, 5, 5)
    x = torch.zeros((1,10))
    out, hx = layer(x)
    print(hx.shape)