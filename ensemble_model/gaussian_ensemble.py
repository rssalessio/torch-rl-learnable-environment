from functools import reduce
from typing import Dict, List, Tuple
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from .gaussian_ensemble_network import GaussianEnsembleNetwork
from .ensemble_utils import save_best_result
from numpy.typing import NDArray

class GaussianEnsemble(object):
    elite_models_idxs: List[int]

    def __init__(self, ensemble_model: GaussianEnsembleNetwork, lr: float = 1e-3, elite_proportion: float = 0.2):
        assert elite_proportion > 0. and elite_proportion <= 1.
        self.model_list = []
        self.ensemble_model = ensemble_model
        self.optimizer = torch.optim.Adam(self.ensemble_model.parameters(), lr=lr)
        self.scaler = StandardScaler()
        self.network_size = ensemble_model.ensemble_size
        self.elite_proportion = elite_proportion
        self.elite_models_idxs = [i for i in range(self.network_size)]

    def predict(self, input: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        input = torch.from_numpy(self.scaler.transform(input)).float()
        mean, logvar = self.ensemble_model(input[None, :, :].repeat(self.network_size, 1, 1))
        return mean.detach().numpy(), logvar.exp().detach().numpy()

    def _ll_loss(self, mean: torch.Tensor, logvar: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        mean, logvar: Ensemble_size x batch_size x dim
        labels:  batch_size x dim
        """
        assert len(mean.shape) == len(logvar.shape) == len(targets.shape) == 3
        inv_var = torch.exp(-logvar)
        # Average over batch and dim, sum over ensembles.
        mse_loss = ((mean - targets).square() * inv_var).mean(-1).mean(-1).sum() 
        var_loss = logvar.mean(-1).mean(-1).sum()
        return mse_loss, var_loss

    def _training_step(self, data: torch.Tensor, target: torch.Tensor, use_decay: False) -> float:
        mean, logvar = self.ensemble_model(data)
        mse_loss, var_loss = self._ll_loss(mean, logvar, target)
        loss = mse_loss + var_loss
        self.optimizer.zero_grad()
        loss += 1e-2 * self.ensemble_model.get_variance_regularizer()
        if use_decay:
            loss += self.ensemble_model.get_decay_loss()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _training_loop(self, data: NDArray[np.float64], target: NDArray[np.float64], batch_size: int) -> List[float]:
        train_idx = np.vstack([np.random.permutation(data.shape[0]) for _ in range(self.network_size)])
        losses = []
        for start_pos in range(0, data.shape[0], batch_size):
            idx = train_idx[:, start_pos: start_pos + batch_size]
            train_input = torch.from_numpy(data[idx]).float()
            train_label = torch.from_numpy(target[idx]).float()
            mean, logvar = self.ensemble_model(train_input)
            loss = self._training_step(mean, logvar, train_label)
            losses.append(loss)
        return losses

    def _test_loop(self,
            epoch: int,
            test_data: torch.Tensor,
            test_target: torch.Tensor,
            snapshots: Dict[int, Tuple[int, float]]) -> Tuple[bool, Dict[int, Tuple[int, float]]]:

        with torch.no_grad():
            holdout_mean, holdout_logvar = self.ensemble_model(test_data)
            holdout_mse_losses, _ = self._ll_loss(holdout_mean, holdout_logvar, test_target)
            holdout_mse_losses = holdout_mse_losses.detach().cpu().numpy()
            sorted_loss_idx = np.argsort(holdout_mse_losses)
            elite_size = int(len(sorted_loss_idx) * self.elite_proportion)
            self.elite_models_idxs = sorted_loss_idx[:elite_size].tolist()
        
        return save_best_result(epoch, snapshots, holdout_mse_losses)

    def train(self,
            data: NDArray[np.float64],
            target: NDArray[np.float64],
            batch_size: int,
            holdout_ratio: int = 0.8,
            max_epochs: int = 1000,
            max_epochs_since_update: int = 5):
        epochs_since_update = 0
        num_training = int(data.shape[0] * holdout_ratio)
        snapshots = {i: (None, np.infty) for i in range(self.network_size)}

        # Shuffle
        permutation = np.random.permutation(data.shape[0])
        data, target = data[permutation], target[permutation]

        # Split
        training_data, training_target = data[:num_training], target[:num_training]
        test_data, test_target = data[num_training:], target[num_training:]

        # Train scaler
        training_data = self.scaler.fit_transform(training_data)
        test_data = self.scaler.transform(test_data)

        test_data = torch.from_numpy(test_data).float()
        test_target = torch.from_numpy(test_target).float()
        test_data = test_data[None, :, :].repeat([self.network_size, 1, 1])
        test_target = test_target[None, :, :].repeat([self.network_size, 1, 1])

        losses = []

        epoch = 0
        while epoch < max_epochs:
            _losses = self._training_loop(training_data, training_target, batch_size)
            losses.append(np.mean(_losses))
            updated, snapshots = self._test_loop(epoch, test_data, test_target, snapshots)

            epochs_since_update = 0 if updated else epochs_since_update + 1
            if epochs_since_update > max_epochs_since_update:
                break
            epoch += 1

        return losses


if __name__ == "__main__":
    from gaussian_ensemble_network import GaussianEnsembleNetwork
    from ensemble_utils import LayerInfo
    n_models = 5
    layers = [
        LayerInfo(input_size = 2, output_size = 4, weight_decay = 0.1), 
        LayerInfo(input_size = 4, output_size = 2, weight_decay = 0.05)]
    network = GaussianEnsembleNetwork(n_models, layers)
    ensemble = GaussianEnsemble(network)
    ensemble.scaler.fit([[1, 1], [3, 2]])

    x = np.array([[1, 0]])
    y = ensemble.predict(x)
    print(y)
    print(ensemble.predict(x)[0].shape)
