from functools import reduce
from typing import Dict, List, Tuple
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from learnable_environment.ensemble_model.ensemble_model import EnsembleModel
from learnable_environment.ensemble_model.gaussian_ensemble_network import GaussianEnsembleNetwork
from learnable_environment.ensemble_model.ensemble_utils import save_best_result
from numpy.typing import NDArray

class GaussianEnsembleModel(EnsembleModel):
    """ Represents an ensemble model that outputs mean and variance of a Gaussian distribution """
    def __init__(self,
            ensemble_model: GaussianEnsembleNetwork,
            lr: float = 1e-3,
            elite_proportion: float = 0.2):
        super(GaussianEnsembleModel, self).__init__(ensemble_model, elite_proportion)
        self.optimizer = torch.optim.Adam(self.ensemble_model.parameters(), lr=lr)
        self.scaler = StandardScaler()

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
        # Average over batch and dim
        mse_loss = ((mean - targets).square() * inv_var).mean(-1).mean(-1) 
        var_loss = logvar.mean(-1).mean(-1)
        return mse_loss, var_loss

    def _training_step(self, data: torch.Tensor, target: torch.Tensor, use_decay: bool, variance_regularizer_factor: float) -> float:
        mean, logvar = self.ensemble_model(data)
        mse_loss, var_loss = self._ll_loss(mean, logvar, target)
        
        # Sum over ensembles
        mse_loss = mse_loss.sum()
        var_loss = var_loss.sum()
        loss = (mse_loss + var_loss)
        self.optimizer.zero_grad()
        loss += variance_regularizer_factor * self.ensemble_model.get_variance_regularizer()
        if use_decay:
            loss += self.ensemble_model.get_decay_loss()
        loss.backward()
        self.optimizer.step()
        return mse_loss.item(), var_loss.item()

    def _training_loop(self,
            data: NDArray[np.float64],
            target: NDArray[np.float64],
            batch_size: int,
            use_decay: bool,
            variance_regularizer_factor: float) -> List[float]:
        train_idx = np.vstack([np.random.permutation(data.shape[0]) for _ in range(self.network_size)])
        losses = []
        for start_pos in range(0, data.shape[0], batch_size):
            idx = train_idx[:, start_pos: start_pos + batch_size]
            train_input = torch.from_numpy(data[idx]).float()
            train_label = torch.from_numpy(target[idx]).float()
            mse_loss, var_loss = self._training_step(train_input, train_label, use_decay, variance_regularizer_factor)
            losses.append(mse_loss + var_loss)
        return losses

    def _test_loop(self,
            epoch: int,
            test_data: torch.Tensor,
            test_target: torch.Tensor,
            snapshots: Dict[int, Tuple[int, float]]) -> Tuple[float, bool, Dict[int, Tuple[int, float]]]:

        with torch.no_grad():
            holdout_mean, holdout_logvar = self.ensemble_model(test_data)
            holdout_mse_losses, holdout_var_losses = self._ll_loss(holdout_mean, holdout_logvar, test_target)
            holdout_mse_losses = holdout_mse_losses.detach().cpu().numpy()
            holdout_var_losses = holdout_var_losses.detach().cpu().numpy()
            sorted_loss_idx = np.argsort(holdout_mse_losses)
            elite_size = int(len(sorted_loss_idx) * self.elite_proportion)
            self.elite_models_idxs = sorted_loss_idx[:elite_size].tolist()
        
        return [np.sum(holdout_mse_losses + holdout_var_losses), *save_best_result(epoch, snapshots, holdout_mse_losses)]

    def train(self,
            state: NDArray,
            action: NDArray,
            reward: NDArray,
            next_state: NDArray,
            batch_size: int,
            holdout_ratio: int = 0.2,
            max_epochs: int = 1000,
            max_epochs_since_update: int = 5,
            use_decay: bool = False,
            variance_regularizer_factor: float = 5e-3) -> Tuple[List[float], List[float]]:
        assert holdout_ratio > 0 and holdout_ratio < 1.
        assert len(state) == len(action) == len(reward) == len(next_state)

        delta_state = next_state - state
        data = np.concatenate((state, action[:, None]), axis=-1)
        target = np.concatenate((delta_state, reward[:, None]), axis=-1)

        epochs_since_update = 0
        num_training = int(data.shape[0] * (1-holdout_ratio))
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

        training_losses = []
        test_losses = []

        epoch = 0
        while epoch < max_epochs:
            _training_losses = self._training_loop(training_data, training_target, batch_size, use_decay, variance_regularizer_factor)
            training_losses.append(np.mean(_training_losses))
            _test_loss, updated, snapshots = self._test_loop(epoch, test_data, test_target, snapshots)
            test_losses.append(_test_loss)

            epochs_since_update = 0 if updated else epochs_since_update + 1
            if epochs_since_update > max_epochs_since_update:
                break
            epoch += 1

        return training_losses, test_losses


if __name__ == "__main__":
    from gaussian_ensemble_network import GaussianEnsembleNetwork
    from ensemble_utils import LayerInfo
    n_models = 5
    layers = [
        LayerInfo(input_size = 2, output_size = 4, weight_decay = 0.1), 
        LayerInfo(input_size = 4, output_size = 2, weight_decay = 0.05)]
    network = GaussianEnsembleNetwork(n_models, layers)
    ensemble = GaussianEnsembleModel(network)
    ensemble.scaler.fit([[1, 1], [3, 2]])

    x = np.array([[1, 0]])
    y = ensemble.predict(x)
    print(y)
    print(ensemble.predict(x)[0].shape)
