from __future__ import annotations
from functools import reduce
from typing import Dict, List, Tuple, Type, Union, NamedTuple
import numpy as np
import torch
import random
from sklearn.preprocessing import StandardScaler
from learnable_environment.ensemble_model.layers.ensemble_gru_layer import EnsembleGRU
from learnable_environment.ensemble_model.ensemble_model import EnsembleModel
from learnable_environment.ensemble_model.gaussian_ensemble_network import GaussianEnsembleNetwork
from learnable_environment.ensemble_model.ensemble_utils import save_best_result
from numpy.typing import NDArray

class GaussianEnsembleModel(EnsembleModel):
    """ Represents an ensemble model that outputs mean and variance of a Gaussian distribution """

    def __init__(self,
            ensemble_model: GaussianEnsembleNetwork,
            lr: float = 1e-3,
            elite_proportion: float = 0.2,
            device: torch.device = torch.device('cpu')):
        super(GaussianEnsembleModel, self).__init__(ensemble_model, lr, elite_proportion, device)
        self._scaler = StandardScaler()
    
    @property
    def scaler(self) -> StandardScaler:
        """ Returns the scaler used by the ensemble """
        return self._scaler

    def predict(self, input: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """ Predict using the ensemble """
        mean, var = self._predict(input)
        return mean.detach().numpy(), var.detach().numpy()

    def reset(self):
        self.ensemble_model.reset()

    def _predict(self, input: NDArray[np.float64]) -> Tuple[torch.Tensor[torch.float64], torch.Tensor[torch.float64]]:
        """ Predict using the ensemble """
        input = torch.from_numpy(self.scaler.transform(input)).to(self.device).float()
        mean, logvar, _ = self.ensemble_model(input[None, :, :].repeat(self.network_size, 1, 1))
        return mean, logvar.exp()

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

    def _training_step(self, 
            data: torch.Tensor,
            target: torch.Tensor,
            use_decay: bool,
            variance_regularizer_factor: float = 1e-2,
            decay_regularizer_factor: float = 1e-3,
            hidden_state: torch.Tensor = None) -> Tuple[float, float, Dict[int, torch.Tensor]]:
        mean, logvar, new_hidden_state = self.ensemble_model(data, hidden_state)
        mse_loss, var_loss = self._ll_loss(mean, logvar, target)
        
        # Sum over ensembles
        mse_loss = mse_loss.sum()
        var_loss = var_loss.sum()
        loss = (mse_loss + var_loss)
        self.optimizer.zero_grad()
        loss += variance_regularizer_factor * self.ensemble_model.get_variance_regularizer()
        if use_decay:
           loss += decay_regularizer_factor * self.ensemble_model.get_decay_loss()
        loss.backward()
        self.optimizer.step()
        return mse_loss.item(), var_loss.item(), new_hidden_state

    def _training_loop(self,
            data: NDArray[np.float64],
            target: NDArray[np.float64],
            batch_size: int,
            use_decay: bool,
            variance_regularizer_factor: float = 1e-2,
            decay_regularizer_factor: float = 1e-3) -> Tuple[List[float], Dict[int, torch.Tensor]]:
        train_idx = np.vstack([np.random.permutation(data.shape[0]) for _ in range(self.network_size)])

        losses = []
        for start_pos in range(0, data.shape[0], batch_size):
            idx = train_idx[:, start_pos: start_pos + batch_size]
            train_input = torch.from_numpy(data[idx]).to(self.device).float()
            train_label = torch.from_numpy(target[idx]).to(self.device).float()
     
            mse_loss, var_loss, _ = self._training_step(
                train_input, train_label, use_decay, variance_regularizer_factor, decay_regularizer_factor, None)

            losses.append(mse_loss)
        return losses

    def _training_loop_recurrent(self,
            data: torch.Tensor,
            target: torch.Tensor,
            hidden_state: torch.Tensor,
            sequence_length: int,
            use_decay: bool,
            variance_regularizer_factor: float = 1e-2,
            decay_regularizer_factor: float = 1e-3) -> Tuple[List[float], Dict[int, torch.Tensor]]:
        # Randomize over trajectories
        idxs = np.vstack([np.random.permutation(data.shape[0])[None,:] for _ in range(self.network_size)])
        # @TODO we should implement  a better randomization. We only randomize over trajectories, and not samples inside every trajectory
        losses = []
        for i in range(data.shape[0]):
            for start_pos in range(0, data.shape[1], sequence_length):
                train_input = data[idxs[:,i], start_pos : start_pos + sequence_length, :]
                train_target = target[idxs[:,i], start_pos : start_pos + sequence_length, :]
                train_hidden = hidden_state[idxs[:, i], start_pos : start_pos + sequence_length]
                mse_loss, var_loss, new_hidden_state = self._training_step(
                    train_input, train_target, use_decay, variance_regularizer_factor, decay_regularizer_factor, train_hidden)
                
                losses.append(mse_loss)
        return losses
        # train_idx = np.vstack([np.random.permutation(data.shape[0]) for _ in range(self.network_size)])

        # losses = []
        # for start_pos in range(0, data.shape[0], batch_size):
        #     idx = train_idx[:, start_pos: start_pos + batch_size]
        #     train_input = torch.from_numpy(data[idx]).to(self.device).float()
        #     train_label = torch.from_numpy(target[idx]).to(self.device).float()
     
        #     mse_loss, var_loss, new_info = self._training_step(
        #         train_input, train_label, use_decay, variance_regularizer_factor, decay_regularizer_factor, None)

        #     losses.append(mse_loss)
        # return losses

    def _test_loop(self,
            epoch: int,
            test_data: torch.Tensor,
            test_target: torch.Tensor,
            snapshots: Dict[int, Tuple[int, float]],
            info: Dict[Union[str, int], torch.Tensor] = None) -> Tuple[float, bool, Dict[int, Tuple[int, float]]]:

        with torch.no_grad():
            test_info = None
            holdout_mean, holdout_logvar, new_info = self.ensemble_model(test_data, info)
            holdout_mse_losses, holdout_var_losses = self._ll_loss(holdout_mean, holdout_logvar, test_target)
            holdout_mse_losses = holdout_mse_losses.detach().cpu().numpy()
            holdout_var_losses = holdout_var_losses.detach().cpu().numpy()
            sorted_loss_idx = np.argsort(holdout_mse_losses)
            elite_size = int(len(sorted_loss_idx) * self.elite_proportion)
            self.elite_models_idxs = sorted_loss_idx[:elite_size].tolist()

        
        return [np.sum(holdout_mse_losses), *save_best_result(epoch, snapshots, holdout_mse_losses)]

    def train_on_trajectories(self,
            data_trajectories: List[NDArray],
            target_trajectories: List[NDArray],
            sequence_length: int = 2,
            holdout_ratio: float = 0.2,
            max_epochs: int = 1000,
            max_epochs_since_update: int = 5,
            use_decay: bool = False,
            variance_regularizer_factor: float = 1e-2,
            decay_regularizer_factor: float = 1e-3
        ):
        assert len(target_trajectories) == len(data_trajectories)
        assert holdout_ratio > 0 and holdout_ratio < 1.
        
        num_trajectories = len(data_trajectories)
        num_training = int(num_trajectories * (1-holdout_ratio))
        permutation = np.random.permutation(num_trajectories)
        data_trajectories, target_trajectories = np.array(data_trajectories, dtype=object), np.array(target_trajectories, dtype=object)
        data_trajectories, target_trajectories = data_trajectories[permutation], target_trajectories[permutation]

        # Split
        training_data, training_target = data_trajectories[:num_training], target_trajectories[:num_training]
        test_data, test_target = data_trajectories[num_training:], target_trajectories[num_training:]

        # Train scaler
        self.scaler.fit(np.concatenate(training_data))
        training_data = list(map(self.scaler.transform, training_data))
        test_data = list(map(self.scaler.transform, test_data))

        # Pytorch transform
        test_data = list(map(lambda x: torch.from_numpy(x).to(self.device).float(), test_data))
        test_target = list(map(lambda x: torch.from_numpy(x).to(self.device).float(), test_target))
        test_data = list(map(lambda x: x[None, :, :].repeat([self.network_size, 1, 1]), test_data))
        test_target = list(map(lambda x: x[None, :, :].repeat([self.network_size, 1, 1]), test_target))

        # Check if model has any recurrent layer
        training_hidden_state = []
        test_hidden_state = []
        
        for idx, layer in enumerate(self.ensemble_model.model):
            # @TODO works if there's only 1 recurrent layer
            if isinstance(layer, EnsembleGRU):
                for x in range(max(len(training_data), len(test_data))):
                    # @TODO we should have different hidden states for different networks!
                    if x < len(training_data):
                        training_hidden_state.append(torch.zeros((len(training_data[x]), layer.num_layers, layer.hidden_features)))
                    if x < len(test_data):
                        test_hidden_state.append(torch.zeros((len(test_data[x]), layer.num_layers,  layer.hidden_features)))
        import pdb
        pdb.set_trace()
        training_hidden_state = torch.nn.utils.rnn.pad_sequence(training_hidden_state, batch_first=True, padding_value=0).to(self.device).float()
        test_hidden_state = torch.nn.utils.rnn.pad_sequence(test_hidden_state, batch_first=True, padding_value=0).to(self.device).float()
        training_data = torch.nn.utils.rnn.pad_sequence([torch.from_numpy(training_data[x]) for x in range(len(training_data))], batch_first=True, padding_value=0).to(self.device).float()
        training_target = torch.nn.utils.rnn.pad_sequence([torch.from_numpy(training_target[x]) for x in range(len(training_target))], batch_first=True, padding_value=0).to(self.device).float()
        
        epoch = 0
        training_losses, test_losses = [], []
        while epoch < max_epochs:
            _training_losses = self._training_loop_recurrent(
                training_data, training_target, training_hidden_state, sequence_length,
                use_decay, variance_regularizer_factor, decay_regularizer_factor)
            training_losses.append(np.mean(_training_losses))
            # _test_loss, updated, snapshots = self._test_loop(epoch, test_data, test_target, snapshots)
            # test_losses.append(_test_loss)

            # epochs_since_update = 0 if updated else epochs_since_update + 1
            # if epochs_since_update > max_epochs_since_update:
            #     break
            epoch += 1

        return 0


    def train(self,
            data: NDArray,
            target: NDArray,
            batch_size: int,
            holdout_ratio: float = 0.2,
            max_epochs: int = 1000,
            max_epochs_since_update: int = 5,
            use_decay: bool = False,
            variance_regularizer_factor: float = 1e-2,
            decay_regularizer_factor: float = 1e-3) -> Tuple[List[float], List[float]]:
        assert holdout_ratio > 0 and holdout_ratio < 1.

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
        test_data = torch.from_numpy(test_data).to(self.device).float()
        test_target = torch.from_numpy(test_target).to(self.device).float()
        test_data = test_data[None, :, :].repeat([self.network_size, 1, 1])
        test_target = test_target[None, :, :].repeat([self.network_size, 1, 1])

        training_losses = []
        test_losses = []

        epoch = 0
        while epoch < max_epochs:
            _training_losses = self._training_loop(
                training_data, training_target, batch_size, use_decay, variance_regularizer_factor, decay_regularizer_factor)
            training_losses.append(np.mean(_training_losses))
            _test_loss, updated, snapshots = self._test_loop(epoch, test_data, test_target, snapshots)
            test_losses.append(_test_loss)

            epochs_since_update = 0 if updated else epochs_since_update + 1
            if epochs_since_update > max_epochs_since_update:
                break
            epoch += 1

        return training_losses, test_losses

    def compute_kl_divergence(self,
            inputs: NDArray,
            modelB: GaussianEnsembleModel,
            mask: NDArray[np.bool_] = None,
            num_avg_over_ensembles: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.network_size == modelB.network_size
        assert num_avg_over_ensembles > 0

        if mask:
            inputs = inputs[:, mask]

        input_dim = inputs.shape[-1]
        meanA, varA = self._predict(inputs)
        meanB, varB = modelB._predict(inputs)

        stdA, stdB = torch.sqrt(varA), torch.sqrt(varB)
        batch_size = meanA.shape[1]
        batch_idxs = np.arange(0, batch_size)

        results = torch.zeros(num_avg_over_ensembles)

        for idx in range(num_avg_over_ensembles):
            modelA_idxs = np.random.choice(self.network_size, size=batch_size)
            modelB_idxs = np.random.choice(self.network_size, size=batch_size)
            meanA_sampled, meanB_sampled = meanA[modelA_idxs, batch_idxs], meanB[modelB_idxs, batch_idxs]
            stdA_sampled, stdB_sampled = stdA[modelA_idxs, batch_idxs], stdB[modelB_idxs, batch_idxs]

            term_1 = torch.divide(stdA_sampled, stdB_sampled).sum(-1)


            term_2 = torch.multiply(1 / stdB_sampled, torch.pow(meanA_sampled - meanB_sampled, 2)).sum(-1)
            term_3 = torch.log(torch.divide(torch.prod(stdB_sampled, axis=-1) , torch.prod(stdA_sampled, axis=-1)))
            
            res = 0.5 * (term_1 + term_2 - input_dim + term_3)
            results[idx] = res.mean(-1)

        return results.mean(), results.std()

if __name__ == "__main__":
    from layers import EnsembleGRULayerInfo, EnsembleLinearLayerInfo
    
    layers = [
        EnsembleGRULayerInfo(input_size = 1 + 1, hidden_features = 32, num_layers = 2),
        EnsembleLinearLayerInfo(input_size = 32, output_size = 1 + 1)]
    network = GaussianEnsembleNetwork(5, layers)
    ensemble = GaussianEnsembleModel(network)
    ensemble.scaler.fit([[1, 1], [3, 2]])

    horizons = [np.random.randint(5, 10) for x in range(5)]
    data = [np.random.uniform(low=0, high =1, size=(horizons[x], 2)) for x in range(5)]
    test = [np.random.uniform(low=0, high =1, size=(horizons[x], 2)) for x in range(5)]
    
    ensemble.train_on_trajectories(
        data_trajectories = data,
        target_trajectories =  test,
        sequence_length = 2
    )
