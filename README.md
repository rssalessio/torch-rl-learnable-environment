# Learnable Environments for Model-Based Reinforcement Learning

The goal of this repository is to provide a gym-compatible library to easily perform model-based Reinforcement Learning experiments using PyTorch.
The library makes it easier to create learnable environments and ensembles of networks that can be used to learn the dynamics of an environment.

Specifically, the library provides the following:

- Provide a gym-like interface for environments that use a neural network to model the transition/reward function
- Provide implementations of neural networks used to model the transition/reward function (leveraging ensembles of networks)

See also the example in `examples/example.py` to see an example of ensemble network trained on the `CartPole` environment.

Author: Alessio Russo, alessior@kth.se

## How to use it

Clone the library and install it using `pip install`. Check the example in `examples/example.py`, or the example down below.

### Requirements

The library needs Python 3.7 to run and the following liraries:

- NumPy, Scikit-learn, Gym, Pydantic, PyTorch, Mujoco-py
- Matplotlib to run the example

## Example

In the following example we create an ensemble of 5 networks to emulate the CartPole environment in OpenGym. Refer to the file in examples/example.py for more details.

```python
import numpy as np
from learnable_environment import CartPoleLearnableEnvironment
from learnable_environment.ensemble_model import GaussianEnsemble, EnsembleLinearLayerInfo, GaussianEnsembleNetwork

# Ensemble size
n_models = 5

# State dimension, action dimension, reward dimension
state_dim = np.prod(CartPoleLearnableEnvironment.observation_space.shape)
action_dim = 1
reward_dim = 1

# Ensemble Network definition
layers = [
    EnsembleLinearLayerInfo(input_size = state_dim + action_dim, output_size = 120), 
    EnsembleLinearLayerInfo(input_size = 120, output_size = 40),
    EnsembleLinearLayerInfo(input_size = 40, output_size = state_dim + reward_dim)]
network = GaussianEnsembleNetwork(n_models, layers)

# Use ensemble network to create a model
model = GaussianEnsemble(network, lr=1e-2)

# Load ensemble or train it using maximum likelihood
# .....


# Test ensemble
envEnsemble = CartPoleLearnableEnvironment(model=model)
done = False
while not done:
    action = envEnsemble.action_space.sample()
    next_state, reward, done, info = envEnsemble.step(action)
    state =  next_state

```

## Example - Prediction error

In the following plots we see the performance of an ensemble of 5 networks used to learn the dynamics of the CartPole and the MountainCar environments (CartPole-v1 and MountainCar-v0 in OpenGym). The network has been trained with 3000 samples, batch size of 64 elements and a learning rate of 0.01. The network has 1 hidden layer with `in_features=32` and `out_features=16`.

![Prediction error at different horizon lengths](examples/img/example-mountaincar.png "MountainCar") ![Prediction error at different horizon lengths](examples/img/example-mountaincarcontinuous.png "MountainCarContinuous")![Prediction error at different horizon lengths](examples/img/example-cartpole.png "Cartpole")

## How to add new environments

All environments are in `learnable_environment/envs`. Any new environment needs to implement 3 functions:

- `_termination_fn(state, action, next_state)`: termination function. Evaluates if the MDP has reached a terminal state
- `_reset_fn()`: reset function (returns the initial state)
- `_reward_fn(state, action, next_state, done)`: reward function (in case you don't want to use the one learnt by the ensemble)

In addition to that, make sure to define the `observation_space` and `action_space` variables in the constructor.

Optionally, you can decide to set `_reward_fn` to `None` in case it is not possible to compute the reward (e.g., like in the Hopper environment where
the reward function depends on some hidden variable).

Make sure to regisgter your new environment in `learnable_environment/envs/registration.py` if you want to use the `make` function.

Check the example in `learnable_environment/envs/classic_control/cartpole.py` for more details.

## How to add new ensembles or transition function models

To add a new type of model that represents a transition function/reward function, you need to create an `EnsembleModel` that implements the `predict(inputs)` function, where `inputs` is a numpy array that concatenates `(state, action)`. The function must return the next state, and possibly, the predicted reward. Check an example of implementation in `learnable_environment/ensemble_model/gaussian_ensemble_model.py`.

Integrate this model in the `LearnableEnvironment` class (in `learnable_environment/learnable_environment.py`) in the `_step` function (add a new `if isinstance(self.model, YourModelName)` with your code).

## Roadmap

### Features implemented

- Implemented Gaussian Ensemble
  - Implemented KL-divergence computation between two different ensembles
  - Implemented log-likelihood computation
- Added Training support
- Added Multi-trajectory support
- Added gym-like make function
- Implemented environments
  - Classic Control
    - CartPole
    - MountainCar
    - MountainCarContinuous
  - MujoCo
    - InvertedPendulum
    - Hopper

### Todo

- Add Ant, Half-Cheetah, Inverted double-pendulum, Swimmer, Humanoid environments
- Update to new Gym API
- Implement tests
- Add render
- Add DOCS
- Rebranding (LearnableGym?)
