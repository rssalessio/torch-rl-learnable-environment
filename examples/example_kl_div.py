import numpy as np
import gym
import matplotlib.pyplot as plt
from typing import List

from sklearn import ensemble
from learnable_environment import CartPoleLearnableEnvironment, MountainCarLearnableEnvironment, MountainCarContinuousLearnableEnvironment
from learnable_environment.ensemble_model import GaussianEnsembleModel, EnsembleLinearLayerInfo, GaussianEnsembleNetwork
from learnable_environment.environments.mujoco.invertedpendulum import InvertedPendulumLearnableEnvironment
from utils.experience_buffer import Experience, ExperienceBuffer

# Create environment
ENV_NAME = 'CartPole-v1'
env = gym.make(ENV_NAME)

# Parameters
num_samples = 10000
num_evaluation = 1000
n_models = 5
state_dim = np.prod(env.observation_space.shape)
action_dim = 1 if isinstance(env.action_space, gym.spaces.Discrete) else np.prod(env.action_space.shape)
reward_dim = 1

# Buffer to save data
buffer = ExperienceBuffer(num_samples)

# Network definition
layers = [
    EnsembleLinearLayerInfo(input_size = state_dim + action_dim, output_size = 32),
    EnsembleLinearLayerInfo(input_size = 32, output_size = 16),
    EnsembleLinearLayerInfo(input_size = 16, output_size = state_dim + reward_dim)]
network1 = GaussianEnsembleNetwork(n_models, layers)
network2 = GaussianEnsembleNetwork(n_models, layers)

# Use ensemble network to create a model
envEnsemble1 = GaussianEnsembleModel(network1, lr=1e-2)
envEnsemble2 = GaussianEnsembleModel(network2, lr=1e-2)


state = env.reset()

# Sample data from true environment
for i in range(num_samples):
    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)
    buffer.append(Experience(state, action, reward, next_state, done))
    state = next_state
    if done:
        state = env.reset()

# Train ensemble
state, action, reward, next_state, done = buffer.sample_all()
n = int(num_samples * 0.5)

env1 = CartPoleLearnableEnvironment(model=envEnsemble1)
env2 = CartPoleLearnableEnvironment(model=envEnsemble2)

env1.train(state[:n], action[:n], reward[:n], next_state[:n], batch_size = 64, holdout_ratio = 0.2, use_decay=False)
env2.train(state[n:-num_evaluation], action[n:-num_evaluation], reward[n:-num_evaluation], next_state[n:-num_evaluation], batch_size = 64, holdout_ratio = 0.2, use_decay=False)

inputs = np.concatenate((state[-num_evaluation:], action[-num_evaluation:][:,None]), axis = -1)
mean_kl, std_kl = envEnsemble1.compute_kl_divergence(inputs, envEnsemble2, num_avg_over_ensembles=10)
print(f'KL Divergence between the two models: {mean_kl.item()} +- {1.96 * std_kl.item() / np.sqrt(10)} (95% confidence)')

mean_kl, std_kl = env1.compute_kl_divergence(state[-num_evaluation:], action[-num_evaluation:], env2, num_avg_over_ensembles=10)
print(f'KL Divergence between the two models: {mean_kl.item()} +- {1.96 * std_kl.item() / np.sqrt(10)} (95% confidence)')