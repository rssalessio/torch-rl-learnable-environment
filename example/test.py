import numpy as np
import gym
import matplotlib.pyplot as plt
from typing import List
from learnable_environment import CartPoleLearnableEnvironment, MountainCarLearnableEnvironment
from learnable_environment.ensemble_model import GaussianEnsembleModel, LayerInfo, GaussianEnsembleNetwork
from utils.experience_buffer import Experience, ExperienceBuffer
import torch.nn as nn

# Parameters
num_samples = 1000
batch_size = 64
n_models = 5
state_dim = np.prod(CartPoleLearnableEnvironment.observation_space.shape)
action_dim = 1
reward_dim = 1

# Buffer to save data
buffer = ExperienceBuffer(num_samples)

# Network definition
layers = [
    LayerInfo(input_size = state_dim + action_dim, output_size = 80), 
    LayerInfo(input_size = 80, output_size = 40),
    LayerInfo(input_size = 40, output_size = state_dim + reward_dim)]
network = GaussianEnsembleNetwork(n_models, layers)

# Use ensemble network to create a model
model = GaussianEnsembleModel(network, lr=1e-2)
env = gym.make("CartPole-v0")
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
training_losses, test_losses = model.train(state, action, reward, next_state, batch_size = batch_size, holdout_ratio = 0.2, use_decay=False)

# Test ensemble
prediction_stats = {}
samples: List[Experience] = []
state = env.reset()

# Collect data
for i in range(5000):
    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)
    samples.append(Experience(state, action, reward, next_state, done))
    state = next_state if not done else env.reset()

# Test prediction with different time horizons
envEnsemble = CartPoleLearnableEnvironment(model=model)
max_horizon = 5
for step_length in range(1, max_horizon):
    prediction_stats[step_length] = []
    for idx, experience in enumerate(samples):
        if experience.done: continue
        state, action, _, _, _ = experience
        error = []
        for t in range(step_length):
            if (idx + t) >= len(samples): break
            next_state, reward, done, info = envEnsemble._step(state, samples[idx + t].action)
            error.append(np.linalg.norm(samples[idx + t].next_state - next_state, 2))
            state = next_state
            if done: break
        prediction_stats[step_length].append(np.mean(error))

x_data = np.arange(max_horizon)[1:]
y_mean = np.array([np.mean(prediction_stats[idx]) for idx in prediction_stats.keys()])
y_cf = 1.96*np.array([np.std(prediction_stats[idx])/np.sqrt(len(prediction_stats[idx])) for idx in prediction_stats.keys()])

plt.plot(x_data, y_mean)
plt.fill_between(x_data, (y_mean - y_cf), (y_mean + y_cf), color='b', alpha=.1)
plt.xlabel('Prediction horizon')
plt.title('CartPole - Average $\ell_2$ error')
plt.grid()
plt.show()


