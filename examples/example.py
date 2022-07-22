import numpy as np
import gym
import matplotlib.pyplot as plt
from typing import List
from learnable_environment import CartPoleLearnableEnvironment, MountainCarLearnableEnvironment, MountainCarContinuousLearnableEnvironment
from learnable_environment.ensemble_model import GaussianEnsembleModel, LayerInfo, GaussianEnsembleNetwork
from utils.experience_buffer import Experience, ExperienceBuffer


ENV_NAME = 'MountainCarContinuous-v0'
env = gym.make(ENV_NAME)

# Parameters
num_samples = 3000
batch_size = 64
n_models = 5
state_dim = np.prod(env.observation_space.shape)
action_dim = 1 if isinstance(env.action_space, gym.spaces.Discrete) else np.prod(env.action_space.shape)
reward_dim = 1
max_horizon = 30

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
prediction_stats = {x: [] for x in range(1, max_horizon + 1)}
samples: List[Experience] = []
state = env.reset()

# Collect data
for i in range(5000):
    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)
    samples.append(Experience(state, action, reward, next_state, done))
    state = next_state if not done else env.reset()

# Test prediction with different time horizons
if 'CartPole' in ENV_NAME:
    envEnsemble = CartPoleLearnableEnvironment(model=model)
elif 'MountainCar-v0' == ENV_NAME:
    envEnsemble = MountainCarLearnableEnvironment(model=model)
elif 'MountainCarContinuous-v0' == ENV_NAME:
    envEnsemble = MountainCarContinuousLearnableEnvironment(model=model)
else:
    raise Exception('Model not implemented!')


for idx, experience in enumerate(samples):
    if experience.done: continue
    state, action, _, _, _ = experience
    for t in range(max_horizon):
        if (idx + t) >= len(samples): break
        next_state, reward, done, info = envEnsemble._step(state, samples[idx + t].action)
        prediction_stats[t+1].append(np.linalg.norm(samples[idx + t].next_state - next_state, 2))
        state = next_state
        if done:
            envEnsemble.reset()
            break

x_data = np.arange(max_horizon + 1)[1:]
y_mean = np.array([np.mean(prediction_stats[idx]) for idx in prediction_stats.keys()])
y_cf = 1.96 * np.array([np.std(prediction_stats[idx])/np.sqrt(len(prediction_stats[idx])) for idx in prediction_stats.keys()])

plt.plot(x_data, y_mean)
plt.fill_between(x_data, (y_mean - y_cf), (y_mean + y_cf), color='b', alpha=.1)
plt.xlabel('Prediction horizon')
plt.title(f'{ENV_NAME} - Average $\ell_2$ error (95% confidence interval)')
plt.grid()
plt.show()


