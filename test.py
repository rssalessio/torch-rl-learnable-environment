import numpy as np
import gym
from cartpole import CartPoleLearnableEnvironment
from ensemble_model.gaussian_ensemble import GaussianEnsemble
from ensemble_model.ensemble_utils import LayerInfo
from ensemble_model.gaussian_ensemble_network import GaussianEnsembleNetwork
from experience_buffer import Experience, ExperienceBuffer
import matplotlib.pyplot as plt
from stable_baselines3 import DQN

n_models = 5
state_dim = np.prod(CartPoleLearnableEnvironment.observation_space.shape)
action_dim = 1
reward_dim = 1

layers = [
    LayerInfo(input_size = state_dim + action_dim, output_size = 40, weight_decay = 1e-3), 
    LayerInfo(input_size = 40, output_size = 20, weight_decay = 1e-3),
    LayerInfo(input_size = 20, output_size = state_dim + reward_dim, weight_decay = 5e-4)]


num_samples = 1000
batch_size = 64
network = GaussianEnsembleNetwork(n_models, layers)
model = GaussianEnsemble(network, lr=1e-2)
env = gym.make("CartPole-v0")
state = env.reset()

buffer = ExperienceBuffer(num_samples)
for i in range(num_samples):
    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)
    buffer.append(Experience(state, action, reward, next_state, done))
    state = next_state
    if done:
        state = env.reset()

state, action, reward, next_state, done = buffer.sample_all()
training_losses, test_losses = model.train(state, action, reward, next_state, batch_size = batch_size, holdout_ratio = 0.2)

state = env.reset()

envEnsemble = CartPoleLearnableEnvironment(model)
losses = []

for i in range(5000):
    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)
    next_state_p, reward_p, done_p, info_p = envEnsemble._step(state, action)
    
    losses.append(np.linalg.norm(next_state - next_state_p, 2))
    state = next_state

    if done:
        state = env.reset()

plt.plot(losses)
plt.show()

# plt.plot(training_losses, label='Training losses')
# plt.plot(test_losses, label='Test losses')
# plt.grid()
# plt.legend()
# plt.show()
print(f'num_samples {num_samples} - batch_size {batch_size} - mean: {np.mean(losses)} - std: {np.std(losses)}')
# print('Star')
# model = DQN("MlpPolicy", envEnsemble, verbose=1)
# model.learn(total_timesteps=10000, log_interval=4)
# model.save("dqn_cartpole")