"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
from gym import spaces, logger
import numpy as np
from learnable_environment.learnable_environment import LearnableEnvironment, StateType, ActionType


class CartPoleLearnableEnvironment(LearnableEnvironment):
    # Angle at which to fail the episode
    theta_threshold_radians = 12 * 2 * math.pi / 360
    x_threshold = 2.4

    # Angle limit set to 2 * theta_threshold_radians so failing observation
    # is still within bounds.
    high = np.array([x_threshold * 2,
                    np.finfo(np.float32).max,
                    theta_threshold_radians * 2,
                    np.finfo(np.float32).max],
                    dtype=np.float32)

    action_space = spaces.Discrete(2)
    observation_space = spaces.Box(-high, high, dtype=np.float32)


    def __init__(self, *args, **kwargs):
        super(CartPoleLearnableEnvironment, self).__init__(*args, **kwargs)
        self.steps_beyond_done = None

    def _termination_fn(self, state: StateType, action: ActionType, next_state: StateType) -> bool:
        assert len(next_state) == 4
        x = next_state[0]
        theta = next_state[2]

        return bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
            or self.n_steps == 200
        )

    def _reward_fn(self, state: StateType, action: ActionType, next_state: StateType, done: bool) -> float:
        if self.custom_reward_fn:
            return self.custom_reward_fn(state, action, next_state, done)

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0
        return reward

    def _reset_fn(self) -> StateType:
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return self.state

    def close(self):
        pass

if __name__ == "__main__":
    from learnable_environment.ensemble_model.gaussian_ensemble_model import GaussianEnsembleModel
    from ensemble_model.ensemble_utils import LayerInfo
    from ensemble_model.gaussian_ensemble_network import GaussianEnsembleNetwork
    n_models = 5
    state_dim = 4
    action_dim = 1
    reward_dim = 1
    layers = [
        LayerInfo(input_size = state_dim + action_dim, output_size = 40, weight_decay = 1e-3), 
        LayerInfo(input_size = 40, output_size = 40, weight_decay = 1e-3),
        LayerInfo(input_size = 40, output_size = state_dim + reward_dim, weight_decay = 5e-4)]
    network = GaussianEnsembleNetwork(n_models, layers)
    model = GaussianEnsembleModel(network)
    model.scaler.fit([[1, 1, 1, 1, 1], [3, 2, 2, 1.5, 0]])

    env = CartPoleLearnableEnvironment(model = model)
    state = env.reset()
    
    assert len(state.shape) == 1
    assert state.shape[0] == 4

    print(env.step(1))