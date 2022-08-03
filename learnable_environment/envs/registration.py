from learnable_environment import LearnableEnvironment
from typing import Dict, Type

from learnable_environment.envs.classic_control.cartpole import CartPoleLearnableEnvironment
from learnable_environment.envs.classic_control.mountaincar import MountainCarLearnableEnvironment
from learnable_environment.envs.classic_control.mountaincar_continuous import MountainCarContinuousLearnableEnvironment
from learnable_environment.envs.mujoco.hopper import HopperLearnableEnvironment
from learnable_environment.envs.mujoco.invertedpendulum import InvertedPendulumLearnableEnvironment


environments: Dict[str, Type[LearnableEnvironment]] = {
    'CartPole': CartPoleLearnableEnvironment,
    'MountainCar': MountainCarLearnableEnvironment,
    'MountainCarContinuous': MountainCarContinuousLearnableEnvironment,
    'Hopper': HopperLearnableEnvironment,
    'InvertedPendulum': InvertedPendulumLearnableEnvironment
}


def make(env_name: str, **kwargs) -> LearnableEnvironment:
    assert env_name in environments, f"Could not find the environment '{env_name}'"
    return environments[env_name](**kwargs)
