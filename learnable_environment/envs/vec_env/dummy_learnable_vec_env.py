import numpy as np
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

class DummyLearnableVecEnv(DummyVecEnv):
    """
    VecEnv that does runs multiple environments sequentially, that is,
    the step and reset commands are send to one environment at a time.
    Useful when debugging and when num_env == 1 (in the latter case,
    avoids communication overhead)
    """
    def __init__(self, env_fns):
        """
        Arguments:

        env_fns: iterable of callables      functions that build environments
        """
        self.__init__(env_fns)
   