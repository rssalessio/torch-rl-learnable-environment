from gym.vector.vector_env import VectorEnv

class LearnableVecEnv(VectorEnv):
    def __init__(self, *args, **kwargs):
        super(VectorEnv, self).__init__(*args, **kwargs)
