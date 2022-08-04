from learnable_environment.envs.registration import make
from learnable_environment.envs.vec_env.learnable_sync_vector_env import LearnableSyncVectorEnv
from typing import Optional, Dict, Any

def make_vec_env(
    env_name: str,
    n_envs: int = 1,
    env_kwargs: Optional[Dict[str, Any]] = None
    ) -> LearnableSyncVectorEnv:
    env_kwargs = {} if env_kwargs is None else env_kwargs

    def make_env(idx):
        def _init():
            return make(env_name, **env_kwargs)

        return _init

    return LearnableSyncVectorEnv([make_env(i) for i in range(n_envs)])
