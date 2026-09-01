"""JAX backend: environments as pure functions, networks in Flax, PPO jitted."""

from .agent import ActorCritic, ActorCriticNet
from .ppo import PPO
from .vec_env import JaxVecEnv, vec_reset, vec_step

__all__ = ["ActorCritic", "ActorCriticNet", "JaxVecEnv", "PPO", "vec_reset", "vec_step"]
