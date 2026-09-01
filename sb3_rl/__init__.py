"""SB3 backend: Stable-Baselines3's PPO on our environments.

Our vectorized environments reach SB3 through :class:`sb3_rl.vec_env.VecEnvAdapter`
(and the wider ecosystem through :mod:`rl_common.gym_api`); the trainer is
driven by the same shared loop as the Warp and JAX backends.
"""

from .agent import ActorCritic
from .ppo import PPO
from .vec_env import VecEnvAdapter

__all__ = ["ActorCritic", "PPO", "VecEnvAdapter"]
