"""Backend-agnostic reinforcement learning pieces.

``rl_common`` holds everything that does not depend on how the environment or
the network is computed: the environment specifications (constants and
formulas), the PPO hyperparameters, the registry that maps an environment id to
its per-backend implementation, the pygame renderers, and the training-loop /
evaluation scaffolding.

The backends are ``warp_rl`` (Warp kernels + warp-nn), ``jax_rl``
(JAX + Flax) and ``sb3_rl`` (Stable-Baselines3's PPO on our environments through
the Gymnasium API); none is imported until asked for.
"""

from . import cli
from .agent import Agent
from .arrays import to_numpy
from .config import PPOConfig
from .registry import (
    BACKENDS,
    ENV_BACKENDS,
    EnvSpec,
    default_config,
    env_ids,
    make,
    make_agent,
    make_renderer,
    make_trainer,
    normalize_id,
    spec,
)
from .trainer import Trainer
from .training import evaluate_greedy, format_iteration, run_training_loop

__all__ = [
    "Agent",
    "cli",
    "BACKENDS",
    "ENV_BACKENDS",
    "EnvSpec",
    "PPOConfig",
    "Trainer",
    "default_config",
    "env_ids",
    "evaluate_greedy",
    "format_iteration",
    "make",
    "make_agent",
    "make_renderer",
    "make_trainer",
    "normalize_id",
    "run_training_loop",
    "spec",
    "to_numpy",
]
