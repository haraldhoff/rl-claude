"""Hyperparameters, shared by every backend."""

from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass
class PPOConfig:
    env_id: str = "cartpole"
    backend: str = "warp"  # which PPO implementation trains: warp, jax or sb3
    env_backend: str = ""  # which environment implementation it trains on
    env_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
    num_envs: int = 256
    num_steps: int = 32
    total_timesteps: int = 500_000
    learning_rate: float = 1e-3
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 8
    update_epochs: int = 10
    clip_coef: float = 0.2
    ent_coef: float = 0.005
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    norm_adv: bool = True
    hidden: tuple[int, ...] = (64, 64)
    max_episode_steps: int = 500
    seed: int = 0
    use_graph: bool = True  # warp backend: capture rollout/update epochs as CUDA graphs

    @property
    def resolved_env_backend(self) -> str:
        """Environment implementation to use.

        Defaults to the trainer's own (``warp`` trains on Warp environments,
        ``jax`` on JAX ones); SB3 has no environments of its own, so it trains
        on the Warp ones unless told otherwise.
        """
        if self.env_backend:
            return self.env_backend
        return "warp" if self.backend == "sb3" else self.backend

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.num_steps

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches

    @property
    def num_iterations(self) -> int:
        return max(1, self.total_timesteps // self.batch_size)

    def learning_rate_at(self, iteration: int) -> float:
        """Linearly annealed learning rate for a 1-based iteration index."""
        if not self.anneal_lr:
            return self.learning_rate
        return self.learning_rate * (1.0 - (iteration - 1) / self.num_iterations)
