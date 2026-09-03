"""The trainer skeleton every backend fills in.

A backend's PPO differs only in how one iteration is computed.  Everything
around it -- building the environment from the config, the loop with its
learning-rate annealing and logging, and the greedy evaluation at the end -- is
the same, and lives here.

A subclass sets ``self.cfg``, ``self.agent`` and ``self.device`` in its
constructor and implements :meth:`iterate`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

from .config import PPOConfig
from .registry import make
from .training import evaluate_greedy, run_training_loop


class Trainer(ABC):
    """Shared training/evaluation scaffolding for a PPO backend."""

    cfg: PPOConfig
    agent: object
    device: object = None  # where the learner runs
    env_device: object = None  # where the environment runs (None: the backend's default)
    global_step: int = 0

    # -- the one thing a backend must provide -------------------------------

    @abstractmethod
    def iterate(self, lr: float) -> dict:
        """Run one rollout plus one update; return this iteration's metrics.

        The metrics must include ``entropy``, ``approx_kl``, ``clipfrac``,
        ``value_loss`` and the episode summary (``episodic_return``,
        ``episodic_length``, ``episodes``).
        """

    # -- shared -------------------------------------------------------------

    def make_env(self, num_envs: int, **kwargs):
        """An environment of the configured kind, with the config's settings."""
        cfg = self.cfg
        kwargs.setdefault("max_episode_steps", cfg.max_episode_steps)
        kwargs.setdefault("seed", cfg.seed)
        kwargs.setdefault("device", self.env_device)
        return make(cfg.env_id, num_envs, backend=cfg.resolved_env_backend, **cfg.env_kwargs, **kwargs)

    def train(self, *, log_every: int = 1, callback: Callable[[dict], None] | None = None) -> list[dict]:
        """Train for ``cfg.num_iterations`` iterations."""
        history = run_training_loop(self.cfg, self.iterate, log_every=log_every, callback=callback)
        self.global_step = history[-1]["global_step"] if history else 0
        return history

    def evaluate(self, *, num_envs: int = 64, max_episode_steps: int | None = None, seed: int = 12345) -> dict:
        """Run the greedy (argmax) policy until every environment finishes an episode."""
        limit = max_episode_steps or self.cfg.max_episode_steps
        env = self.make_env(num_envs, max_episode_steps=limit, autoreset=False, seed=seed)
        return evaluate_greedy(env, self.agent.act, max_steps=limit)
