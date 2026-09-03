"""Helpers shared by the test modules.

Four tests used to spell out the same short-training run: build a config with
``total_timesteps=num_envs * num_steps * n``, train with a callback that
collects ``episodic_return``, then compare the mean of the first few iterations
against the mean of the last few.  The arithmetic in particular was worth
removing -- what those tests mean is "run n iterations", and stating it that
way cannot drift out of step with the batch size the registry recommends.
"""

from __future__ import annotations

import dataclasses

import numpy as np

import rl_common


def short_config(env_id: str, backend: str, *, iterations: int, seed: int = 0, **overrides):
    """A :class:`~rl_common.PPOConfig` budgeted in *iterations* rather than steps.

    Everything else is the environment's recommendation, so a test says how long
    to run without restating how wide the batch is.
    """
    cfg = rl_common.default_config(env_id, backend=backend, seed=seed, **overrides)
    return dataclasses.replace(cfg, total_timesteps=cfg.batch_size * iterations)


def train_short(env_id: str, backend: str, *, iterations: int, **overrides):
    """Train briefly; return the trainer and its per-iteration episodic returns."""
    cfg = short_config(env_id, backend, iterations=iterations, **overrides)
    trainer = rl_common.make_trainer(cfg)
    history: list[float] = []
    trainer.train(callback=lambda stats: history.append(stats["episodic_return"]))
    return trainer, history


def assert_learns(env_id: str, backend: str, *, iterations: int, gain: float, window: int = 2, **overrides):
    """Assert a short run improves the episodic return by at least ``gain``.

    Returns ``(trainer, first, last)`` so a caller can go on to assert something
    stronger -- a greedy evaluation score, say.
    """
    trainer, history = train_short(env_id, backend, iterations=iterations, **overrides)
    first, last = float(np.mean(history[:window])), float(np.mean(history[-window:]))
    assert last > first + gain, f"{env_id} on {backend} did not learn: {first:.1f} -> {last:.1f} (need +{gain:.0f})"
    return trainer, first, last
