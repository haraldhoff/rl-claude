"""Gymnasium views of our vectorized environments.

Our environments are natively vectorized and device-resident, which is not what
the wider ecosystem expects.  :class:`GymEnv` presents a single environment
through the standard ``gymnasium.Env`` API (host numpy in, host numpy out), so
our physics can be handed to anything that speaks Gymnasium -- SB3, RLlib, a
notebook -- and :func:`register` publishes them under ids like
``WarpCartPole-v0`` / ``JaxLunarLander-v0`` for ``gymnasium.make``.

This is a thin shell: one vectorized environment with ``num_envs=1`` and
auto-reset switched off, which is exactly Gymnasium's single-env contract.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .arrays import to_numpy
from .registry import env_ids, make, normalize_id, spec

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover
    gym = None

_GYM_NAMES = {"cartpole": "CartPole", "lunarlander": "LunarLander", "mountaincar": "MountainCar"}
_BACKEND_PREFIX = {"warp": "Warp", "jax": "Jax"}


class GymEnv(gym.Env if gym is not None else object):
    """One of our environments as a standard ``gymnasium.Env``."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        env_id: str,
        *,
        backend: str = "warp",
        max_episode_steps: int | None = None,
        render_mode: str | None = None,
        seed: int = 0,
        **env_kwargs: Any,
    ):
        if gym is None:  # pragma: no cover
            raise ImportError("the Gymnasium API adapter requires gymnasium")
        self.env_id = normalize_id(env_id)
        self.backend = backend
        self.spec_entry = spec(self.env_id)
        self.vec = make(
            self.env_id,
            1,
            backend=backend,
            autoreset=False,  # Gymnasium's contract: the caller resets
            max_episode_steps=max_episode_steps or self.spec_entry.max_episode_steps,
            seed=seed,
            **env_kwargs,
        )
        self.observation_space = self.vec.single_observation_space
        self.action_space = self.vec.single_action_space
        self.render_mode = render_mode
        self.metadata = {**self.metadata, "render_fps": self.spec_entry.render_fps}
        self._renderer = None
        self.vec.reset()

    # -- Gymnasium API ------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, _ = self.vec.reset(seed=seed)
        return np.asarray(to_numpy(obs)[0], dtype=np.float32), {}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.vec.step(np.asarray([action], dtype=np.int32))
        return (
            np.asarray(to_numpy(obs)[0], dtype=np.float32),
            float(to_numpy(reward)[0]),
            bool(to_numpy(terminated)[0]),
            bool(to_numpy(truncated)[0]),
            {},
        )

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            from .registry import make_renderer

            self._renderer = make_renderer(self.env_id, self.vec, mode="rgb_array", num_render=1)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self.vec.close()


def gym_id(env_id: str, backend: str = "warp") -> str:
    """The Gymnasium id this environment is registered under, e.g. ``WarpCartPole-v0``."""
    return f"{_BACKEND_PREFIX[backend]}{_GYM_NAMES[normalize_id(env_id)]}-v0"


def register(backends=("warp", "jax")) -> list[str]:
    """Register every environment with Gymnasium; returns the ids."""
    if gym is None:  # pragma: no cover
        raise ImportError("registering requires gymnasium")
    ids = []
    for backend in backends:
        for env_id in env_ids():
            identifier = gym_id(env_id, backend)
            if identifier not in gym.registry:
                gym.register(
                    id=identifier,
                    entry_point="rl_common.gym_api:GymEnv",
                    kwargs={"env_id": env_id, "backend": backend},
                    max_episode_steps=None,  # the environment enforces its own limit
                )
            ids.append(identifier)
    return ids
