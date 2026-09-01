"""Vectorized environments in JAX: a pure functional core plus a stateful shell.

An environment supplies two *single-environment* pure functions -- ``reset(key)``
and ``step(key, state, action)`` -- which :func:`vec_reset` / :func:`vec_step` ``vmap``
over ``num_envs`` and wrap in exactly the episode bookkeeping the Warp backend
does in kernels: return/length accumulation, the time limit, and same-step
auto-reset with the pre-reset observation kept for bootstrapping.

Those functions are jittable and are what the PPO trainer scans over.
:class:`JaxVecEnv` wraps them in the same Python API the Warp environments
expose (``reset`` / ``step`` / ``render_state`` / ``pop_episode_stats``), so the
renderers, the evaluation helper and ``play.py`` work with either backend.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

try:  # keep Gymnasium optional -- only used for the space objects
    from gymnasium import spaces as _gym_spaces
except Exception:  # pragma: no cover
    _gym_spaces = None


@struct.dataclass
class VecState:
    """Per-environment physics state plus the episode counters."""

    physics: Any
    obs: jax.Array  # (N, obs_dim)
    steps: jax.Array  # (N,) int32
    ep_return: jax.Array  # (N,) float32
    key: jax.Array


def _select(mask: jax.Array, new, old):
    """``where(mask, new, old)`` over a pytree of leading-dim-N leaves."""
    return jax.tree.map(lambda a, b: jnp.where(mask.reshape(mask.shape + (1,) * (a.ndim - 1)), a, b), new, old)


def vec_reset(env, key: jax.Array, num_envs: int) -> VecState:
    """Reset every environment."""
    key, subkey = jax.random.split(key)
    physics, obs = jax.vmap(env.reset)(jax.random.split(subkey, num_envs))
    return VecState(
        physics=physics,
        obs=obs,
        steps=jnp.zeros(num_envs, jnp.int32),
        ep_return=jnp.zeros(num_envs, jnp.float32),
        key=key,
    )


def vec_step(env, state: VecState, actions: jax.Array, *, max_episode_steps: int, autoreset: bool = True):
    """Step every environment and apply the shared episode bookkeeping.

    Returns ``(state, transition)`` where ``transition`` is a dict with the
    post-auto-reset ``obs``, the pre-auto-reset ``final_obs``, ``reward``,
    ``terminated``, ``truncated``, and the return/length of any episode that
    finished on this step (``done_return`` / ``done_length``, zero elsewhere).
    """
    num_envs = state.steps.shape[0]
    key, step_key, reset_key = jax.random.split(state.key, 3)

    physics, final_obs, reward, terminated = jax.vmap(env.step)(
        jax.random.split(step_key, num_envs), state.physics, actions
    )

    steps = state.steps + 1
    truncated = jnp.where((terminated == 0.0) & (steps >= max_episode_steps), 1.0, 0.0)
    done = jnp.maximum(terminated, truncated)
    ep_return = state.ep_return + reward

    if autoreset:
        fresh_physics, fresh_obs = jax.vmap(env.reset)(jax.random.split(reset_key, num_envs))
        mask = done > 0.0
        physics = _select(mask, fresh_physics, physics)
        obs = jnp.where(mask[:, None], fresh_obs, final_obs)
        next_steps = jnp.where(mask, 0, steps)
        next_return = jnp.where(mask, 0.0, ep_return)
    else:
        obs = final_obs
        next_steps, next_return = steps, ep_return

    state = VecState(physics=physics, obs=obs, steps=next_steps, ep_return=next_return, key=key)
    transition = {
        "obs": obs,
        "final_obs": final_obs,
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        "done": done,
        "done_return": jnp.where(done > 0.0, ep_return, 0.0),
        "done_length": jnp.where(done > 0.0, steps, 0).astype(jnp.float32),
    }
    return state, transition


class JaxVecEnv:
    """Stateful shell around a functional environment.

    Mirrors :class:`warp_rl.vec_env.WarpVecEnv`'s API so that everything in
    ``rl_common`` -- renderers, greedy evaluation, ``play.py`` -- is shared.
    """

    env_id: str = "jax-env"
    obs_dim: int = 0
    num_actions: int = 0
    env_cls: type = None
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        num_envs: int = 1,
        *,
        max_episode_steps: int,
        autoreset: bool = True,
        seed: int = 0,
        device: Any = None,  # accepted for API parity with the Warp backend
        **env_kwargs,
    ):
        self.num_envs = int(num_envs)
        self.max_episode_steps = int(max_episode_steps)
        self.autoreset = bool(autoreset)
        self.env = self.env_cls(**env_kwargs)
        self.device = jax.devices()[0] if device is None else device

        self._reset_fn = jax.jit(lambda key: vec_reset(self.env, key, self.num_envs))
        self._step_fn = jax.jit(
            lambda state, actions: vec_step(
                self.env, state, actions, max_episode_steps=self.max_episode_steps, autoreset=self.autoreset
            )
        )
        self.state: VecState | None = None
        self._stats = [0.0, 0.0, 0]
        self._seed(seed)
        self._build_spaces()

    @property
    def obs(self):
        """Current observations (the Warp backend exposes the same attribute)."""
        return self.state.obs

    # -- helpers ------------------------------------------------------------

    def _seed(self, seed: int) -> None:
        self.key = jax.random.PRNGKey(int(seed))

    def _build_spaces(self) -> None:
        high = np.asarray(self.env.observation_high(), dtype=np.float32)
        low = np.asarray(self.env.observation_low(), dtype=np.float32)
        if _gym_spaces is None:  # pragma: no cover
            self.single_observation_space = None
            self.single_action_space = None
            self.observation_space = None
            self.action_space = None
            return
        self.single_observation_space = _gym_spaces.Box(low, high, dtype=np.float32)
        self.single_action_space = _gym_spaces.Discrete(self.num_actions)
        self.observation_space = _gym_spaces.Box(
            np.tile(low, (self.num_envs, 1)), np.tile(high, (self.num_envs, 1)), dtype=np.float32
        )
        self.action_space = _gym_spaces.MultiDiscrete(np.full(self.num_envs, self.num_actions, dtype=np.int64))

    # -- Gymnasium API ------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._seed(seed)
        self.key, subkey = jax.random.split(self.key)
        self.state = self._reset_fn(subkey)
        self._stats = [0.0, 0.0, 0]
        return self.state.obs, {}

    def step(self, actions):
        actions = jnp.asarray(actions, dtype=jnp.int32).reshape(self.num_envs)
        self.state, transition = self._step_fn(self.state, actions)

        done = np.asarray(transition["done"]) > 0
        if done.any():
            self._stats[0] += float(np.asarray(transition["done_return"])[done].sum())
            self._stats[1] += float(np.asarray(transition["done_length"])[done].sum())
            self._stats[2] += int(done.sum())

        return (
            transition["obs"],
            transition["reward"],
            transition["terminated"],
            transition["truncated"],
            {"final_observation": transition["final_obs"]},
        )

    # -- rendering / logging -------------------------------------------------

    def render_state(self) -> dict:
        return {
            "obs": np.asarray(self.state.obs),
            "steps": np.asarray(self.state.steps),
            "ep_return": np.asarray(self.state.ep_return),
        }

    def pop_episode_stats(self) -> tuple[float, float, int]:
        ret_sum, len_sum, count = self._stats
        self._stats = [0.0, 0.0, 0]
        if count == 0:
            return float("nan"), float("nan"), 0
        return ret_sum / count, len_sum / count, count

    def close(self) -> None:
        pass
