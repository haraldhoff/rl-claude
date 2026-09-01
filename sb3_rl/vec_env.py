"""Our vectorized environments as a Stable-Baselines3 ``VecEnv``.

SB3 drives ``num_envs`` environments through ``step_async`` / ``step_wait`` and
expects host numpy, ``dones = terminated | truncated``, and -- because it
auto-resets -- the pre-reset observation in ``info["terminal_observation"]``
plus ``info["TimeLimit.truncated"]`` when the episode was cut by the time limit.
That is exactly the same-step convention our environments already use, so this
adapter is a translation layer rather than a re-implementation: one step of the
whole batch, one host copy, no per-environment Python loop.

Episode returns are accumulated here and reported through
``info["episode"]``, which is what SB3's logger reads for ``ep_rew_mean``.
"""

from __future__ import annotations

import time

import numpy as np
from stable_baselines3.common.vec_env import VecEnv

from rl_common import to_numpy


class VecEnvAdapter(VecEnv):
    """Wrap one of our vectorized environments for SB3."""

    def __init__(self, env, *, render_mode: str | None = None):
        self.env = env
        self.render_mode = render_mode  # set first: SB3's VecEnv warns if it is missing
        super().__init__(env.num_envs, env.single_observation_space, env.single_action_space)
        self._actions = None
        self._returns = np.zeros(env.num_envs, dtype=np.float64)
        self._lengths = np.zeros(env.num_envs, dtype=np.int64)
        self._start = time.time()

    # -- VecEnv API ---------------------------------------------------------

    def reset(self) -> np.ndarray:
        obs, _ = self.env.reset()
        self._returns[:] = 0.0
        self._lengths[:] = 0
        return np.array(to_numpy(obs), dtype=np.float32, copy=True)

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = np.asarray(actions, dtype=np.int32).reshape(self.num_envs)

    def step_wait(self):
        obs, reward, terminated, truncated, info = self.env.step(self._actions)

        # copies, not views: torch rejects read-only arrays, and our device
        # buffers are reused on the next step anyway
        obs = np.array(to_numpy(obs), dtype=np.float32, copy=True)
        rewards = np.array(to_numpy(reward), dtype=np.float32, copy=True)
        term = to_numpy(terminated) > 0
        trunc = to_numpy(truncated) > 0
        dones = term | trunc

        self._returns += rewards
        self._lengths += 1

        infos: list[dict] = [{} for _ in range(self.num_envs)]
        if dones.any():
            final_obs = np.array(to_numpy(info["final_observation"]), dtype=np.float32, copy=True)
            elapsed = time.time() - self._start
            for i in np.flatnonzero(dones):
                infos[i]["terminal_observation"] = final_obs[i]
                if trunc[i]:
                    infos[i]["TimeLimit.truncated"] = True
                infos[i]["episode"] = {
                    "r": float(self._returns[i]),
                    "l": int(self._lengths[i]),
                    "t": round(elapsed, 6),
                }
            self._returns[dones] = 0.0
            self._lengths[dones] = 0

        return obs, rewards, dones, infos

    def close(self) -> None:
        self.env.close()

    # -- the rest of the VecEnv surface -------------------------------------

    def get_attr(self, attr_name: str, indices=None) -> list:
        # render_mode belongs to the adapter; everything else to the environment
        target = self if attr_name == "render_mode" else self.env
        return [getattr(target, attr_name)] * len(self._indices(indices))

    def set_attr(self, attr_name: str, value, indices=None) -> None:
        setattr(self.env, attr_name, value)

    def env_method(self, method_name: str, *args, indices=None, **kwargs) -> list:
        method = getattr(self.env, method_name)
        return [method(*args, **kwargs)] * len(self._indices(indices))

    def env_is_wrapped(self, wrapper_class, indices=None) -> list[bool]:
        return [False] * len(self._indices(indices))

    def _indices(self, indices) -> list[int]:
        if indices is None:
            return list(range(self.num_envs))
        if isinstance(indices, int):
            return [indices]
        return list(indices)

    def seed(self, seed: int | None = None) -> list:
        if seed is not None:
            self.env.reset(seed=seed)
        return [seed] * self.num_envs

    def render(self, mode: str | None = None):  # pragma: no cover - SB3 rendering is unused here
        return None
