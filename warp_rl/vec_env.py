"""Base class for vectorized environments written in Warp kernels.

Everything that is the same for every environment lives here: the device
buffers, RNG state, episode bookkeeping (return/length accumulation,
truncation, auto-reset, on-device statistics) and the Gymnasium-style API.

An environment subclass owns its own physics state and supplies two hooks:

``_step(actions)``
    Advance the physics.  Writes ``final_obs`` (the observation *before* any
    auto-reset), ``rewards`` and ``terminated``; ``truncated`` is handled here
    unless the environment sets it itself.

``_reset()``
    Re-initialize every environment whose ``needs_reset`` flag is set and write
    its first observation into ``obs``.

Auto-reset is *same-step* (the SB3/EnvPool convention): a finished environment
is reset inside the same ``step`` call, and ``info["final_observation"]`` keeps
the true next observation so truncated episodes can still be bootstrapped.
"""

from __future__ import annotations

import numpy as np
import warp as wp

try:  # keep Gymnasium optional -- only used for the space objects
    from gymnasium import spaces as _gym_spaces
except Exception:  # pragma: no cover
    _gym_spaces = None


@wp.kernel(enable_backward=False)
def seed_kernel(seed: wp.int32, rng_states: wp.array(dtype=wp.uint32)):
    i = wp.tid()
    rng_states[i] = wp.rand_init(seed, i)


@wp.kernel(enable_backward=False)
def fill_mask_kernel(value: wp.int32, mask: wp.array(dtype=wp.int32)):
    mask[wp.tid()] = value


@wp.kernel(enable_backward=False)
def bookkeeping_kernel(
    rewards: wp.array(dtype=wp.float32),
    terminated: wp.array(dtype=wp.float32),
    truncated: wp.array(dtype=wp.float32),
    max_episode_steps: wp.int32,
    autoreset: wp.int32,
    steps: wp.array(dtype=wp.int32),
    ep_return: wp.array(dtype=wp.float32),
    ep_length: wp.array(dtype=wp.int32),
    stats: wp.array(dtype=wp.float32),  # [return_sum, length_sum, episode_count]
    needs_reset: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    n = steps[i] + 1
    steps[i] = n

    term = terminated[i]
    trunc = truncated[i]
    if term == 0.0 and n >= max_episode_steps:
        trunc = 1.0
        truncated[i] = trunc

    ret = ep_return[i] + rewards[i]
    ep_return[i] = ret
    ep_length[i] = n

    if wp.max(term, trunc) > 0.0:
        wp.atomic_add(stats, 0, ret)
        wp.atomic_add(stats, 1, float(n))
        wp.atomic_add(stats, 2, 1.0)
        needs_reset[i] = autoreset
    else:
        needs_reset[i] = 0


@wp.kernel(enable_backward=False)
def clear_counters_kernel(
    needs_reset: wp.array(dtype=wp.int32),
    steps: wp.array(dtype=wp.int32),
    ep_return: wp.array(dtype=wp.float32),
    ep_length: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    if needs_reset[i] != 0:
        steps[i] = 0
        ep_return[i] = 0.0
        ep_length[i] = 0


class WarpVecEnv:
    """``num_envs`` copies of an environment, stepped in lockstep on the device.

    Observations, rewards and the termination flags are device-resident
    ``wp.array`` objects (``float32``; the flags are 0.0/1.0 so they can feed
    the GAE kernel directly).  The arrays are reused every step -- copy them if
    you need to keep a step's values.
    """

    env_id: str = "warp-env"
    obs_dim: int = 0
    num_actions: int = 0
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        num_envs: int = 1,
        *,
        max_episode_steps: int,
        autoreset: bool = True,
        device: str | wp.Device | None = None,
        seed: int = 0,
    ):
        self.num_envs = int(num_envs)
        self.device = wp.get_device(device)
        self.autoreset = bool(autoreset)
        self.max_episode_steps = int(max_episode_steps)

        n, d = self.num_envs, self.device
        self.obs = wp.zeros((n, self.obs_dim), dtype=wp.float32, device=d)
        self.final_obs = wp.zeros((n, self.obs_dim), dtype=wp.float32, device=d)
        self.rewards = wp.zeros(n, dtype=wp.float32, device=d)
        self.terminated = wp.zeros(n, dtype=wp.float32, device=d)
        self.truncated = wp.zeros(n, dtype=wp.float32, device=d)
        self.steps = wp.zeros(n, dtype=wp.int32, device=d)
        self.ep_return = wp.zeros(n, dtype=wp.float32, device=d)
        self.ep_length = wp.zeros(n, dtype=wp.int32, device=d)
        self.needs_reset = wp.zeros(n, dtype=wp.int32, device=d)
        self.rng_states = wp.zeros(n, dtype=wp.uint32, device=d)
        self._stats = wp.zeros(3, dtype=wp.float32, device=d)

        self._seed(seed)
        self._build_spaces()

    # -- subclass hooks -----------------------------------------------------

    def observation_high(self) -> np.ndarray:
        """Upper bound of a single observation (the lower bound is its negation)."""
        return np.full(self.obs_dim, np.inf, dtype=np.float32)

    def _reset(self) -> None:
        raise NotImplementedError

    def _step(self, actions: wp.array) -> None:
        raise NotImplementedError

    # -- helpers ------------------------------------------------------------

    def _build_spaces(self) -> None:
        high = np.asarray(self.observation_high(), dtype=np.float32)
        if _gym_spaces is None:  # pragma: no cover
            self.single_observation_space = None
            self.single_action_space = None
            self.observation_space = None
            self.action_space = None
            return
        self.single_observation_space = _gym_spaces.Box(-high, high, dtype=np.float32)
        self.single_action_space = _gym_spaces.Discrete(self.num_actions)
        self.observation_space = _gym_spaces.Box(
            np.tile(-high, (self.num_envs, 1)), np.tile(high, (self.num_envs, 1)), dtype=np.float32
        )
        self.action_space = _gym_spaces.MultiDiscrete(np.full(self.num_envs, self.num_actions, dtype=np.int64))

    def _seed(self, seed: int) -> None:
        wp.launch(seed_kernel, dim=self.num_envs, inputs=[int(seed), self.rng_states], device=self.device)

    def _fill_needs_reset(self, value: int) -> None:
        wp.launch(fill_mask_kernel, dim=self.num_envs, inputs=[int(value), self.needs_reset], device=self.device)

    def _as_action_array(self, actions) -> wp.array:
        if isinstance(actions, wp.array):
            if actions.dtype != wp.int32:
                raise TypeError(f"actions must be int32, got {actions.dtype}")
            return actions
        arr = np.ascontiguousarray(np.asarray(actions, dtype=np.int32)).reshape(self.num_envs)
        return wp.array(arr, dtype=wp.int32, device=self.device)

    # -- Gymnasium API ------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Reset every environment.  Returns ``(obs, info)``."""
        if seed is not None:
            self._seed(seed)
        self._fill_needs_reset(1)
        self._reset()
        wp.launch(
            clear_counters_kernel,
            dim=self.num_envs,
            inputs=[self.needs_reset, self.steps, self.ep_return, self.ep_length],
            device=self.device,
        )
        self._stats.zero_()
        return self.obs, {}

    def step(self, actions):
        """Step every environment.  Returns ``(obs, reward, terminated, truncated, info)``."""
        self._step(self._as_action_array(actions))
        wp.launch(
            bookkeeping_kernel,
            dim=self.num_envs,
            inputs=[
                self.rewards,
                self.terminated,
                self.truncated,
                self.max_episode_steps,
                1 if self.autoreset else 0,
                self.steps,
                self.ep_return,
                self.ep_length,
                self._stats,
                self.needs_reset,
            ],
            device=self.device,
        )
        if self.autoreset:
            self._reset()
            wp.launch(
                clear_counters_kernel,
                dim=self.num_envs,
                inputs=[self.needs_reset, self.steps, self.ep_return, self.ep_length],
                device=self.device,
            )
        return self.obs, self.rewards, self.terminated, self.truncated, {"final_observation": self.final_obs}

    # -- logging ------------------------------------------------------------

    def pop_episode_stats(self) -> tuple[float, float, int]:
        """Mean return / length over the episodes finished since the last call."""
        ret_sum, len_sum, count = (float(v) for v in self._stats.numpy())
        self._stats.zero_()
        if count == 0:
            return float("nan"), float("nan"), 0
        return ret_sum / count, len_sum / count, int(count)

    def close(self) -> None:
        pass
