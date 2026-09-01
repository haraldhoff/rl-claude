"""MountainCar-v0 in JAX (with the same ``action_repeat`` knob as the Warp backend)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from rl_common.specs import mountain_car as spec

from ..vec_env import JaxVecEnv


@struct.dataclass
class MountainCarState:
    obs: jax.Array  # (2,): position, velocity


class MountainCarEnv:
    """Functional MountainCar; ``action_repeat`` holds an action for k physics steps."""

    obs_dim = spec.OBS_DIM
    num_actions = spec.NUM_ACTIONS

    def __init__(self, action_repeat: int = 1):
        self.action_repeat = int(action_repeat)

    def observation_high(self) -> np.ndarray:
        return np.array([spec.MAX_POSITION, spec.MAX_SPEED], dtype=np.float32)

    def observation_low(self) -> np.ndarray:
        return np.array([spec.MIN_POSITION, -spec.MAX_SPEED], dtype=np.float32)

    def reset(self, key: jax.Array):
        position = jax.random.uniform(key, (), jnp.float32, spec.RESET_LOW, spec.RESET_HIGH)
        obs = jnp.stack([position, jnp.float32(0.0)])
        return MountainCarState(obs=obs), obs

    def step(self, key: jax.Array, state: MountainCarState, action: jax.Array):
        def physics_step(carry, _):
            position, velocity, terminated, reward = carry
            # once the flag is reached the episode is over: freeze the state
            live = 1.0 - terminated
            new_velocity = jnp.clip(
                velocity + (action.astype(jnp.float32) - 1.0) * spec.FORCE + jnp.cos(3.0 * position) * -spec.GRAVITY,
                -spec.MAX_SPEED,
                spec.MAX_SPEED,
            )
            new_position = jnp.clip(position + new_velocity, spec.MIN_POSITION, spec.MAX_POSITION)
            new_velocity = jnp.where((new_position <= spec.MIN_POSITION) & (new_velocity < 0.0), 0.0, new_velocity)
            hit = jnp.where(
                (new_position >= spec.GOAL_POSITION) & (new_velocity >= spec.GOAL_VELOCITY), 1.0, 0.0
            )
            position = jnp.where(live > 0, new_position, position)
            velocity = jnp.where(live > 0, new_velocity, velocity)
            terminated = jnp.maximum(terminated, live * hit)
            return (position, velocity, terminated, reward - live), None

        position, velocity = state.obs
        (position, velocity, terminated, reward), _ = jax.lax.scan(
            physics_step,
            (position, velocity, jnp.float32(0.0), jnp.float32(0.0)),
            None,
            length=self.action_repeat,
        )
        obs = jnp.stack([position, velocity]).astype(jnp.float32)
        return MountainCarState(obs=obs), obs, reward, terminated


class MountainCarVectorEnv(JaxVecEnv):
    """``num_envs`` independent MountainCar-v0 environments stepped in lockstep."""

    env_id = "mountaincar"
    obs_dim = spec.OBS_DIM
    num_actions = spec.NUM_ACTIONS
    env_cls = MountainCarEnv

    def __init__(
        self,
        num_envs: int = 1,
        *,
        max_episode_steps: int = spec.MAX_EPISODE_STEPS,
        action_repeat: int = 1,
        **kwargs,
    ):
        self.action_repeat = int(action_repeat)
        super().__init__(num_envs, max_episode_steps=max_episode_steps, action_repeat=action_repeat, **kwargs)

    @property
    def physics_steps_per_episode(self) -> int:
        """The MountainCar-v0 time limit is 200 *physics* steps."""
        return self.max_episode_steps * self.action_repeat

    def set_state(self, state) -> None:
        """Overwrite ``(position, velocity)`` for every environment."""
        obs = jnp.asarray(np.asarray(state, dtype=np.float32).reshape(self.num_envs, 2))
        self.state = self.state.replace(
            physics=MountainCarState(obs=obs),
            obs=obs,
            steps=jnp.zeros_like(self.state.steps),
            ep_return=jnp.zeros_like(self.state.ep_return),
        )
