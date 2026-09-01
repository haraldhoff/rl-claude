"""CartPole-v1 in JAX.

The same update rule as the Warp backend, from the same constants in
:mod:`rl_common.specs.cartpole`, written as pure functions over a single
environment; :mod:`jax_rl.vec_env` vmaps them.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from rl_common.specs import cartpole as spec

from ..vec_env import JaxVecEnv


@struct.dataclass
class CartPoleState:
    obs: jax.Array  # (4,): x, x_dot, theta, theta_dot


class CartPoleEnv:
    """Functional CartPole: ``reset(key)`` and ``step(key, state, action)``."""

    obs_dim = spec.OBS_DIM
    num_actions = spec.NUM_ACTIONS

    def observation_high(self) -> np.ndarray:
        return np.array([spec.X_THRESHOLD * 2.0, np.inf, spec.THETA_THRESHOLD * 2.0, np.inf], dtype=np.float32)

    def observation_low(self) -> np.ndarray:
        return -self.observation_high()

    def reset(self, key: jax.Array):
        obs = jax.random.uniform(key, (4,), jnp.float32, -spec.RESET_BOUND, spec.RESET_BOUND)
        return CartPoleState(obs=obs), obs

    def step(self, key: jax.Array, state: CartPoleState, action: jax.Array):
        x, x_dot, theta, theta_dot = state.obs

        force = jnp.where(action == 1, spec.FORCE_MAG, -spec.FORCE_MAG)
        costheta = jnp.cos(theta)
        sintheta = jnp.sin(theta)

        temp = (force + spec.POLEMASS_LENGTH * theta_dot * theta_dot * sintheta) / spec.TOTAL_MASS
        thetaacc = (spec.GRAVITY * sintheta - costheta * temp) / (
            spec.LENGTH * (4.0 / 3.0 - spec.MASSPOLE * costheta * costheta / spec.TOTAL_MASS)
        )
        xacc = temp - spec.POLEMASS_LENGTH * thetaacc * costheta / spec.TOTAL_MASS

        # explicit Euler, in Gymnasium's update order
        x = x + spec.TAU * x_dot
        x_dot = x_dot + spec.TAU * xacc
        theta = theta + spec.TAU * theta_dot
        theta_dot = theta_dot + spec.TAU * thetaacc

        obs = jnp.stack([x, x_dot, theta, theta_dot]).astype(jnp.float32)
        terminated = jnp.where(
            (jnp.abs(x) > spec.X_THRESHOLD) | (jnp.abs(theta) > spec.THETA_THRESHOLD), 1.0, 0.0
        ).astype(jnp.float32)
        reward = jnp.float32(1.0)
        return CartPoleState(obs=obs), obs, reward, terminated


class CartPoleVectorEnv(JaxVecEnv):
    """``num_envs`` independent CartPole-v1 environments stepped in lockstep."""

    env_id = "cartpole"
    obs_dim = spec.OBS_DIM
    num_actions = spec.NUM_ACTIONS
    env_cls = CartPoleEnv

    def __init__(self, num_envs: int = 1, *, max_episode_steps: int = spec.MAX_EPISODE_STEPS, **kwargs):
        super().__init__(num_envs, max_episode_steps=max_episode_steps, **kwargs)

    def set_state(self, state) -> None:
        """Overwrite the raw state of every environment (shape ``(num_envs, 4)``)."""
        obs = jnp.asarray(np.asarray(state, dtype=np.float32).reshape(self.num_envs, 4))
        self.state = self.state.replace(
            physics=CartPoleState(obs=obs),
            obs=obs,
            steps=jnp.zeros_like(self.state.steps),
            ep_return=jnp.zeros_like(self.state.ep_return),
        )
