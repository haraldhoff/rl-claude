"""Flax networks and the JAX backend's agent.

Same architecture and initialization as the Warp backend: two tanh MLPs
(policy and value) with orthogonal weights -- gain sqrt(2) on hidden layers,
0.01 on the policy head and 1.0 on the value head.
"""

from __future__ import annotations

import math

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.traverse_util import flatten_dict, unflatten_dict

from rl_common.agent import Agent

from .vec_env import resolve_device


class MLP(nn.Module):
    hidden: tuple[int, ...]
    out_features: int
    out_gain: float

    @nn.compact
    def __call__(self, x):
        for size in self.hidden:
            x = nn.Dense(
                size,
                kernel_init=nn.initializers.orthogonal(math.sqrt(2.0)),
                bias_init=nn.initializers.zeros,
            )(x)
            x = nn.tanh(x)
        return nn.Dense(
            self.out_features,
            kernel_init=nn.initializers.orthogonal(self.out_gain),
            bias_init=nn.initializers.zeros,
        )(x)


class ActorCriticNet(nn.Module):
    """Separate actor and critic MLPs, returning ``(logits, value)``."""

    num_actions: int
    hidden: tuple[int, ...] = (64, 64)

    @nn.compact
    def __call__(self, x):
        logits = MLP(self.hidden, self.num_actions, 0.01, name="policy")(x)
        value = MLP(self.hidden, 1, 1.0, name="value")(x)
        return logits, value[..., 0]


class ActorCritic(Agent):
    """Parameters plus the jitted apply functions (the JAX backend's agent)."""

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        *,
        hidden: tuple[int, ...] = (64, 64),
        seed: int = 0,
        device=None,  # "cpu", "cuda:0", a JAX device, or None for the default
    ):
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.hidden = tuple(hidden)
        self.device = resolve_device(device)
        self.net = ActorCriticNet(num_actions=num_actions, hidden=self.hidden)
        params = self.net.init(jax.random.PRNGKey(seed), jnp.zeros((1, obs_dim), jnp.float32))
        self.params = jax.device_put(params, self.device)
        self.key = jax.device_put(jax.random.PRNGKey(seed + 12345), self.device)
        self.apply = jax.jit(self.net.apply)
        self._greedy = jax.jit(lambda params, obs: jnp.argmax(self.net.apply(params, obs)[0], axis=-1))
        self._sample = jax.jit(lambda params, obs, key: jax.random.categorical(key, self.net.apply(params, obs)[0]))

    def act(self, obs, *, stochastic: bool = False):
        """Actions for a batch of observations (argmax, or sampled)."""
        obs = jnp.asarray(obs)
        if not stochastic:
            return self._greedy(self.params, obs)
        self.key, subkey = jax.random.split(self.key)
        return self._sample(self.params, obs, subkey)

    # -- checkpointing ------------------------------------------------------

    def save(self, path: str) -> None:
        flat = flatten_dict(self.params, sep="/")
        np.savez(path, **{k: np.asarray(v) for k, v in flat.items()})

    def load(self, path: str) -> None:
        data = np.load(path)
        flat = {k: jnp.asarray(data[k]) for k in data.files}
        self.params = jax.device_put(unflatten_dict(flat, sep="/"), self.device)
