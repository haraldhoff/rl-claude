"""The agent interface every backend implements.

An agent is a policy plus a value function that knows how to act, and how to be
saved and reloaded.  Backends differ in what an "observation" is -- a
``wp.array`` on the GPU, a JAX array, host numpy -- so the contract is written
in terms of *that backend's* arrays, with :meth:`Agent.act_numpy` as the common
host-side entry point used by the Gymnasium cross-checks and ``play.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .arrays import to_numpy


class Agent(ABC):
    """Policy + value function, in whatever form a backend represents them."""

    obs_dim: int
    num_actions: int

    @abstractmethod
    def act(self, obs, *, stochastic: bool = False):
        """Actions for a batch of this backend's observations."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Write the weights to ``path``."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Read the weights from ``path``."""

    def to_backend(self, obs: np.ndarray):
        """Host observations in this backend's array type (numpy by default)."""
        return obs

    def act_numpy(self, obs, *, stochastic: bool = False) -> np.ndarray:
        """Actions for host observations, as numpy.

        The one entry point that works the same on every backend, which is what
        the Gymnasium cross-check and the Box2D transfer test use.
        """
        obs = np.ascontiguousarray(np.asarray(obs, dtype=np.float32)).reshape(-1, self.obs_dim)
        return np.asarray(to_numpy(self.act(self.to_backend(obs), stochastic=stochastic)), dtype=np.int32)
