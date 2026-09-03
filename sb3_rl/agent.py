"""The SB3 backend's agent facade.

Wraps a Stable-Baselines3 ``ActorCriticPolicy`` in the same ``act`` /
``act_numpy`` / ``save`` / ``load`` interface the Warp and JAX agents expose, so
``play.py``, the renderers and the evaluation helper do not care which backend
produced the policy.

Checkpoints hold the policy (SB3's own ``policy.save`` format, a torch pickle),
not the optimizer or the rollout buffer -- the same convention as the other two
backends.  For a full SB3 checkpoint use ``trainer.model.save(path)``.
"""

from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy

from rl_common.agent import Agent


class ActorCritic(Agent):
    """An SB3 policy behind our uniform agent interface."""

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        *,
        hidden: tuple[int, ...] = (64, 64),
        seed: int = 0,
        device=None,
    ):
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.hidden = tuple(hidden)
        self.device = resolve_device(device)
        torch.manual_seed(seed)

        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(num_actions)
        self.policy = make_policy(self.observation_space, self.action_space, self.hidden).to(self.device)

    # -- uniform agent interface --------------------------------------------

    def act(self, obs, *, stochastic: bool = False) -> np.ndarray:
        from rl_common import to_numpy

        obs = np.asarray(to_numpy(obs), dtype=np.float32).reshape(-1, self.obs_dim)
        actions, _ = self.policy.predict(obs, deterministic=not stochastic)
        return np.asarray(actions, dtype=np.int32)

    def save(self, path: str) -> None:
        self.policy.save(path)

    def load(self, path: str) -> None:
        self.policy = ActorCriticPolicy.load(path, device=self.device)


def resolve_device(device) -> str:
    """Map a Warp-style device string ("cuda:0", "cpu", None) onto torch's.

    The default is CPU: these are small MLP policies, and SB3 itself recommends
    the CPU for them -- the environment stays on whichever device its own
    backend uses.
    """
    if device is None:
        return "cpu"
    name = str(device)
    return name if name.startswith("cuda") else "cpu"  # keep the index: "cuda:1" is a torch device


def policy_kwargs(hidden: tuple[int, ...]) -> dict:
    """The policy shape shared by the agent facade and the trainer.

    Separate tanh MLPs for actor and critic with orthogonal initialization --
    SB3's default, and ours.  The Adam epsilon is ours: SB3 defaults to 1e-5
    (the OpenAI-baselines value) where warp-nn and optax use 1e-8, and on a
    sparse-reward task that difference decides the run.
    """
    return dict(
        net_arch=dict(pi=list(hidden), vf=list(hidden)),
        activation_fn=torch.nn.Tanh,
        ortho_init=True,
        optimizer_kwargs=dict(eps=1e-8),
    )


def make_policy(observation_space, action_space, hidden: tuple[int, ...]) -> ActorCriticPolicy:
    """A standalone policy of that shape (the trainer gets its own from SB3)."""
    return ActorCriticPolicy(
        observation_space,
        action_space,
        lr_schedule=lambda _: 1e-3,  # replaced by the trainer's schedule
        **policy_kwargs(hidden),
    )
