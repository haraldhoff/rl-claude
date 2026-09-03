"""Environment registry: one entry per environment, shared by every backend.

An entry names the environment's shape, its recommended PPO settings, its
renderer, and the class each backend implements it with.  Everything is
referenced by ``"module:attribute"`` string and imported on demand, so using
the JAX backend never imports Warp and vice versa.

``BACKENDS`` are the PPO implementations (``warp``, ``jax``, ``sb3``);
``ENV_BACKENDS`` are the environment implementations (``warp``, ``jax``, and
``gym`` for stock Gymnasium, which only the SB3 trainer can use).
"""

from __future__ import annotations

import dataclasses
import importlib
from typing import Any

from .config import PPOConfig
from .specs import cartpole as cartpole_spec
from .specs import lunar_lander as lander_spec
from .specs import mountain_car as car_spec

BACKENDS = ("warp", "jax", "sb3")  # PPO implementations
ENV_BACKENDS = ("warp", "jax", "gym")  # environment implementations


def _load(path: str):
    module, _, attr = path.partition(":")
    return getattr(importlib.import_module(module), attr)


def _implementations(module: str, class_name: str) -> dict[str, str]:
    """Where each backend keeps its version of one environment."""
    return {backend: f"{backend}_rl.envs.{module}:{class_name}" for backend in ("warp", "jax")}


@dataclasses.dataclass(frozen=True)
class EnvSpec:
    env_id: str
    obs_dim: int
    num_actions: int
    max_episode_steps: int
    render_fps: int
    solved_return: float
    renderer: str
    backends: dict[str, str]
    ppo: dict[str, Any] = dataclasses.field(default_factory=dict)

    def env_cls(self, backend: str):
        if backend not in self.backends:
            raise KeyError(f"environment '{self.env_id}' has no {backend} implementation")
        return _load(self.backends[backend])

    def renderer_cls(self):
        return _load(self.renderer)


REGISTRY: dict[str, EnvSpec] = {
    "cartpole": EnvSpec(
        env_id="cartpole",
        obs_dim=cartpole_spec.OBS_DIM,
        num_actions=cartpole_spec.NUM_ACTIONS,
        max_episode_steps=cartpole_spec.MAX_EPISODE_STEPS,
        render_fps=50,
        solved_return=475.0,
        renderer="rl_common.render.cartpole:CartPoleRenderer",
        backends=_implementations("cartpole", "CartPoleVectorEnv"),
        ppo=dict(
            num_envs=256,
            num_steps=32,
            total_timesteps=500_000,
            learning_rate=1e-3,
            num_minibatches=8,
            update_epochs=10,
            ent_coef=0.005,
            gamma=0.99,
            gae_lambda=0.95,
            hidden=(64, 64),
        ),
    ),
    "lunarlander": EnvSpec(
        env_id="lunarlander",
        obs_dim=lander_spec.OBS_DIM,
        num_actions=lander_spec.NUM_ACTIONS,
        max_episode_steps=lander_spec.MAX_EPISODE_STEPS,
        render_fps=50,
        solved_return=200.0,
        renderer="rl_common.render.lunar_lander:LunarLanderRenderer",
        backends=_implementations("lunar_lander", "LunarLanderVectorEnv"),
        ppo=dict(
            num_envs=512,
            num_steps=64,
            total_timesteps=16_000_000,
            learning_rate=1e-3,
            num_minibatches=8,
            update_epochs=4,
            ent_coef=0.01,
            gamma=0.999,
            gae_lambda=0.98,
            hidden=(128, 128),
        ),
    ),
    "mountaincar": EnvSpec(
        env_id="mountaincar",
        obs_dim=car_spec.OBS_DIM,
        num_actions=car_spec.NUM_ACTIONS,
        # 25 decisions x 8 physics steps = MountainCar-v0's 200-step limit
        max_episode_steps=car_spec.MAX_EPISODE_STEPS // car_spec.ACTION_REPEAT,
        render_fps=30,
        solved_return=-110.0,
        renderer="rl_common.render.mountain_car:MountainCarRenderer",
        backends=_implementations("mountain_car", "MountainCarVectorEnv"),
        ppo=dict(
            env_kwargs=dict(action_repeat=car_spec.ACTION_REPEAT),
            num_envs=512,
            num_steps=32,
            total_timesteps=4_000_000,
            learning_rate=1e-3,
            num_minibatches=8,
            update_epochs=4,
            ent_coef=0.02,
            gamma=0.99,
            gae_lambda=0.95,
            hidden=(64, 64),
        ),
    ),
}

_ALIASES = {
    "cart_pole": "cartpole",
    "lunar_lander": "lunarlander",
    "lander": "lunarlander",
    "mountain_car": "mountaincar",
    "car": "mountaincar",
}

_TRAINERS = {"warp": "warp_rl.ppo:PPO", "jax": "jax_rl.ppo:PPO", "sb3": "sb3_rl.ppo:PPO"}
_AGENTS = {
    "warp": "warp_rl.agent:ActorCritic",
    "jax": "jax_rl.agent:ActorCritic",
    "sb3": "sb3_rl.agent:ActorCritic",
}


def normalize_id(env_id: str) -> str:
    key = "".join(c for c in env_id.lower() if c.isalnum() or c == "_")
    for suffix in ("v0", "v1", "v2", "v3"):
        key = key.removesuffix(suffix)
    key = key.rstrip("_-")
    key = _ALIASES.get(key, key)
    if key not in REGISTRY:
        raise KeyError(f"unknown environment '{env_id}'; available: {', '.join(sorted(REGISTRY))}")
    return key


def env_ids() -> list[str]:
    return sorted(REGISTRY)


def spec(env_id: str) -> EnvSpec:
    return REGISTRY[normalize_id(env_id)]


def make(env_id: str, num_envs: int = 1, *, backend: str = "warp", **kwargs):
    """Create a vectorized environment by id, on the given backend."""
    s = spec(env_id)
    kwargs.setdefault("max_episode_steps", s.max_episode_steps)
    return s.env_cls(backend)(num_envs, **kwargs)


def make_renderer(env_id: str, env, **kwargs):
    """Create the renderer that belongs to an environment id (backend-agnostic)."""
    s = spec(env_id)
    kwargs.setdefault("fps", s.render_fps)
    return s.renderer_cls()(env, **kwargs)


def make_trainer(cfg: PPOConfig, **kwargs):
    """Create the PPO trainer of ``cfg.backend``."""
    if cfg.backend not in _TRAINERS:
        raise KeyError(f"unknown backend '{cfg.backend}'; available: {', '.join(BACKENDS)}")
    return _load(_TRAINERS[cfg.backend])(cfg, **kwargs)


def make_agent(env_id: str, *, backend: str = "warp", hidden: tuple[int, ...] | None = None, **kwargs):
    """Create the actor-critic of ``backend`` shaped for ``env_id``."""
    if backend not in _AGENTS:
        raise KeyError(f"unknown backend '{backend}'; available: {', '.join(BACKENDS)}")
    s = spec(env_id)
    hidden = hidden if hidden is not None else s.ppo.get("hidden", (64, 64))
    return _load(_AGENTS[backend])(s.obs_dim, s.num_actions, hidden=tuple(hidden), **kwargs)


def default_config(env_id: str, *, backend: str = "warp", **overrides) -> PPOConfig:
    """Recommended PPO settings for an environment, with optional overrides.

    Overrides that are ``None`` are ignored, so a CLI can pass every flag
    through and let the environment's recommendation win.
    """
    s = spec(env_id)
    fields = {f.name for f in dataclasses.fields(PPOConfig)}
    cfg = dict(env_id=normalize_id(env_id), backend=backend, max_episode_steps=s.max_episode_steps, **s.ppo)
    for key, value in overrides.items():
        if key not in fields:
            raise TypeError(f"unknown PPOConfig field '{key}'")
        if value is not None:
            cfg[key] = value
    return PPOConfig(**cfg)
