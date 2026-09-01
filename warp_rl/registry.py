"""Environment registry: id -> (environment, renderer, recommended PPO settings)."""

from __future__ import annotations

import dataclasses
from typing import Any

from .envs.cartpole import CartPoleRenderer, CartPoleVectorEnv
from .envs.lunar_lander import LunarLanderRenderer, LunarLanderVectorEnv
from .envs.mountain_car import MountainCarRenderer, MountainCarVectorEnv
from .ppo import PPOConfig


@dataclasses.dataclass(frozen=True)
class EnvSpec:
    env_cls: type
    renderer_cls: type
    max_episode_steps: int
    render_fps: int
    solved_return: float
    ppo: dict[str, Any] = dataclasses.field(default_factory=dict)


REGISTRY: dict[str, EnvSpec] = {
    "cartpole": EnvSpec(
        env_cls=CartPoleVectorEnv,
        renderer_cls=CartPoleRenderer,
        max_episode_steps=500,
        render_fps=50,
        solved_return=475.0,
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
        env_cls=LunarLanderVectorEnv,
        renderer_cls=LunarLanderRenderer,
        max_episode_steps=1000,
        render_fps=50,
        solved_return=200.0,
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
        env_cls=MountainCarVectorEnv,
        renderer_cls=MountainCarRenderer,
        max_episode_steps=25,  # 25 decisions x 8 physics steps = MountainCar-v0's 200
        render_fps=30,
        solved_return=-110.0,
        ppo=dict(
            env_kwargs=dict(action_repeat=8),
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

_ALIASES = {"cart_pole": "cartpole", "lunar_lander": "lunarlander", "lander": "lunarlander", "mountain_car": "mountaincar", "car": "mountaincar"}


def normalize_id(env_id: str) -> str:
    key = "".join(c for c in env_id.lower() if c.isalnum() or c == "_")
    key = key.removesuffix("v0").removesuffix("v1").removesuffix("v2").removesuffix("v3").rstrip("_-")
    key = _ALIASES.get(key, key)
    if key not in REGISTRY:
        raise KeyError(f"unknown environment '{env_id}'; available: {', '.join(sorted(REGISTRY))}")
    return key


def env_ids() -> list[str]:
    return sorted(REGISTRY)


def spec(env_id: str) -> EnvSpec:
    return REGISTRY[normalize_id(env_id)]


def make(env_id: str, num_envs: int = 1, **kwargs):
    """Create a vectorized environment by id."""
    s = spec(env_id)
    kwargs.setdefault("max_episode_steps", s.max_episode_steps)
    return s.env_cls(num_envs, **kwargs)


def make_renderer(env_id: str, env, **kwargs):
    """Create the renderer that belongs to an environment id."""
    s = spec(env_id)
    kwargs.setdefault("fps", s.render_fps)
    return s.renderer_cls(env, **kwargs)


def default_config(env_id: str, **overrides) -> PPOConfig:
    """Recommended PPO settings for an environment, with optional overrides."""
    s = spec(env_id)
    fields = {f.name for f in dataclasses.fields(PPOConfig)}
    cfg = dict(env_id=normalize_id(env_id), max_episode_steps=s.max_episode_steps, **s.ppo)
    for k, v in overrides.items():
        if k not in fields:
            raise TypeError(f"unknown PPOConfig field '{k}'")
        if v is not None:
            cfg[k] = v
    return PPOConfig(**cfg)
