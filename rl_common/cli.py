"""Command-line plumbing shared by ``train.py`` and ``play.py``.

Both scripts choose an environment, a backend and a set of PPO overrides; only
what they do afterwards differs.  The overrides all default to ``None`` so that
:func:`config_from_args` can let each environment's recommended value win unless
the flag was actually given.
"""

from __future__ import annotations

import argparse

from .config import PPOConfig
from .registry import BACKENDS, ENV_BACKENDS, default_config, env_ids

# PPOConfig fields that can be overridden from the command line, as
# ``--flag-name`` -> type
_OVERRIDES = {
    "num_envs": int,
    "num_steps": int,
    "total_timesteps": int,
    "learning_rate": float,
    "gamma": float,
    "gae_lambda": float,
    "num_minibatches": int,
    "update_epochs": int,
    "clip_coef": float,
    "ent_coef": float,
    "vf_coef": float,
    "max_grad_norm": float,
    "max_episode_steps": int,
    "hidden": None,  # handled separately: a list of ints
}


def add_arguments(parser: argparse.ArgumentParser, *, overrides: bool = True) -> argparse.ArgumentParser:
    """Add the environment/backend selection (and optionally the PPO overrides)."""
    parser.add_argument("--env", type=str, default="cartpole", help=f"environment id: {', '.join(env_ids())}")
    parser.add_argument(
        "--backend", type=str, default="warp", choices=BACKENDS, help="which PPO/policy implementation"
    )
    parser.add_argument(
        "--env-backend",
        type=str,
        default=None,
        choices=ENV_BACKENDS,
        help="which environment implementation (default: the backend's own; sb3 uses warp)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None, help="e.g. cuda:0 or cpu")

    if overrides:
        for name, kind in _OVERRIDES.items():
            flag = "--" + name.replace("_", "-")
            if name == "hidden":
                parser.add_argument(flag, type=int, nargs="+", default=None, help="hidden layer sizes")
            else:
                parser.add_argument(flag, type=kind, default=None)
        parser.add_argument("--no-anneal-lr", action="store_true")
        parser.add_argument("--no-graph", action="store_true", help="warp backend: disable CUDA graph capture")
    return parser


def config_from_args(args: argparse.Namespace, **extra) -> PPOConfig:
    """Build a :class:`PPOConfig` from parsed arguments plus any extra overrides."""
    overrides = {name: getattr(args, name, None) for name in _OVERRIDES}
    if overrides.get("hidden") is not None:
        overrides["hidden"] = tuple(overrides["hidden"])
    if getattr(args, "no_anneal_lr", False):
        overrides["anneal_lr"] = False
    if getattr(args, "no_graph", False):
        overrides["use_graph"] = False
    overrides.update(extra)
    return default_config(
        args.env,
        backend=args.backend,
        env_backend=args.env_backend,
        seed=args.seed,
        **overrides,
    )
