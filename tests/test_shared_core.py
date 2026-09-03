"""The backend-agnostic surface: the CLI, the registry, and checkpoint portability.

``rl_common.cli`` and ``rl_common.registry`` decide what every run actually
trains -- which environment, on which backend, with which hyperparameters -- and
``tools/convert_weights.py`` is what makes a checkpoint mean the same thing on
either backend.  None of it involves physics, so none of it needs a GPU except
the round-trip test, which loads both backends' agents.

Run with pytest, or directly:  python tests/test_shared_core.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

import rl_common
from rl_common import cli

# ---------------------------------------------------------------------------
# the registry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("env_id", rl_common.env_ids())
def test_every_environment_has_a_complete_spec(env_id):
    """A registry entry must fully describe an environment for every backend."""
    s = rl_common.spec(env_id)
    assert s.obs_dim > 0 and s.num_actions > 0
    assert s.max_episode_steps > 0 and s.render_fps > 0
    assert set(s.backends) == {"warp", "jax"}, f"{env_id} is missing a backend implementation"
    assert s.renderer.startswith("rl_common.render."), s.renderer
    # the recommended settings must be real PPOConfig fields
    cfg = rl_common.default_config(env_id)
    assert cfg.env_id == env_id
    assert cfg.batch_size == cfg.num_envs * cfg.num_steps
    assert cfg.minibatch_size * cfg.num_minibatches == cfg.batch_size, "batch must divide into whole minibatches"


@pytest.mark.parametrize(
    "alias,expected",
    [
        ("cartpole", "cartpole"),
        ("CartPole-v1", "cartpole"),
        ("cart_pole", "cartpole"),
        ("lander", "lunarlander"),
        ("LunarLander-v3", "lunarlander"),
        ("mountain_car", "mountaincar"),
        ("MountainCar-v0", "mountaincar"),
        ("car", "mountaincar"),
    ],
)
def test_environment_ids_normalize(alias, expected):
    assert rl_common.normalize_id(alias) == expected


def test_unknown_environment_is_rejected():
    with pytest.raises(KeyError):
        rl_common.normalize_id("pendulum")


def test_unknown_backend_is_rejected():
    cfg = rl_common.default_config("cartpole", backend="tensorflow")
    with pytest.raises(KeyError):
        rl_common.make_trainer(cfg)


# ---------------------------------------------------------------------------
# the CLI
# ---------------------------------------------------------------------------


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = cli.add_arguments(argparse.ArgumentParser())
    return parser.parse_args(argv)


def test_bare_cli_uses_the_environment_recommendations():
    """With no overrides, every value must come from the registry."""
    cfg = cli.config_from_args(_parse(["--env", "lunarlander"]))
    recommended = rl_common.spec("lunarlander").ppo
    assert cfg.backend == "warp"
    for name, value in recommended.items():
        if name == "hidden":
            assert tuple(cfg.hidden) == tuple(value)
        elif name != "env_kwargs":
            assert getattr(cfg, name) == value, name


def test_cli_overrides_win_over_recommendations():
    cfg = cli.config_from_args(
        _parse(
            [
                "--env", "cartpole",
                "--backend", "jax",
                "--num-envs", "64",
                "--num-steps", "16",
                "--learning-rate", "3e-4",
                "--gamma", "0.5",
                "--ent-coef", "0.0",
                "--hidden", "32", "16",
                "--seed", "7",
            ]
        )
    )
    assert (cfg.backend, cfg.num_envs, cfg.num_steps, cfg.seed) == ("jax", 64, 16, 7)
    assert cfg.learning_rate == pytest.approx(3e-4)
    assert cfg.gamma == pytest.approx(0.5)
    assert cfg.ent_coef == 0.0  # a zero override must not be mistaken for "unset"
    assert tuple(cfg.hidden) == (32, 16)
    assert cfg.batch_size == 64 * 16


def test_cli_boolean_flags():
    assert cli.config_from_args(_parse([])).anneal_lr is True
    assert cli.config_from_args(_parse(["--no-anneal-lr"])).anneal_lr is False
    assert cli.config_from_args(_parse([])).use_graph is True
    assert cli.config_from_args(_parse(["--no-graph"])).use_graph is False


def test_env_backend_defaults_per_backend():
    """sb3 has no environments of its own; the other two use their own."""
    assert cli.config_from_args(_parse(["--backend", "warp"])).resolved_env_backend == "warp"
    assert cli.config_from_args(_parse(["--backend", "jax"])).resolved_env_backend == "jax"
    assert cli.config_from_args(_parse(["--backend", "sb3"])).resolved_env_backend == "warp"
    explicit = _parse(["--backend", "sb3", "--env-backend", "gym"])
    assert cli.config_from_args(explicit).resolved_env_backend == "gym"


def test_unknown_config_field_is_rejected():
    with pytest.raises(TypeError):
        rl_common.default_config("cartpole", nonexistent_field=1)


# ---------------------------------------------------------------------------
# annealing
# ---------------------------------------------------------------------------


def test_learning_rate_anneals_to_zero_over_the_run():
    cfg = rl_common.default_config("cartpole", learning_rate=1e-3)
    n = cfg.num_iterations
    assert cfg.learning_rate_at(1) == pytest.approx(1e-3)
    assert cfg.learning_rate_at(n) == pytest.approx(1e-3 / n)
    schedule = [cfg.learning_rate_at(i) for i in range(1, n + 1)]
    assert all(a > b for a, b in zip(schedule, schedule[1:])), "schedule must be strictly decreasing"

    flat = rl_common.default_config("cartpole", learning_rate=1e-3, anneal_lr=False)
    assert {flat.learning_rate_at(i) for i in range(1, flat.num_iterations + 1)} == {1e-3}


# ---------------------------------------------------------------------------
# checkpoint portability (tools/convert_weights.py)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("env_id", ["cartpole", "lunarlander"])
def test_weight_conversion_preserves_the_policy(env_id, tmp_path):
    """warp -> jax -> warp must be the identity, and the JAX copy in the middle
    must produce the same logits and values as the Warp original.

    This is what makes ``weights/*.npz`` and ``weights/jax/*.npz`` the same
    policy rather than two policies that happen to score similarly.
    """
    import convert_weights

    warp_agent = rl_common.make_agent(env_id, backend="warp")
    original = tmp_path / "warp.npz"
    warp_agent.save(str(original))

    as_jax = tmp_path / "jax.npz"
    convert_weights.warp_to_jax(env_id, str(original), str(as_jax))

    back = tmp_path / "warp_again.npz"
    convert_weights.jax_to_warp(env_id, str(as_jax), str(back))

    # the round trip is exact: a rename and a transpose, no arithmetic
    before, after = np.load(str(original)), np.load(str(back))
    assert set(before.files) == set(after.files)
    for name in before.files:
        np.testing.assert_array_equal(before[name], after[name], err_msg=name)

    # ... and the intermediate JAX checkpoint is the same function
    obs_dim = rl_common.spec(env_id).obs_dim
    obs = np.asarray(np.random.default_rng(0).normal(size=(32, obs_dim)), dtype=np.float32)

    jax_agent = rl_common.make_agent(env_id, backend="jax")
    jax_agent.load(str(as_jax))

    import warp as wp

    device_obs = wp.array(obs, dtype=wp.float32, device=warp_agent.device)
    warp_logits = warp_agent.policy(device_obs).numpy()
    warp_values = warp_agent.value(device_obs).numpy()[:, 0]
    jax_logits, jax_values = (np.asarray(x) for x in jax_agent.net.apply(jax_agent.params, obs))

    assert np.abs(warp_logits - jax_logits).max() < 1e-5, "converted policy disagrees"
    assert np.abs(warp_values - jax_values).max() < 1e-5, "converted value function disagrees"
    print(f"[{env_id}] round trip exact; logits {np.abs(warp_logits - jax_logits).max():.2e}")


if __name__ == "__main__":
    for env in rl_common.env_ids():
        test_every_environment_has_a_complete_spec(env)
    test_bare_cli_uses_the_environment_recommendations()
    test_cli_overrides_win_over_recommendations()
    test_cli_boolean_flags()
    test_env_backend_defaults_per_backend()
    test_learning_rate_anneals_to_zero_over_the_run()
    print("all shared-core checks passed")
