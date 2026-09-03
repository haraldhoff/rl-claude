"""The JAX backend's trainer: device placement, annealing and reproducibility.

``tests/test_backend_parity.py`` already pins this backend's *maths* to the Warp
one -- the GAE scan, the networks and one-step dynamics all agree, and both
learn CartPole.  What is checked here is the machinery around it: that the
learning rate the shared loop computes actually reaches Adam (it is injected
into an ``optax`` state, which is easy to break silently), that ``--device``
means something, and that a run is reproducible from its seed.

Run with pytest, or directly:  python tests/test_jax_ppo.py
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import optax

import rl_common
from jax_rl.ppo import set_learning_rate
from jax_rl.vec_env import resolve_device


def _config(**overrides):
    """A CartPole run small enough to be a unit test."""
    settings = dict(num_envs=32, num_steps=16, total_timesteps=32 * 16 * 4, update_epochs=2, seed=0)
    settings.update(overrides)
    return rl_common.default_config("cartpole", backend="jax", **settings)


def _injected(opt_state):
    """The ``inject_hyperparams`` member of the optimizer chain."""
    return next(s for s in opt_state if getattr(s, "hyperparams", None) is not None)


# ---------------------------------------------------------------------------
# device selection
# ---------------------------------------------------------------------------


def test_resolve_device_accepts_the_warp_style_spellings():
    default = resolve_device(None)
    assert resolve_device("cpu").platform == "cpu"
    assert resolve_device("cpu:0").platform == "cpu"
    assert resolve_device(default) is default  # a device passes through unchanged


def test_resolve_device_refuses_rather_than_falling_back():
    """A device that does not exist must raise: silently training on the wrong
    one is the failure this replaced."""
    with pytest.raises(ValueError):
        resolve_device("cpu:99")
    with pytest.raises(RuntimeError):
        resolve_device("nonsense")


def test_trainer_honors_the_requested_device():
    """``--device`` must place the parameters, the optimizer and the environment."""
    device = resolve_device("cpu")
    trainer = rl_common.make_trainer(_config(), device="cpu")

    assert trainer.device == device
    assert trainer.vec_env.device == device
    for name, tree in [
        ("params", trainer.runner.params),
        ("opt_state", trainer.runner.opt_state),
        ("env_state", trainer.runner.env_state),
    ]:
        for leaf in jax.tree.leaves(tree):
            assert device in leaf.devices(), f"{name} leaf is not on {device}"


# ---------------------------------------------------------------------------
# annealing
# ---------------------------------------------------------------------------


def test_annealed_learning_rate_reaches_adam():
    """The shared loop owns the schedule; Adam must actually see each value."""
    cfg = _config()
    trainer = rl_common.make_trainer(cfg)
    seen = []

    def record(stats):
        seen.append((stats["lr"], float(_injected(trainer.runner.opt_state).hyperparams["learning_rate"])))

    trainer.train(callback=record)

    assert len(seen) == cfg.num_iterations
    for expected, actual in seen:
        assert actual == pytest.approx(expected, rel=1e-6), f"loop asked for {expected}, Adam holds {actual}"
    assert seen[0][0] > seen[-1][0], "the schedule did not anneal"
    print(f"annealing reaches Adam: {seen[0][0]:.2e} -> {seen[-1][0]:.2e} over {len(seen)} iterations")


def test_set_learning_rate_rejects_a_chain_it_cannot_drive():
    """Guard against the silent version of this bug: if the optimizer chain is
    changed so nothing carries an injected learning rate, fail loudly."""
    drivable = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.inject_hyperparams(optax.adam)(learning_rate=1e-3),
    )
    state = drivable.init({"w": jnp.zeros(2)})
    assert float(_injected(set_learning_rate(state, 5e-4)).hyperparams["learning_rate"]) == pytest.approx(5e-4)

    plain = optax.chain(optax.clip_by_global_norm(0.5), optax.sgd(1e-3))
    with pytest.raises(ValueError, match="learning_rate"):
        set_learning_rate(plain.init({"w": jnp.zeros(2)}), 5e-4)


# ---------------------------------------------------------------------------
# reproducibility
# ---------------------------------------------------------------------------


def test_the_same_seed_gives_the_same_run():
    """Unlike the Warp backend (whose loss is accumulated with atomics), the JAX
    backend is deterministic run to run -- the README leans on this."""
    first = rl_common.make_trainer(_config(seed=3)).train(log_every=0)
    second = rl_common.make_trainer(_config(seed=3)).train(log_every=0)

    assert len(first) == len(second)
    for a, b in zip(first, second):
        for key in ("entropy", "approx_kl", "clipfrac", "value_loss", "episodes"):
            assert a[key] == pytest.approx(b[key], rel=1e-6, nan_ok=True), (
                f"{key} differs at iteration {a['iteration']}"
            )

    changed = rl_common.make_trainer(_config(seed=4)).train(log_every=0)
    assert changed[-1]["value_loss"] != pytest.approx(first[-1]["value_loss"], rel=1e-9), "the seed made no difference"
    print(f"identical seeds reproduce ({len(first)} iterations); a different seed diverges")


if __name__ == "__main__":
    test_resolve_device_accepts_the_warp_style_spellings()
    test_resolve_device_refuses_rather_than_falling_back()
    test_trainer_honors_the_requested_device()
    test_annealed_learning_rate_reaches_adam()
    test_set_learning_rate_rejects_a_chain_it_cannot_drive()
    test_the_same_seed_gives_the_same_run()
    print("all jax ppo checks passed")
