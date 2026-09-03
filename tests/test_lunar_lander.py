"""Checks for the Warp LunarLander.

Box2D parity is impossible by construction (see the module docstring of
``rl_common/specs/lunar_lander.py``), so instead these tests pin down everything
the agent actually sees -- the observation vector, the reward shaping, the
terrain layout and the termination rules -- against Gymnasium's definitions,
and validate the dynamics by flying Gymnasium's own heuristic controller.

Run with pytest, or directly:  python tests/test_lunar_lander.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rl_common
from rl_common import to_numpy
import support
from rl_common.specs import lunar_lander as ll

BACKENDS = ["warp", "jax"]


def heuristic(s: np.ndarray) -> np.ndarray:
    """Gymnasium's ``demo_heuristic_lander`` policy, vectorized over envs."""
    angle_targ = np.clip(s[:, 0] * 0.5 + s[:, 2] * 1.0, -0.4, 0.4)
    hover_targ = 0.55 * np.abs(s[:, 0])
    angle_todo = (angle_targ - s[:, 4]) * 0.5 - s[:, 5] * 1.0
    hover_todo = (hover_targ - s[:, 1]) * 0.5 - s[:, 3] * 0.5
    grounded = (s[:, 6] + s[:, 7]) > 0
    angle_todo = np.where(grounded, 0.0, angle_todo)
    hover_todo = np.where(grounded, -s[:, 3] * 0.5, hover_todo)
    a = np.zeros(len(s), dtype=np.int32)
    a = np.where(angle_todo > 0.05, 1, a)
    a = np.where(angle_todo < -0.05, 3, a)
    a = np.where((hover_todo > np.abs(angle_todo)) & (hover_todo > 0.05), 2, a)
    return a


def _shaping(s: np.ndarray) -> np.ndarray:
    """Gymnasium's reward-shaping potential, straight from the observation."""
    return (
        -100.0 * np.sqrt(s[:, 0] ** 2 + s[:, 1] ** 2)
        - 100.0 * np.sqrt(s[:, 2] ** 2 + s[:, 3] ** 2)
        - 100.0 * np.abs(s[:, 4])
        + 10.0 * s[:, 6]
        + 10.0 * s[:, 7]
    )


def test_body_properties_match_the_box2d_model():
    """Mass/inertia come from the same polygons and densities Box2D is given."""
    hull_mass, _, _ = ll.polygon_properties([(x / ll.SCALE, y / ll.SCALE) for x, y in ll.LANDER_POLY], 5.0)
    assert abs(hull_mass - 4.8167) < 1e-3, hull_mass  # density 5 x 0.9633 m^2
    assert abs(ll.MASS - (hull_mass + 2 * 0.0711)) < 2e-3, ll.MASS  # + two 1.0-density legs
    assert abs(ll.COM[0]) < 1e-6 and 0.0 < ll.COM[1] < 0.2
    assert 0.5 < ll.INERTIA < 1.5


@pytest.mark.parametrize("backend", BACKENDS)
def test_terrain_matches_gymnasium_layout(backend):
    env = rl_common.make("lunarlander", 32, backend=backend, seed=3)
    env.reset()
    terrain = env.render_state()["terrain"]

    # chunks 4..6 are the helipad: flat, and at Gymnasium's 0.33-smoothed height
    pad = terrain[:, ll.CHUNKS // 2 - 1 : ll.CHUNKS // 2 + 2]
    assert np.allclose(pad, 0.99 * ll.HELIPAD_Y, atol=1e-5), pad[0]
    # the rest is random but inside the world
    assert terrain.min() >= 0.0 and terrain.max() <= ll.WORLD_H * 0.5 + 1e-4
    assert terrain[:, 0].std() > 0.1, "terrain outside the pad should be randomized"
    print(f"[{backend}] terrain: helipad flat at {pad[0, 0]:.4f} (= 0.99 * H/4), off-pad std {terrain[:, 0].std():.2f}")


@pytest.mark.parametrize("backend", BACKENDS)
def test_observation_matches_gymnasium_formula(backend):
    env = rl_common.make("lunarlander", 64, backend=backend, seed=1)
    obs, _ = env.reset()
    rng = np.random.default_rng(0)
    for _ in range(50):
        obs, _, _, _, _ = env.step(rng.integers(0, 4, size=64).astype(np.int32))

    state = env.render_state()
    body, contacts = state["body"], state["contacts"]
    com, vel, angle = body[:, 0:2], body[:, 2:4], body[:, 4]
    omega = body[:, 5]

    c, s = np.cos(angle), np.sin(angle)
    offset = np.stack([c * ll.COM[0] - s * ll.COM[1], s * ll.COM[0] + c * ll.COM[1]], axis=1)
    origin = com - offset  # Gymnasium reports the body origin, not the centre of mass

    half_w, half_h = ll.WORLD_W / 2, ll.WORLD_H / 2
    expected = np.stack(
        [
            (origin[:, 0] - half_w) / half_w,
            (origin[:, 1] - (ll.HELIPAD_Y + ll.LEG_DOWN / ll.SCALE)) / half_h,
            vel[:, 0] * half_w / ll.FPS,
            vel[:, 1] * half_h / ll.FPS,
            angle,
            20.0 * omega / ll.FPS,
            contacts[:, 0],
            contacts[:, 1],
        ],
        axis=1,
    )
    diff = np.abs(expected - to_numpy(obs)).max()
    assert diff < 1e-5, f"observation differs from the Gymnasium formula by {diff}"
    print(f"[{backend}] observation matches the Gymnasium formula (max diff {diff:.2e})")


@pytest.mark.parametrize("backend", BACKENDS)
def test_reward_is_shaping_delta_plus_fuel(backend):
    env = rl_common.make("lunarlander", 64, backend=backend, autoreset=False, seed=2)
    obs, _ = env.reset()
    rng = np.random.default_rng(1)
    prev = _shaping(to_numpy(obs))
    alive = np.ones(64, dtype=bool)
    checked = 0

    for _ in range(120):
        actions = rng.integers(0, 4, size=64).astype(np.int32)
        obs, reward, terminated, truncated, _ = env.step(actions)
        s = to_numpy(obs)
        fuel = np.where(actions == 2, 0.30, 0.0) + np.where((actions == 1) | (actions == 3), 0.03, 0.0)
        expected = _shaping(s) - prev - fuel
        done = (to_numpy(terminated) + to_numpy(truncated)) > 0
        ok = alive & ~done  # terminal steps pay the flat +/-100 instead
        if ok.any():
            diff = np.abs(expected[ok] - to_numpy(reward)[ok]).max()
            assert diff < 2e-3, f"reward differs from shaping delta by {diff}"
            checked += int(ok.sum())
        prev = _shaping(s)
        alive &= ~done
    assert checked > 1000
    print(f"[{backend}] reward = shaping delta - fuel on {checked} transitions")


@pytest.mark.parametrize("backend", BACKENDS)
def test_gravity_and_main_engine_thrust(backend):
    env = rl_common.make("lunarlander", 4, backend=backend, seed=0)
    env.reset()
    flat = np.tile(np.array([10.0, 10.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32), (4, 1))
    terrain = env.render_state()["terrain"]
    env.set_state(flat, terrain)

    env.step(np.zeros(4, dtype=np.int32))
    vy = env.render_state()["body"][:, 3]
    assert np.allclose(vy, ll.GRAVITY / ll.FPS, atol=1e-4), vy  # -0.2 m/s after one 20 ms step

    env.set_state(flat, terrain)
    env.step(np.full(4, 2, dtype=np.int32))
    dv = env.render_state()["body"][:, 3] - ll.GRAVITY / ll.FPS
    # impulse magnitude is MAIN_ENGINE_POWER * (4/SCALE +/- dispersion), upward
    nominal = ll.MAIN_ENGINE_POWER * (4.0 / ll.SCALE) / ll.MASS
    assert np.all(dv > 0.5 * nominal) and np.all(dv < 1.6 * nominal), (dv, nominal)
    print(f"[{backend}] free fall {vy[0]:+.3f} m/s per step; main engine adds {dv.mean():+.3f} m/s "
          f"(nominal {nominal:.3f})")


@pytest.mark.parametrize("backend", BACKENDS)
def test_crash_and_out_of_bounds_terminate(backend):
    # drop the lander onto the ground with no control: the hull hits -> crash
    n = 16
    env = rl_common.make("lunarlander", n, backend=backend, autoreset=False, seed=11)
    env.reset()
    alive = np.ones(n, dtype=bool)
    terminal = np.full(n, np.nan)
    for _ in range(300):
        _, reward, terminated, truncated, _ = env.step(np.zeros(n, dtype=np.int32))
        done = (to_numpy(terminated) + to_numpy(truncated)) > 0
        terminal = np.where(alive & done, to_numpy(reward), terminal)
        alive &= ~done
        if not alive.any():
            break
    assert not alive.any(), "an unpowered lander must hit the ground"
    assert np.allclose(terminal, -100.0), terminal
    print(f"[{backend}] unpowered descent ends at -100 for all {n} envs")

    # a lander pushed sideways past |s0| >= 1 terminates with -100 as well
    env = rl_common.make("lunarlander", 4, backend=backend, autoreset=False, seed=0)
    env.reset()
    rendered = env.render_state()
    body = rendered["body"].copy()
    body[:, 0] = ll.WORLD_W + 0.3  # past the right edge -> |s0| >= 1
    body[:, 1] = 8.0
    body[:, 2] = 5.0
    env.set_state(body, rendered["terrain"])
    _, reward, terminated, _, _ = env.step(np.zeros(4, dtype=np.int32))
    assert np.all(to_numpy(terminated) == 1.0) and np.allclose(to_numpy(reward), -100.0)
    print(f"[{backend}] leaving the viewport terminates with -100")


@pytest.mark.parametrize("backend", BACKENDS)
def test_heuristic_controller_lands(backend):
    """Gymnasium's reference controller must fly our dynamics too."""
    n = 128
    env = rl_common.make("lunarlander", n, backend=backend, autoreset=False, seed=7)
    obs, _ = env.reset()
    alive = np.ones(n, dtype=bool)
    returns = np.zeros(n)
    last = np.zeros(n)

    for _ in range(1000):
        obs, reward, terminated, truncated, _ = env.step(heuristic(to_numpy(obs)))
        done = (to_numpy(terminated) + to_numpy(truncated)) > 0
        returns += alive * to_numpy(reward)
        last = np.where(alive & done, to_numpy(reward), last)
        alive &= ~done
        if not alive.any():
            break

    landed = int((last > 50).sum())
    assert returns.mean() > 150.0, f"heuristic only scored {returns.mean():.1f}"
    assert landed > 0.8 * n, f"heuristic only landed {landed}/{n} times"
    print(
        f"[{backend}] heuristic controller: mean return {returns.mean():.1f}, "
        f"landed {landed}/{n} (+100 on touchdown)"
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_training_improves_return(backend):
    """Both backends: 40 iterations is ~19 s on either now that JAX has a GPU
    build.  This was Warp-only while jaxlib was CPU-only and the same run took
    minutes."""
    _, first, last = support.assert_learns(
        "lunarlander", backend, iterations=40, gain=50.0, window=3, num_envs=256, num_steps=64
    )
    print(f"[{backend}] short PPO run improves return {first:.1f} -> {last:.1f}")


if __name__ == "__main__":
    test_body_properties_match_the_box2d_model()
    for backend in BACKENDS:
        test_terrain_matches_gymnasium_layout(backend)
        test_observation_matches_gymnasium_formula(backend)
        test_reward_is_shaping_delta_plus_fuel(backend)
        test_gravity_and_main_engine_thrust(backend)
        test_crash_and_out_of_bounds_terminate(backend)
        test_heuristic_controller_lands(backend)
        test_training_improves_return(backend)
    print("all lunar lander checks passed")
