"""Checks for the Warp LunarLander.

Box2D parity is impossible by construction (see the module docstring of
``warp_rl/envs/lunar_lander.py``), so instead these tests pin down everything
the agent actually sees -- the observation vector, the reward shaping, the
terrain layout and the termination rules -- against Gymnasium's definitions,
and validate the dynamics by flying Gymnasium's own heuristic controller.

Run with pytest, or directly:  python tests/test_lunar_lander.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import warp as wp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warp_rl  # noqa: E402
from warp_rl.envs import lunar_lander as ll  # noqa: E402


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
    hull_mass, _, _ = ll._polygon_properties([(x / ll.SCALE, y / ll.SCALE) for x, y in ll.LANDER_POLY], 5.0)
    assert abs(hull_mass - 4.8167) < 1e-3, hull_mass  # density 5 x 0.9633 m^2
    assert abs(ll.MASS - (hull_mass + 2 * 0.0711)) < 2e-3, ll.MASS  # + two 1.0-density legs
    assert abs(ll.COM[0]) < 1e-6 and 0.0 < ll.COM[1] < 0.2
    assert 0.5 < ll.INERTIA < 1.5


def test_terrain_matches_gymnasium_layout():
    env = warp_rl.make("lunarlander", 32, seed=3)
    env.reset()
    terrain = env.terrain.numpy()

    # chunks 4..6 are the helipad: flat, and at Gymnasium's 0.33-smoothed height
    pad = terrain[:, ll.CHUNKS // 2 - 1 : ll.CHUNKS // 2 + 2]
    assert np.allclose(pad, 0.99 * ll.HELIPAD_Y, atol=1e-5), pad[0]
    # the rest is random but inside the world
    assert terrain.min() >= 0.0 and terrain.max() <= ll.WORLD_H * 0.5 + 1e-4
    assert terrain[:, 0].std() > 0.1, "terrain outside the pad should be randomized"
    print(f"terrain: helipad flat at {pad[0, 0]:.4f} (= 0.99 * H/4), off-pad std {terrain[:, 0].std():.2f}")


def test_observation_matches_gymnasium_formula():
    env = warp_rl.make("lunarlander", 64, seed=1)
    obs, _ = env.reset()
    rng = np.random.default_rng(0)
    for _ in range(50):
        obs, _, _, _, _ = env.step(rng.integers(0, 4, size=64).astype(np.int32))

    body = env.body.numpy()
    contacts = env.contacts.numpy()
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
    diff = np.abs(expected - obs.numpy()).max()
    assert diff < 1e-5, f"observation differs from the Gymnasium formula by {diff}"
    print(f"observation matches the Gymnasium formula (max diff {diff:.2e})")


def test_reward_is_shaping_delta_plus_fuel():
    env = warp_rl.make("lunarlander", 64, autoreset=False, seed=2)
    obs, _ = env.reset()
    rng = np.random.default_rng(1)
    prev = _shaping(obs.numpy())
    alive = np.ones(64, dtype=bool)
    checked = 0

    for _ in range(120):
        actions = rng.integers(0, 4, size=64).astype(np.int32)
        obs, reward, terminated, truncated, _ = env.step(actions)
        s = obs.numpy()
        fuel = np.where(actions == 2, 0.30, 0.0) + np.where((actions == 1) | (actions == 3), 0.03, 0.0)
        expected = _shaping(s) - prev - fuel
        done = (terminated.numpy() + truncated.numpy()) > 0
        ok = alive & ~done  # terminal steps pay the flat +/-100 instead
        if ok.any():
            diff = np.abs(expected[ok] - reward.numpy()[ok]).max()
            assert diff < 2e-3, f"reward differs from shaping delta by {diff}"
            checked += int(ok.sum())
        prev = _shaping(s)
        alive &= ~done
    assert checked > 1000
    print(f"reward = shaping delta - fuel on {checked} transitions")


def test_gravity_and_main_engine_thrust():
    env = warp_rl.make("lunarlander", 4, seed=0)
    env.reset()
    env.body.assign(np.tile(np.array([10.0, 10.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32), (4, 1)))

    env.step(np.zeros(4, dtype=np.int32))
    vy = env.body.numpy()[:, 3]
    assert np.allclose(vy, ll.GRAVITY / ll.FPS, atol=1e-4), vy  # -0.2 m/s after one 20 ms step

    env.body.assign(np.tile(np.array([10.0, 10.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32), (4, 1)))
    env.step(np.full(4, 2, dtype=np.int32))
    dv = env.body.numpy()[:, 3] - ll.GRAVITY / ll.FPS
    # impulse magnitude is MAIN_ENGINE_POWER * (4/SCALE +/- dispersion), upward
    nominal = ll.MAIN_ENGINE_POWER * (4.0 / ll.SCALE) / ll.MASS
    assert np.all(dv > 0.5 * nominal) and np.all(dv < 1.6 * nominal), (dv, nominal)
    print(f"free fall {vy[0]:+.3f} m/s per step; main engine adds {dv.mean():+.3f} m/s (nominal {nominal:.3f})")


def test_crash_and_out_of_bounds_terminate():
    # drop the lander onto the ground with no control: the hull hits -> crash
    n = 16
    env = warp_rl.make("lunarlander", n, autoreset=False, seed=11)
    env.reset()
    alive = np.ones(n, dtype=bool)
    terminal = np.full(n, np.nan)
    for _ in range(300):
        _, reward, terminated, truncated, _ = env.step(np.zeros(n, dtype=np.int32))
        done = (terminated.numpy() + truncated.numpy()) > 0
        terminal = np.where(alive & done, reward.numpy(), terminal)
        alive &= ~done
        if not alive.any():
            break
    assert not alive.any(), "an unpowered lander must hit the ground"
    assert np.allclose(terminal, -100.0), terminal
    print(f"unpowered descent ends at -100 for all {n} envs")

    # a lander pushed sideways past |s0| >= 1 terminates with -100 as well
    env = warp_rl.make("lunarlander", 4, autoreset=False, seed=0)
    env.reset()
    state = env.body.numpy()
    state[:, 0] = ll.WORLD_W + 0.3  # past the right edge -> |s0| >= 1
    state[:, 1] = 8.0
    state[:, 2] = 5.0
    env.body.assign(state)
    _, reward, terminated, _, _ = env.step(np.zeros(4, dtype=np.int32))
    assert np.all(terminated.numpy() == 1.0) and np.allclose(reward.numpy(), -100.0)
    print("leaving the viewport terminates with -100")


def test_heuristic_controller_lands():
    """Gymnasium's reference controller must fly our dynamics too."""
    n = 128
    env = warp_rl.make("lunarlander", n, autoreset=False, seed=7)
    obs, _ = env.reset()
    alive = np.ones(n, dtype=bool)
    returns = np.zeros(n)
    last = np.zeros(n)

    for _ in range(1000):
        obs, reward, terminated, truncated, _ = env.step(heuristic(obs.numpy()))
        done = (terminated.numpy() + truncated.numpy()) > 0
        returns += alive * reward.numpy()
        last = np.where(alive & done, reward.numpy(), last)
        alive &= ~done
        if not alive.any():
            break

    landed = int((last > 50).sum())
    assert returns.mean() > 150.0, f"heuristic only scored {returns.mean():.1f}"
    assert landed > 0.8 * n, f"heuristic only landed {landed}/{n} times"
    print(f"heuristic controller: mean return {returns.mean():.1f}, landed {landed}/{n} (+100 on touchdown)")


def test_training_improves_return():
    cfg = warp_rl.default_config("lunarlander", num_envs=256, num_steps=64, total_timesteps=256 * 64 * 40, seed=0)
    trainer = warp_rl.PPO(cfg)
    history = []
    trainer.train(callback=lambda s: history.append(s["episodic_return"]))
    first, last = np.mean(history[:3]), np.mean(history[-3:])
    assert last > first + 50.0, f"return did not improve: {first:.1f} -> {last:.1f}"
    print(f"short PPO run improves return {first:.1f} -> {last:.1f}")


if __name__ == "__main__":
    wp.init()
    test_body_properties_match_the_box2d_model()
    test_terrain_matches_gymnasium_layout()
    test_observation_matches_gymnasium_formula()
    test_reward_is_shaping_delta_plus_fuel()
    test_gravity_and_main_engine_thrust()
    test_crash_and_out_of_bounds_terminate()
    test_heuristic_controller_lands()
    test_training_improves_return()
    print("all lunar lander checks passed")
