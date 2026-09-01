"""Check the CartPole implementations against Gymnasium's CartPole-v1.

Both backends are pure ports, so parity is exact up to float32 rounding.

Run with pytest, or directly:  python tests/test_cartpole_env.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym  # noqa: E402

import rl_common  # noqa: E402

BACKENDS = ["warp", "jax"]
NUM_ENVS = 16
STEPS = 200


def _gym_envs(n: int, seed: int = 0):
    envs, states = [], []
    for i in range(n):
        env = gym.make("CartPole-v1").unwrapped
        obs, _ = env.reset(seed=seed + i)
        envs.append(env)
        states.append(obs)
    return envs, np.asarray(states, dtype=np.float32)


@pytest.mark.parametrize("backend", BACKENDS)
def test_single_step_dynamics_match(backend):
    """Re-sync the state each step so only one-step dynamics are compared."""
    rng = np.random.default_rng(0)
    envs, states = _gym_envs(NUM_ENVS)
    env = rl_common.make("cartpole", NUM_ENVS, backend=backend, autoreset=False)
    env.reset(seed=0)

    max_diff = 0.0
    for _ in range(STEPS):
        env.set_state(states)
        actions = rng.integers(0, 2, size=NUM_ENVS).astype(np.int32)

        _, reward, terminated, _, info = env.step(actions)
        next_obs = rl_common.to_numpy(info["final_observation"])
        terminated = rl_common.to_numpy(terminated)
        reward = rl_common.to_numpy(reward)

        next_states, terms = [], []
        for i, gym_env in enumerate(envs):
            obs, gym_reward, gym_terminated, _, _ = gym_env.step(int(actions[i]))
            assert gym_reward == reward[i]
            next_states.append(obs)
            terms.append(float(gym_terminated))
            if gym_terminated:
                obs, _ = gym_env.reset()
                next_states[-1] = obs
        next_states = np.asarray(next_states, dtype=np.float32)

        live = np.asarray(terms) == 0.0
        assert np.array_equal(np.asarray(terms), terminated), "termination flags differ"
        max_diff = max(max_diff, float(np.abs(next_states[live] - next_obs[live]).max(initial=0.0)))
        states = next_states

    assert max_diff < 1e-5, f"one-step dynamics differ by {max_diff}"
    print(f"[{backend}] one-step dynamics: max |backend - gymnasium| = {max_diff:.3e} over {STEPS} steps")


@pytest.mark.parametrize("backend", BACKENDS)
def test_full_episode_matches(backend):
    """Free-running episodes: same actions, same length, same trajectory."""
    rng = np.random.default_rng(1)
    envs, states = _gym_envs(NUM_ENVS, seed=100)
    env = rl_common.make("cartpole", NUM_ENVS, backend=backend, autoreset=False)
    env.reset(seed=0)
    env.set_state(states)

    alive = np.ones(NUM_ENVS, dtype=bool)
    gym_len = np.zeros(NUM_ENVS, dtype=np.int64)
    backend_len = np.zeros(NUM_ENVS, dtype=np.int64)
    max_diff = 0.0

    for _ in range(500):
        actions = rng.integers(0, 2, size=NUM_ENVS).astype(np.int32)
        _, _, terminated, _, info = env.step(actions)
        next_obs = rl_common.to_numpy(info["final_observation"])
        done = rl_common.to_numpy(terminated) > 0

        gym_next, gym_done = [], []
        for i, gym_env in enumerate(envs):
            if not alive[i]:  # Gymnasium warns if you step a finished episode
                gym_next.append(np.zeros(4, dtype=np.float32))
                gym_done.append(True)
                continue
            obs, _, terminated_i, _, _ = gym_env.step(int(actions[i]))
            gym_next.append(obs)
            gym_done.append(terminated_i)
        gym_next = np.asarray(gym_next, dtype=np.float32)
        gym_done = np.asarray(gym_done)

        max_diff = max(max_diff, float(np.abs(gym_next[alive] - next_obs[alive]).max(initial=0.0)))
        gym_len += alive & ~gym_done
        backend_len += alive & ~done
        alive &= ~(gym_done | done)
        if not alive.any():
            break

    assert np.array_equal(gym_len, backend_len), f"episode lengths differ: {gym_len} vs {backend_len}"
    assert max_diff < 1e-3, f"trajectories drifted by {max_diff}"
    print(f"[{backend}] free-running episodes: lengths identical, max drift {max_diff:.3e}")


@pytest.mark.parametrize("backend", BACKENDS)
def test_truncation_and_autoreset(backend):
    env = rl_common.make("cartpole", 4, backend=backend, max_episode_steps=10, autoreset=True)
    env.reset(seed=3)
    for t in range(10):  # alternating pushes keep the pole up for 10 steps
        _, _, terminated, truncated, _ = env.step(np.full(4, t % 2, dtype=np.int32))
    assert np.all(rl_common.to_numpy(truncated) == 1.0), "expected truncation at max_episode_steps"
    assert np.all(rl_common.to_numpy(terminated) == 0.0)
    assert np.all(env.render_state()["steps"] == 0), "auto-reset must clear the step counter"
    print(f"[{backend}] truncation and auto-reset behave as expected")


@pytest.mark.parametrize("backend", BACKENDS)
def test_autoreset_starts_new_episode(backend):
    env = rl_common.make("cartpole", 64, backend=backend, autoreset=True)
    env.reset(seed=7)
    rng = np.random.default_rng(0)
    finished = 0
    for _ in range(400):
        obs, _, terminated, truncated, _ = env.step(rng.integers(0, 2, size=64).astype(np.int32))
        done = (rl_common.to_numpy(terminated) + rl_common.to_numpy(truncated)) > 0
        if done.any():
            finished += int(done.sum())
            fresh = rl_common.to_numpy(obs)[done]
            assert np.abs(fresh).max() <= 0.05, "auto-reset must draw a fresh U(-0.05, 0.05) state"
    mean_return, mean_length, count = env.pop_episode_stats()
    assert count == finished
    assert 5.0 < mean_length < 100.0, f"random policy episode length looks wrong: {mean_length}"
    print(f"[{backend}] auto-reset: {count} episodes, mean length {mean_length:.1f} (random policy)")


if __name__ == "__main__":
    for backend in BACKENDS:
        test_single_step_dynamics_match(backend)
        test_full_episode_matches(backend)
        test_truncation_and_autoreset(backend)
        test_autoreset_starts_new_episode(backend)
    print("all cartpole checks passed")
