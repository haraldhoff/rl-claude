"""Check the Warp CartPole against Gymnasium's CartPole-v1.

Run with pytest, or directly:  python tests/test_env_parity.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import warp as wp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym  # noqa: E402

from warp_rl.envs.cartpole import CartPoleVectorEnv  # noqa: E402

NUM_ENVS = 16
STEPS = 200


def _gym_envs(n: int, seed: int = 0, max_episode_steps: int = 500):
    envs, states = [], []
    for i in range(n):
        env = gym.make("CartPole-v1", max_episode_steps=max_episode_steps).unwrapped
        obs, _ = env.reset(seed=seed + i)
        envs.append(env)
        states.append(obs)
    return envs, np.asarray(states, dtype=np.float32)


def test_single_step_dynamics_match():
    """Re-sync the Warp state each step so only one-step dynamics are compared."""
    rng = np.random.default_rng(0)
    envs, states = _gym_envs(NUM_ENVS)
    warp_env = CartPoleVectorEnv(NUM_ENVS, autoreset=False)
    warp_env.reset(seed=0)

    max_diff = 0.0
    for _ in range(STEPS):
        warp_env.set_state(states)
        actions = rng.integers(0, 2, size=NUM_ENVS).astype(np.int32)

        _, w_rew, w_term, _, w_info = warp_env.step(actions)
        warp_next = w_info["final_observation"].numpy()
        warp_term = w_term.numpy()
        warp_rew = w_rew.numpy()

        next_states, terms = [], []
        for i, env in enumerate(envs):
            obs, reward, terminated, _, _ = env.step(int(actions[i]))
            assert reward == warp_rew[i]
            next_states.append(obs)
            terms.append(float(terminated))
            if terminated:
                obs, _ = env.reset()
                next_states[-1] = obs
        next_states = np.asarray(next_states, dtype=np.float32)

        # compare before any reset happened on either side
        live = np.asarray(terms) == 0.0
        assert np.array_equal(np.asarray(terms), warp_term), "termination flags differ"
        diff = np.abs(next_states[live] - warp_next[live]).max(initial=0.0)
        max_diff = max(max_diff, float(diff))
        states = next_states

    assert max_diff < 1e-5, f"one-step dynamics differ by {max_diff}"
    print(f"one-step dynamics: max |warp - gymnasium| = {max_diff:.3e} over {STEPS} steps")


def test_full_episode_matches():
    """Free-running episodes: same actions, same length, same trajectory."""
    rng = np.random.default_rng(1)
    envs, states = _gym_envs(NUM_ENVS, seed=100)
    warp_env = CartPoleVectorEnv(NUM_ENVS, autoreset=False)
    warp_env.reset(seed=0)
    warp_env.set_state(states)

    alive = np.ones(NUM_ENVS, dtype=bool)
    gym_len = np.zeros(NUM_ENVS, dtype=np.int64)
    warp_len = np.zeros(NUM_ENVS, dtype=np.int64)
    max_diff = 0.0

    for _ in range(500):
        actions = rng.integers(0, 2, size=NUM_ENVS).astype(np.int32)
        _, _, w_term, _, w_info = warp_env.step(actions)
        warp_next = w_info["final_observation"].numpy()
        w_done = w_term.numpy() > 0

        g_next, g_done = [], []
        for i, env in enumerate(envs):
            if not alive[i]:  # Gymnasium warns if you step a finished episode
                g_next.append(np.zeros(4, dtype=np.float32))
                g_done.append(True)
                continue
            obs, _, terminated, _, _ = env.step(int(actions[i]))
            g_next.append(obs)
            g_done.append(terminated)
        g_next = np.asarray(g_next, dtype=np.float32)
        g_done = np.asarray(g_done)

        max_diff = max(max_diff, float(np.abs(g_next[alive] - warp_next[alive]).max(initial=0.0)))
        gym_len += alive & ~g_done
        warp_len += alive & ~w_done
        alive &= ~(g_done | w_done)
        if not alive.any():
            break

    assert np.array_equal(gym_len, warp_len), f"episode lengths differ: {gym_len} vs {warp_len}"
    assert max_diff < 1e-3, f"trajectories drifted by {max_diff}"
    print(f"free-running episodes: lengths identical, max drift {max_diff:.3e}, mean length {gym_len.mean():.1f}")


def test_truncation_at_max_episode_steps():
    env = CartPoleVectorEnv(4, max_episode_steps=10, autoreset=True)
    env.reset(seed=3)
    actions = np.zeros(4, dtype=np.int32)  # push left forever -> pole falls first
    truncs = []
    for _ in range(10):
        _, _, term, trunc, _ = env.step(actions)
        truncs.append((term.numpy().copy(), trunc.numpy().copy()))
    # with alternating actions the pole survives 10 steps and must truncate
    env2 = CartPoleVectorEnv(4, max_episode_steps=10, autoreset=True)
    env2.reset(seed=3)
    for t in range(10):
        acts = np.full(4, t % 2, dtype=np.int32)
        _, _, term, trunc, _ = env2.step(acts)
    assert np.all(trunc.numpy() == 1.0), "expected truncation at max_episode_steps"
    assert np.all(term.numpy() == 0.0)
    assert np.all(env2.steps.numpy() == 0), "auto-reset must clear the step counter"
    print("truncation and auto-reset behave as expected")


def test_autoreset_starts_new_episode():
    env = CartPoleVectorEnv(64, max_episode_steps=500, autoreset=True)
    env.reset(seed=7)
    rng = np.random.default_rng(0)
    finished = 0
    for _ in range(400):
        acts = rng.integers(0, 2, size=64).astype(np.int32)
        obs, _, term, trunc, info = env.step(acts)
        done = (term.numpy() + trunc.numpy()) > 0
        if done.any():
            finished += int(done.sum())
            fresh = obs.numpy()[done]
            assert np.abs(fresh).max() <= 0.05, "auto-reset must draw a fresh U(-0.05, 0.05) state"
    mean_ret, mean_len, count = env.pop_episode_stats()
    assert count == finished
    assert 5.0 < mean_len < 100.0, f"random policy episode length looks wrong: {mean_len}"
    print(f"auto-reset: {count} episodes, mean length {mean_len:.1f} (random policy)")


if __name__ == "__main__":
    wp.init()
    test_single_step_dynamics_match()
    test_full_episode_matches()
    test_truncation_at_max_episode_steps()
    test_autoreset_starts_new_episode()
    print("all environment parity checks passed")
