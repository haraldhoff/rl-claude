"""Check the MountainCar implementations against Gymnasium's MountainCar-v0.

The physics is a pure port, so parity here is exact (float32 rounding aside),
exactly as for CartPole, and both backends are checked.  The extra tests cover
``action_repeat`` -- our one addition -- and the exploration wall that motivates it.

Run with pytest, or directly:  python tests/test_mountain_car.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym

import rl_common
from rl_common import to_numpy
from rl_common.specs import mountain_car as mc

BACKENDS = ["warp", "jax"]
NUM_ENVS = 16
STEPS = 200


def _gym_envs(n: int, seed: int = 0):
    envs, states = [], []
    for i in range(n):
        env = gym.make("MountainCar-v0").unwrapped
        obs, _ = env.reset(seed=seed + i)
        envs.append(env)
        states.append(obs)
    return envs, np.asarray(states, dtype=np.float32)


@pytest.mark.parametrize("backend", BACKENDS)
def test_single_step_dynamics_match(backend):
    """Re-sync the state each step so only one-step dynamics are compared."""
    rng = np.random.default_rng(0)
    envs, states = _gym_envs(NUM_ENVS)
    env = rl_common.make(
        "mountaincar", NUM_ENVS, backend=backend, max_episode_steps=STEPS,
        action_repeat=1, autoreset=False,  # stock MountainCar-v0, one step per action
    )
    env.reset(seed=0)

    max_diff = 0.0
    for _ in range(STEPS):
        env.set_state(states)
        actions = rng.integers(0, 3, size=NUM_ENVS).astype(np.int32)
        _, reward_arr, term_arr, _, info = env.step(actions)
        next_obs = to_numpy(info["final_observation"])

        next_states, terms = [], []
        for i, gym_env in enumerate(envs):
            obs, reward, terminated, _, _ = gym_env.step(int(actions[i]))
            assert reward == to_numpy(reward_arr)[i]
            next_states.append(obs)
            terms.append(float(terminated))
            if terminated:
                obs, _ = gym_env.reset()
                next_states[-1] = obs
        next_states = np.asarray(next_states, dtype=np.float32)

        live = np.asarray(terms) == 0.0
        assert np.array_equal(np.asarray(terms), to_numpy(term_arr)), "termination flags differ"
        max_diff = max(max_diff, float(np.abs(next_states[live] - next_obs[live]).max(initial=0.0)))
        states = next_states

    assert max_diff < 1e-6, f"one-step dynamics differ by {max_diff}"
    print(f"[{backend}] one-step dynamics: max |backend - gymnasium| = {max_diff:.3e} over {STEPS} steps")


@pytest.mark.parametrize("backend", BACKENDS)
def test_full_episode_matches(backend):
    """Free-running episodes: same actions, same trajectory, same 200-step limit."""
    rng = np.random.default_rng(1)
    envs, states = _gym_envs(NUM_ENVS, seed=100)
    env = rl_common.make(
        "mountaincar", NUM_ENVS, backend=backend, max_episode_steps=STEPS,
        action_repeat=1, autoreset=False,  # stock MountainCar-v0, one step per action
    )
    env.reset(seed=0)
    env.set_state(states)

    max_diff = 0.0
    for _ in range(STEPS):
        actions = rng.integers(0, 3, size=NUM_ENVS).astype(np.int32)
        _, _, _, trunc_arr, info = env.step(actions)
        next_obs = to_numpy(info["final_observation"])
        gym_next = np.asarray([e.step(int(actions[i]))[0] for i, e in enumerate(envs)], dtype=np.float32)
        max_diff = max(max_diff, float(np.abs(gym_next - next_obs).max()))

    assert max_diff < 1e-5, f"trajectories drifted by {max_diff}"
    assert np.all(to_numpy(trunc_arr) == 1.0), "the 200-step limit must truncate"
    print(f"[{backend}] free-running trajectories: max drift {max_diff:.3e}, truncation at step {STEPS}")


@pytest.mark.parametrize("backend", BACKENDS)
def test_reset_distribution(backend):
    env = rl_common.make("mountaincar", 4096, backend=backend, seed=5)
    obs, _ = env.reset()
    position, velocity = to_numpy(obs)[:, 0], to_numpy(obs)[:, 1]
    assert np.all(velocity == 0.0)
    assert position.min() >= mc.RESET_LOW and position.max() <= mc.RESET_HIGH
    assert abs(position.mean() - 0.5 * (mc.RESET_LOW + mc.RESET_HIGH)) < 0.01
    print(f"[{backend}] reset: position ~ U({mc.RESET_LOW}, {mc.RESET_HIGH}), velocity 0")


@pytest.mark.parametrize("backend", BACKENDS)
def test_action_repeat_equals_repeated_actions(backend):
    """One repeat-k step must equal k single steps with the same action."""
    k, n = 8, 32
    rng = np.random.default_rng(2)
    fast = rl_common.make(
        "mountaincar", n, backend=backend, max_episode_steps=25, action_repeat=k, autoreset=False, seed=4
    )
    slow = rl_common.make(
        "mountaincar", n, backend=backend, max_episode_steps=200, action_repeat=1, autoreset=False, seed=4
    )
    fast.reset()
    slow.reset()
    slow.set_state(to_numpy(fast.obs))

    for _ in range(10):
        actions = rng.integers(0, 3, size=n).astype(np.int32)
        _, reward, terminated, _, _ = fast.step(actions)
        sloreward_arr = np.zeros(n)
        done = to_numpy(terminated) * 0
        for _ in range(k):
            _, r, te, _, _ = slow.step(actions)
            sloreward_arr += (done == 0) * to_numpy(r)
            done = np.maximum(done, to_numpy(te))
        assert np.allclose(to_numpy(reward), sloreward_arr), (to_numpy(reward), sloreward_arr)
        assert np.array_equal(to_numpy(terminated), done)
        if done.any():
            break
        assert np.allclose(to_numpy(fast.obs), to_numpy(slow.obs), atol=1e-6)
    print(f"[{backend}] action_repeat={k} matches {k} single steps (state, reward and termination)")


@pytest.mark.parametrize("backend", BACKENDS)
def test_random_policy_cannot_solve_but_held_actions_can(backend):
    """The exploration wall that action_repeat exists to climb."""
    n = 2048
    rng = np.random.default_rng(0)
    reached = {}
    for repeat in (1, 8):
        env = rl_common.make(
            "mountaincar",
            n,
            backend=backend,
            max_episode_steps=200 // repeat,
            action_repeat=repeat,
            autoreset=False,
            seed=1,
        )
        env.reset()
        alive = np.ones(n, dtype=bool)
        solved = np.zeros(n, dtype=bool)
        for _ in range(200 // repeat):
            _, _, terminated, truncated, _ = env.step(rng.integers(0, 3, size=n).astype(np.int32))
            solved |= alive & (to_numpy(terminated) > 0)
            alive &= ~((to_numpy(terminated) + to_numpy(truncated)) > 0)
        reached[repeat] = int(solved.sum())

    assert reached[1] == 0, f"uniform random should never reach the flag, got {reached[1]}"
    assert reached[8] > 0, "held actions should occasionally resonate up the hill"
    print(f"[{backend}] random policy reaches the flag {reached[1]}/{n} at repeat 1, {reached[8]}/{n} at repeat 8")


@pytest.mark.parametrize("backend", BACKENDS)
def test_training_solves_mountain_car(backend):
    cfg = rl_common.default_config("mountaincar", backend=backend, seed=0)
    trainer = rl_common.make_trainer(cfg)
    trainer.train(log_every=0)
    result = trainer.evaluate(num_envs=64)
    assert result["mean_return"] > -110.0, f"not solved: {result}"
    print(f"[{backend}] PPO solves MountainCar-v0: greedy return {result['mean_return']:.1f} "
          f"+/- {result['std_return']:.1f}")


if __name__ == "__main__":
    for backend in BACKENDS:
        test_single_step_dynamics_match(backend)
        test_full_episode_matches(backend)
        test_reset_distribution(backend)
        test_action_repeat_equals_repeated_actions(backend)
        test_random_policy_cannot_solve_but_held_actions_can(backend)
        test_training_solves_mountain_car(backend)
    print("all mountain car checks passed")
