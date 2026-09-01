"""Check the Warp MountainCar against Gymnasium's MountainCar-v0.

The physics is a pure port, so parity here is exact (float32 rounding aside),
exactly as for CartPole.  The extra tests cover ``action_repeat`` -- our one
addition -- and the exploration wall that motivates it.

Run with pytest, or directly:  python tests/test_mountain_car.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import warp as wp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym  # noqa: E402

import warp_rl  # noqa: E402
from warp_rl.envs import mountain_car as mc  # noqa: E402

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


def test_single_step_dynamics_match():
    """Re-sync the Warp state each step so only one-step dynamics are compared."""
    rng = np.random.default_rng(0)
    envs, states = _gym_envs(NUM_ENVS)
    warp_env = warp_rl.make("mountaincar", NUM_ENVS, max_episode_steps=STEPS, autoreset=False)
    warp_env.reset(seed=0)

    max_diff = 0.0
    for _ in range(STEPS):
        warp_env.set_state(states)
        actions = rng.integers(0, 3, size=NUM_ENVS).astype(np.int32)
        _, w_reward, w_term, _, w_info = warp_env.step(actions)
        warp_next = w_info["final_observation"].numpy()

        next_states, terms = [], []
        for i, env in enumerate(envs):
            obs, reward, terminated, _, _ = env.step(int(actions[i]))
            assert reward == w_reward.numpy()[i]
            next_states.append(obs)
            terms.append(float(terminated))
            if terminated:
                obs, _ = env.reset()
                next_states[-1] = obs
        next_states = np.asarray(next_states, dtype=np.float32)

        live = np.asarray(terms) == 0.0
        assert np.array_equal(np.asarray(terms), w_term.numpy()), "termination flags differ"
        max_diff = max(max_diff, float(np.abs(next_states[live] - warp_next[live]).max(initial=0.0)))
        states = next_states

    assert max_diff < 1e-6, f"one-step dynamics differ by {max_diff}"
    print(f"one-step dynamics: max |warp - gymnasium| = {max_diff:.3e} over {STEPS} steps")


def test_full_episode_matches():
    """Free-running episodes: same actions, same trajectory, same 200-step limit."""
    rng = np.random.default_rng(1)
    envs, states = _gym_envs(NUM_ENVS, seed=100)
    warp_env = warp_rl.make("mountaincar", NUM_ENVS, max_episode_steps=STEPS, autoreset=False)
    warp_env.reset(seed=0)
    warp_env.set_state(states)

    max_diff = 0.0
    for _ in range(STEPS):
        actions = rng.integers(0, 3, size=NUM_ENVS).astype(np.int32)
        _, _, _, w_trunc, w_info = warp_env.step(actions)
        warp_next = w_info["final_observation"].numpy()
        gym_next = np.asarray([env.step(int(actions[i]))[0] for i, env in enumerate(envs)], dtype=np.float32)
        max_diff = max(max_diff, float(np.abs(gym_next - warp_next).max()))

    assert max_diff < 1e-5, f"trajectories drifted by {max_diff}"
    assert np.all(w_trunc.numpy() == 1.0), "the 200-step limit must truncate"
    print(f"free-running trajectories: max drift {max_diff:.3e}, truncation at step {STEPS}")


def test_reset_distribution():
    env = warp_rl.make("mountaincar", 4096, seed=5)
    obs, _ = env.reset()
    position, velocity = obs.numpy()[:, 0], obs.numpy()[:, 1]
    assert np.all(velocity == 0.0)
    assert position.min() >= mc.RESET_LOW and position.max() <= mc.RESET_HIGH
    assert abs(position.mean() - 0.5 * (mc.RESET_LOW + mc.RESET_HIGH)) < 0.01
    print(f"reset: position ~ U({mc.RESET_LOW}, {mc.RESET_HIGH}), velocity 0")


def test_action_repeat_equals_repeated_actions():
    """One repeat-k step must equal k single steps with the same action."""
    k, n = 8, 32
    rng = np.random.default_rng(2)
    fast = warp_rl.make("mountaincar", n, max_episode_steps=25, action_repeat=k, autoreset=False, seed=4)
    slow = warp_rl.make("mountaincar", n, max_episode_steps=200, action_repeat=1, autoreset=False, seed=4)
    fast.reset()
    slow.reset()
    slow.set_state(fast.obs.numpy())

    for _ in range(10):
        actions = rng.integers(0, 3, size=n).astype(np.int32)
        _, reward, terminated, _, _ = fast.step(actions)
        slow_reward = np.zeros(n)
        done = terminated.numpy() * 0
        for _ in range(k):
            _, r, te, _, _ = slow.step(actions)
            slow_reward += (done == 0) * r.numpy()
            done = np.maximum(done, te.numpy())
        assert np.allclose(reward.numpy(), slow_reward), (reward.numpy(), slow_reward)
        assert np.array_equal(terminated.numpy(), done)
        if done.any():
            break
        assert np.allclose(fast.obs.numpy(), slow.obs.numpy(), atol=1e-6)
    print(f"action_repeat={k} matches {k} single steps (state, reward and termination)")


def test_random_policy_cannot_solve_but_held_actions_can():
    """The exploration wall that action_repeat exists to climb."""
    n = 2048
    rng = np.random.default_rng(0)
    reached = {}
    for repeat in (1, 8):
        env = warp_rl.make(
            "mountaincar", n, max_episode_steps=200 // repeat, action_repeat=repeat, autoreset=False, seed=1
        )
        env.reset()
        alive = np.ones(n, dtype=bool)
        solved = np.zeros(n, dtype=bool)
        for _ in range(200 // repeat):
            _, _, terminated, truncated, _ = env.step(rng.integers(0, 3, size=n).astype(np.int32))
            solved |= alive & (terminated.numpy() > 0)
            alive &= ~((terminated.numpy() + truncated.numpy()) > 0)
        reached[repeat] = int(solved.sum())

    assert reached[1] == 0, f"uniform random should never reach the flag, got {reached[1]}"
    assert reached[8] > 0, "held actions should occasionally resonate up the hill"
    print(f"random policy reaches the flag {reached[1]}/{n} times at repeat 1, {reached[8]}/{n} at repeat 8")


def test_training_solves_mountain_car():
    cfg = warp_rl.default_config("mountaincar", seed=0)
    trainer = warp_rl.PPO(cfg)
    trainer.train(log_every=0)
    result = trainer.evaluate(num_envs=64)
    assert result["mean_return"] > -110.0, f"not solved: {result}"
    print(f"PPO solves MountainCar-v0: greedy return {result['mean_return']:.1f} +/- {result['std_return']:.1f}")


if __name__ == "__main__":
    wp.init()
    test_single_step_dynamics_match()
    test_full_episode_matches()
    test_reset_distribution()
    test_action_repeat_equals_repeated_actions()
    test_random_policy_cannot_solve_but_held_actions_can()
    test_training_solves_mountain_car()
    print("all mountain car checks passed")
