"""Checks for the Gymnasium adapter and the Stable-Baselines3 backend.

The point of this backend is interoperability: our environments should look like
ordinary Gymnasium environments to the outside world, and SB3's reference PPO
should be able to train on them unmodified.  These tests check the adapters
against the upstream checkers and against the vectorized environment they wrap.

Run with pytest, or directly:  python tests/test_sb3_backend.py
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
import support
from rl_common.gym_api import GymEnv, gym_id, register

BACKENDS = ["warp", "jax"]
ENV_IDS = ["cartpole", "mountaincar", "lunarlander"]


@pytest.mark.parametrize("env_id", ENV_IDS)
@pytest.mark.parametrize("backend", BACKENDS)
def test_gym_adapter_passes_the_sb3_env_checker(env_id, backend):
    from stable_baselines3.common.env_checker import check_env

    env = GymEnv(env_id, backend=backend)
    check_env(env, warn=True, skip_render_check=True)
    env.close()
    print(f"[{backend}] {env_id}: passes stable_baselines3.common.env_checker.check_env")


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_gym_adapter_matches_the_vector_env(env_id):
    """The single-env view must step exactly like the vectorized one."""
    rng = np.random.default_rng(0)
    adapter = GymEnv(env_id, backend="warp", seed=3)
    vec = rl_common.make(env_id, 1, backend="warp", autoreset=False, seed=3)

    obs_a, _ = adapter.reset(seed=7)
    obs_v, _ = vec.reset(seed=7)
    assert np.allclose(obs_a, to_numpy(obs_v)[0], atol=1e-6)

    num_actions = rl_common.spec(env_id).num_actions
    for _ in range(100):
        action = int(rng.integers(0, num_actions))
        obs_a, reward_a, term_a, trunc_a, _ = adapter.step(action)
        obs_v, reward_v, term_v, trunc_v, _ = vec.step(np.array([action], dtype=np.int32))
        assert np.allclose(obs_a, to_numpy(obs_v)[0], atol=1e-6)
        assert abs(reward_a - float(to_numpy(reward_v)[0])) < 1e-5
        assert term_a == bool(to_numpy(term_v)[0]) and trunc_a == bool(to_numpy(trunc_v)[0])
        if term_a or trunc_a:
            break
    adapter.close()
    print(f"{env_id}: the Gymnasium adapter matches the vectorized environment")


def test_environments_are_registered_with_gymnasium():
    ids = register()
    assert gym_id("cartpole", "warp") in ids
    env = gym.make(gym_id("lunarlander", "jax"))
    obs, _ = env.reset(seed=0)
    assert obs.shape == (8,)
    obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
    assert obs.shape == (8,) and np.isfinite(reward)
    env.close()
    print(f"registered {len(ids)} environments with Gymnasium, e.g. {ids[0]}, {ids[-1]}")


def test_sb3_vec_env_adapter_semantics():
    """dones, terminal observations and episode infos must follow SB3's contract."""
    from sb3_rl.vec_env import VecEnvAdapter

    env = rl_common.make("cartpole", 32, backend="warp", max_episode_steps=20, seed=1)
    venv = VecEnvAdapter(env)
    obs = venv.reset()
    assert obs.shape == (32, 4) and obs.dtype == np.float32

    rng = np.random.default_rng(0)
    saw_terminal, saw_truncation, episodes = 0, 0, 0
    for _ in range(60):
        obs, rewards, dones, infos = venv.step(rng.integers(0, 2, size=32).astype(np.int32))
        assert obs.shape == (32, 4) and rewards.shape == (32,) and dones.shape == (32,)
        for i in np.flatnonzero(dones):
            assert "terminal_observation" in infos[i], "SB3 needs the pre-reset observation"
            assert infos[i]["terminal_observation"].shape == (4,)
            saw_terminal += 1
            if infos[i].get("TimeLimit.truncated"):
                saw_truncation += 1
            episode = infos[i].get("episode")
            assert episode is not None and episode["l"] > 0
            episodes += 1
    assert saw_terminal > 0 and saw_truncation > 0 and episodes == saw_terminal
    venv.close()
    print(f"SB3 VecEnv adapter: {episodes} episodes, {saw_truncation} of them truncated at the time limit")


@pytest.mark.parametrize("env_backend", ["warp", "jax"])
def test_sb3_trains_on_our_environments(env_backend):
    _, first, last = support.assert_learns("cartpole", "sb3", iterations=8, gain=30.0, env_backend=env_backend)
    print(f"sb3 learner on the {env_backend} cartpole: return {first:.0f} -> {last:.0f}")


def test_sb3_agent_roundtrips_weights(tmp_path=None):
    import tempfile

    directory = tempfile.mkdtemp() if tmp_path is None else str(tmp_path)
    path = os.path.join(directory, "policy.zip")

    agent = rl_common.make_agent("cartpole", backend="sb3", seed=0)
    obs = np.random.default_rng(0).normal(size=(16, 4)).astype(np.float32)
    before = agent.act_numpy(obs)
    agent.save(path)

    restored = rl_common.make_agent("cartpole", backend="sb3", seed=1)
    restored.load(path)
    assert np.array_equal(before, restored.act_numpy(obs))
    print("sb3 policies round-trip through save/load")


if __name__ == "__main__":
    for backend in BACKENDS:
        for env_id in ENV_IDS:
            test_gym_adapter_passes_the_sb3_env_checker(env_id, backend)
    for env_id in ENV_IDS:
        test_gym_adapter_matches_the_vector_env(env_id)
    test_environments_are_registered_with_gymnasium()
    test_sb3_vec_env_adapter_semantics()
    for env_backend in ("warp", "jax"):
        test_sb3_trains_on_our_environments(env_backend)
    test_sb3_agent_roundtrips_weights()
    print("all sb3 backend checks passed")
