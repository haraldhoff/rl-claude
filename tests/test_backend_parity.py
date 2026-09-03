"""The Warp and JAX implementations must be the same thing twice.

Both backends read their constants from ``rl_common.specs``, but the physics,
the networks and the PPO maths are written independently -- these tests pin
them to each other: same state and actions give the same transitions, the same
weights give the same logits and values, and the same rollout gives the same
advantages.

The JAX side runs on the **CPU** here, via :data:`JAX_DEVICE`.  These are
equality tests to a fixed tolerance, and GPU JAX computes f32 matmuls in TF32
by default -- that alone put the network comparison at 4.9e-04 against a 1e-04
bound, which measures the GPU's matmul precision rather than whether the two
implementations agree.  The one test here that asks about *behaviour* rather
than numerical agreement, ``test_both_backends_learn_cartpole``, is left on
whatever device the backend picks for itself.

Run with pytest, or directly:  python tests/test_backend_parity.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rl_common
from rl_common import to_numpy

NUM_ENVS = 16

# Where the JAX side of every comparison runs; see the module docstring.
JAX_DEVICE = "cpu"


@pytest.fixture(autouse=True)
def _jax_defaults_to_cpu():
    """Put *uncommitted* JAX work on the CPU too.

    Passing ``device=JAX_DEVICE`` covers everything built through the registry,
    but not the bare ``jnp`` arrays a test makes for itself -- and it cannot,
    because ``resolve_device(None)`` asks for ``jax.devices()[0]`` and never
    consults this default.  Both mechanisms are needed; neither subsumes the
    other.  jax is imported here rather than at module scope so the file still
    collects when the extra is absent.
    """
    import jax

    with jax.default_device(jax.devices("cpu")[0]):
        yield


def _pair(env_id: str, num_envs: int = NUM_ENVS, **kwargs):
    warp_env = rl_common.make(env_id, num_envs, backend="warp", autoreset=False, seed=0, **kwargs)
    jax_env = rl_common.make(env_id, num_envs, backend="jax", autoreset=False, seed=0, device=JAX_DEVICE, **kwargs)
    warp_env.reset()
    jax_env.reset()
    return warp_env, jax_env


def _step_both(warp_env, jax_env, actions):
    w_obs, w_reward, w_term, w_trunc, _ = warp_env.step(actions)
    j_obs, j_reward, j_term, j_trunc, _ = jax_env.step(actions)
    return (
        (to_numpy(w_obs), to_numpy(w_reward), to_numpy(w_term), to_numpy(w_trunc)),
        (to_numpy(j_obs), to_numpy(j_reward), to_numpy(j_term), to_numpy(j_trunc)),
    )


@pytest.mark.parametrize("env_id,num_actions,steps", [("cartpole", 2, 200), ("mountaincar", 3, 200)])
def test_classic_control_single_steps_match(env_id, num_actions, steps):
    """Re-sync both backends each step so only one-step dynamics are compared."""
    rng = np.random.default_rng(0)
    warp_env, jax_env = _pair(env_id, max_episode_steps=steps)
    state = to_numpy(warp_env.obs).copy()

    max_obs_diff = 0.0
    for _ in range(steps):
        warp_env.set_state(state)
        jax_env.set_state(state)
        actions = rng.integers(0, num_actions, size=NUM_ENVS).astype(np.int32)
        (w_obs, w_reward, w_term, _), (j_obs, j_reward, j_term, _) = _step_both(warp_env, jax_env, actions)
        max_obs_diff = max(max_obs_diff, float(np.abs(w_obs - j_obs).max()))
        assert np.allclose(w_reward, j_reward, atol=1e-5), "rewards differ"
        assert np.array_equal(w_term, j_term), "termination flags differ"
        state = w_obs

    assert max_obs_diff < 1e-5, f"{env_id}: one-step dynamics differ by {max_obs_diff}"
    print(f"{env_id}: one-step dynamics identical across backends (max |diff| = {max_obs_diff:.2e})")


@pytest.mark.parametrize("env_id,num_actions,steps", [("cartpole", 2, 200), ("mountaincar", 3, 200)])
def test_classic_control_free_running_stays_together(env_id, num_actions, steps):
    """Left to run, the two backends only drift by float32 rounding."""
    rng = np.random.default_rng(0)
    warp_env, jax_env = _pair(env_id, max_episode_steps=steps)
    start = to_numpy(warp_env.obs).copy()
    warp_env.set_state(start)
    jax_env.set_state(start)

    max_obs_diff = 0.0
    for _ in range(steps):
        actions = rng.integers(0, num_actions, size=NUM_ENVS).astype(np.int32)
        (w_obs, _, w_term, _), (j_obs, _, j_term, _) = _step_both(warp_env, jax_env, actions)
        max_obs_diff = max(max_obs_diff, float(np.abs(w_obs - j_obs).max()))
        assert np.array_equal(w_term, j_term), "termination flags differ"

    assert max_obs_diff < 1e-2, f"{env_id}: trajectories diverged by {max_obs_diff}"
    print(f"{env_id}: free-running drift over {steps} steps = {max_obs_diff:.2e}")


def test_mountain_car_action_repeat_matches():
    rng = np.random.default_rng(1)
    warp_env, jax_env = _pair("mountaincar", max_episode_steps=25, action_repeat=8)
    start = to_numpy(warp_env.obs).copy()
    warp_env.set_state(start)
    jax_env.set_state(start)

    for _ in range(25):
        actions = rng.integers(0, 3, size=NUM_ENVS).astype(np.int32)
        (w_obs, w_reward, w_term, _), (j_obs, j_reward, j_term, _) = _step_both(warp_env, jax_env, actions)
        assert np.allclose(w_obs, j_obs, atol=1e-5)
        assert np.allclose(w_reward, j_reward, atol=1e-5)
        assert np.array_equal(w_term, j_term)
    print("mountaincar with action_repeat=8: warp and jax agree over 25 decisions")


def test_lunar_lander_free_fall_matches():
    """A full unpowered descent -- gravity, leg contacts, crash -- step for step.

    The engines are left off: their impulse dispersion is drawn from each
    backend's own RNG, so a powered trajectory is only statistically comparable.
    """
    warp_env, jax_env = _pair("lunarlander", max_episode_steps=400)

    shared = warp_env.render_state()
    body, terrain = shared["body"].copy(), shared["terrain"].copy()
    warp_env.set_state(body, terrain)
    jax_env.set_state(body, terrain)

    actions = np.zeros(NUM_ENVS, dtype=np.int32)
    max_obs_diff = 0.0
    max_reward_diff = 0.0
    for _ in range(200):
        (w_obs, w_reward, w_term, _), (j_obs, j_reward, j_term, _) = _step_both(warp_env, jax_env, actions)
        max_obs_diff = max(max_obs_diff, float(np.abs(w_obs - j_obs).max()))
        max_reward_diff = max(max_reward_diff, float(np.abs(w_reward - j_reward).max()))
        assert np.array_equal(w_term, j_term), "termination differs during the descent"

    assert max_obs_diff < 1e-3, f"lander states diverged by {max_obs_diff}"
    assert max_reward_diff < 1e-2, f"lander rewards diverged by {max_reward_diff}"
    print(
        f"lunarlander unpowered descent: max |obs diff| = {max_obs_diff:.2e}, "
        f"max |reward diff| = {max_reward_diff:.2e} over 200 steps"
    )


def test_lunar_lander_single_steps_match_exactly():
    """One step from an identical state, with the engines off (no RNG involved)."""
    rng = np.random.default_rng(3)
    warp_env, jax_env = _pair("lunarlander", max_episode_steps=400)
    shared = warp_env.render_state()
    body, terrain = shared["body"].copy(), shared["terrain"].copy()

    max_diff = 0.0
    for _ in range(50):
        body = body.copy()
        body[:, 0] = rng.uniform(4.0, 16.0, NUM_ENVS)  # x
        body[:, 1] = rng.uniform(3.0, 10.0, NUM_ENVS)  # y
        body[:, 2:4] = rng.uniform(-3.0, 3.0, (NUM_ENVS, 2))
        body[:, 4] = rng.uniform(-0.5, 0.5, NUM_ENVS)
        body[:, 5] = rng.uniform(-1.0, 1.0, NUM_ENVS)
        warp_env.set_state(body, terrain)
        jax_env.set_state(body, terrain)

        actions = np.zeros(NUM_ENVS, dtype=np.int32)  # engines off -> deterministic
        (w_obs, w_reward, w_term, _), (j_obs, j_reward, j_term, _) = _step_both(warp_env, jax_env, actions)
        max_diff = max(max_diff, float(np.abs(w_obs - j_obs).max()))
        assert np.allclose(w_reward, j_reward, atol=1e-3), (w_reward, j_reward)
        assert np.array_equal(w_term, j_term)

    assert max_diff < 1e-5, f"one-step lander dynamics differ by {max_diff}"
    print(f"lunarlander one-step dynamics: max |warp - jax| = {max_diff:.2e} over 50 random states")


def _transfer_weights(warp_agent, jax_agent):
    """Copy Warp weights into the Flax parameter tree (same architecture)."""
    state = {k: v.numpy() for k, v in warp_agent.state_dict().items()}
    params = jax_agent.params["params"]
    for net in ("policy", "value"):
        dense = 0
        for layer in range(0, 10, 2):  # Sequential: Linear, Tanh, Linear, Tanh, Linear
            key = f"{net}.{layer}.weight"
            if key not in state:
                continue
            params[net][f"Dense_{dense}"]["kernel"] = state[key].T
            params[net][f"Dense_{dense}"]["bias"] = state[f"{net}.{layer}.bias"].reshape(-1)
            dense += 1
    return jax_agent


def test_networks_agree_given_the_same_weights():
    """Same architecture, same weights -> same logits and values."""
    import warp as wp

    warp_agent = rl_common.make_agent("lunarlander", backend="warp", seed=0)
    jax_agent = rl_common.make_agent("lunarlander", backend="jax", seed=1, device=JAX_DEVICE)
    _transfer_weights(warp_agent, jax_agent)

    obs = np.random.default_rng(0).normal(size=(32, 8)).astype(np.float32)
    warp_logits = warp_agent.policy(wp.array(obs, dtype=wp.float32, device=warp_agent.device)).numpy()
    warp_value = warp_agent.value(wp.array(obs, dtype=wp.float32, device=warp_agent.device)).numpy()[:, 0]
    jax_logits, jax_value = jax_agent.net.apply(jax_agent.params, obs)

    logit_diff = float(np.abs(warp_logits - np.asarray(jax_logits)).max())
    value_diff = float(np.abs(warp_value - np.asarray(jax_value)).max())
    assert logit_diff < 1e-4, f"logits differ by {logit_diff}"
    assert value_diff < 1e-4, f"values differ by {value_diff}"
    print(f"networks agree given identical weights (logits {logit_diff:.2e}, values {value_diff:.2e})")


def test_gae_implementations_agree():
    """The Warp GAE kernel and the JAX scan must produce the same advantages."""
    import jax.numpy as jnp
    import warp as wp

    from jax_rl.ppo import compute_gae
    from warp_rl import kernels as K

    T, N = 32, 8
    rng = np.random.default_rng(0)
    reward = rng.normal(size=(T, N)).astype(np.float32)
    value = rng.normal(scale=5.0, size=(T, N)).astype(np.float32)
    boot = rng.normal(scale=5.0, size=(T, N)).astype(np.float32)
    terminated = (rng.random((T, N)) < 0.1).astype(np.float32)
    truncated = ((rng.random((T, N)) < 0.05) * (1.0 - terminated)).astype(np.float32)
    gamma, lam = 0.99, 0.95

    device = wp.get_device()
    args = [wp.array(a, dtype=wp.float32, device=device) for a in (reward, value, boot, terminated, truncated)]
    adv = wp.zeros((T, N), dtype=wp.float32, device=device)
    ret = wp.zeros((T, N), dtype=wp.float32, device=device)
    wp.launch(K.gae_kernel, dim=N, inputs=[*args, gamma, lam, adv, ret], device=device)

    jax_adv, jax_ret = compute_gae(
        *(jnp.asarray(a) for a in (reward, value, boot, terminated, truncated)), gamma, lam
    )
    adv_diff = float(np.abs(adv.numpy() - np.asarray(jax_adv)).max())
    ret_diff = float(np.abs(ret.numpy() - np.asarray(jax_ret)).max())
    assert adv_diff < 1e-4 and ret_diff < 1e-4, (adv_diff, ret_diff)
    print(f"GAE agrees across backends (advantages {adv_diff:.2e}, returns {ret_diff:.2e})")


def test_both_backends_learn_cartpole():
    """A short run on each backend should improve by a similar amount."""
    results = {}
    for backend in ("warp", "jax"):
        cfg = rl_common.default_config("cartpole", backend=backend, total_timesteps=8192 * 8, seed=0)
        trainer = rl_common.make_trainer(cfg)
        history = []
        trainer.train(callback=lambda s: history.append(s["episodic_return"]))
        results[backend] = (np.mean(history[:2]), np.mean(history[-2:]))

    for backend, (first, last) in results.items():
        assert last > first + 30.0, f"{backend} did not learn: {first:.1f} -> {last:.1f}"
    warp_gain = results["warp"][1] - results["warp"][0]
    jax_gain = results["jax"][1] - results["jax"][0]
    assert abs(warp_gain - jax_gain) < 0.6 * max(warp_gain, jax_gain), (results, "learning curves diverge")
    print(
        f"both backends learn cartpole: warp {results['warp'][0]:.0f} -> {results['warp'][1]:.0f}, "
        f"jax {results['jax'][0]:.0f} -> {results['jax'][1]:.0f}"
    )


if __name__ == "__main__":
    test_classic_control_single_steps_match("cartpole", 2, 200)
    test_classic_control_single_steps_match("mountaincar", 3, 200)
    test_classic_control_free_running_stays_together("cartpole", 2, 200)
    test_classic_control_free_running_stays_together("mountaincar", 3, 200)
    test_mountain_car_action_repeat_matches()
    test_lunar_lander_single_steps_match_exactly()
    test_lunar_lander_free_fall_matches()
    test_networks_agree_given_the_same_weights()
    test_gae_implementations_agree()
    test_both_backends_learn_cartpole()
    print("all backend parity checks passed")
