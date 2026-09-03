"""Correctness checks for the Warp backend's PPO machinery.

These cover what is specific to the Warp implementation: the GAE kernel, the
``wp.Tape`` gradients behind the clipped surrogate loss, and CUDA-graph capture.
The JAX side is covered by ``tests/test_backend_parity.py``, which pins its GAE,
its networks and its learning curve to these.

Run with pytest, or directly:  python tests/test_warp_ppo.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import warp as wp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rl_common
from rl_common import PPOConfig
from warp_rl import PPO
from warp_rl import kernels as K


def _reference_gae(rewards, values, boot_values, terminated, truncated, gamma, lam):
    """Straightforward numpy GAE, used as the ground truth for the kernel."""
    T, N = rewards.shape
    adv = np.zeros((T, N), dtype=np.float64)
    running = np.zeros(N, dtype=np.float64)
    for t in reversed(range(T)):
        done = np.maximum(terminated[t], truncated[t])
        next_value = boot_values[t] * (1.0 - terminated[t])
        delta = rewards[t] + gamma * next_value - values[t]
        running = delta + gamma * lam * (1.0 - done) * running
        adv[t] = running
    return adv, adv + values


def test_gae_matches_numpy_reference():
    T, N = 16, 8
    rng = np.random.default_rng(0)
    rewards = np.ones((T, N), dtype=np.float32)
    values = rng.normal(scale=5.0, size=(T, N)).astype(np.float32)
    boot = rng.normal(scale=5.0, size=(T, N)).astype(np.float32)
    terminated = (rng.random((T, N)) < 0.1).astype(np.float32)
    truncated = ((rng.random((T, N)) < 0.05) * (1.0 - terminated)).astype(np.float32)
    gamma, lam = 0.99, 0.95

    dev = wp.get_device()
    args = [wp.array(a, dtype=wp.float32, device=dev) for a in (rewards, values, boot, terminated, truncated)]
    adv = wp.zeros((T, N), dtype=wp.float32, device=dev)
    ret = wp.zeros((T, N), dtype=wp.float32, device=dev)
    wp.launch(K.gae_kernel, dim=N, inputs=[*args, gamma, lam, adv, ret], device=dev)

    ref_adv, ref_ret = _reference_gae(rewards, values, boot, terminated, truncated, gamma, lam)
    assert np.allclose(adv.numpy(), ref_adv, atol=1e-4), "GAE advantages differ from the reference"
    assert np.allclose(ret.numpy(), ref_ret, atol=1e-4), "GAE returns differ from the reference"
    print(f"GAE kernel matches numpy reference (max diff {np.abs(adv.numpy() - ref_adv).max():.2e})")


def _make_trainer(**kwargs) -> PPO:
    cfg = PPOConfig(
        backend="warp",
        num_envs=8,
        num_steps=8,
        num_minibatches=1,
        update_epochs=1,
        total_timesteps=64,
        use_graph=False,
        **kwargs,
    )
    return PPO(cfg)


def test_loss_gradients_match_finite_differences():
    trainer = _make_trainer()
    cfg = trainer.cfg
    M = cfg.minibatch_size
    rng = np.random.default_rng(0)

    trainer.mb_obs.assign(rng.normal(size=(M, 4)).astype(np.float32))
    trainer.mb_act.assign(rng.integers(0, 2, size=M).astype(np.int32))
    trainer.mb_logp.assign((rng.normal(size=M) * 0.1 - 0.7).astype(np.float32))
    trainer.mb_adv.assign(rng.normal(size=M).astype(np.float32))
    trainer.mb_ret.assign((rng.normal(size=M) * 5.0).astype(np.float32))

    def forward() -> float:
        # the caller zeroes the loss accumulator; keeping that launch out of
        # the tape avoids recording a non-differentiable kernel
        logits = trainer.agent.policy(trainer.mb_obs)
        values = trainer.agent.value(trainer.mb_obs)
        wp.launch(
            trainer.ppo_loss,
            dim=M,
            inputs=[
                logits,
                values,
                trainer.mb_act,
                trainer.mb_logp,
                trainer.mb_adv,
                trainer.mb_ret,
                cfg.clip_coef,
                cfg.vf_coef,
                cfg.ent_coef,
                1.0 / M,
                trainer.loss,
            ],
            device=trainer.device,
        )
        return float(trainer.loss.numpy()[0]), logits, values

    trainer._zero(trainer._loss_view)
    tape = wp.Tape()
    with tape:
        _, logits, values = forward()
    tape.backward(loss=trainer.loss)

    params = [trainer.agent.policy.parameters()[0], trainer.agent.value.parameters()[0]]
    names = ["policy.0.weight", "value.0.weight"]
    eps = 1e-3
    worst = 0.0
    for param, name in zip(params, names):
        grad = param.grad.numpy().copy()
        host = param.numpy().copy()
        for idx in [(0, 0), (1, 2), (5, 3), (17, 1)]:
            original = host[idx]
            perturbed = host.copy()
            perturbed[idx] = original + eps
            param.assign(perturbed)
            trainer._zero(trainer._loss_view)
            plus, _, _ = forward()
            perturbed[idx] = original - eps
            param.assign(perturbed)
            trainer._zero(trainer._loss_view)
            minus, _, _ = forward()
            param.assign(host)

            numeric = (plus - minus) / (2.0 * eps)
            analytic = grad[idx]
            scale = max(1.0, abs(numeric), abs(analytic))
            err = abs(numeric - analytic) / scale
            worst = max(worst, err)
            assert err < 2e-2, f"{name}{idx}: autodiff {analytic:.6f} vs finite diff {numeric:.6f}"
    print(f"PPO loss gradients match finite differences (worst relative error {worst:.2e})")


def test_training_improves_return():
    cfg = rl_common.default_config(
        "cartpole", backend="warp", num_envs=128, num_steps=32, total_timesteps=128 * 32 * 12, seed=0
    )
    trainer = rl_common.make_trainer(cfg)
    history = []
    trainer.train(callback=lambda s: history.append(s["episodic_return"]))

    first, last = np.mean(history[:2]), np.mean(history[-2:])
    assert last > first + 30.0, f"return did not improve: {first:.1f} -> {last:.1f}"

    result = trainer.evaluate(num_envs=32)
    assert result["mean_return"] > 100.0, f"greedy policy is weak: {result}"
    print(
        f"training improves return {first:.1f} -> {last:.1f} in {cfg.num_iterations} iterations "
        f"(greedy eval {result['mean_return']:.1f})"
    )


def test_rollout_graph_matches_eager():
    """The CUDA-graph rollout must produce the same data as the eager one."""
    if not wp.get_device().is_cuda:
        print("skipping graph check on CPU")
        return
    outputs = []
    for use_graph in (False, True):
        cfg = PPOConfig(
            backend="warp", num_envs=64, num_steps=8, total_timesteps=64 * 8 * 3, seed=1, use_graph=use_graph
        )
        trainer = PPO(cfg)
        trainer.env.reset(seed=cfg.seed)
        for _ in range(3):  # third rollout is graph-replayed when use_graph=True
            trainer.rollout()
        outputs.append((trainer.obs_buf.numpy(), trainer.act_buf.numpy(), trainer.logp_buf.numpy()))

    for eager, graph, name in zip(outputs[0], outputs[1], ("obs", "actions", "log_probs")):
        assert np.allclose(eager, graph, atol=1e-6), f"graph rollout differs from eager in {name}"
    print("CUDA-graph rollout is bit-identical to the eager rollout")


if __name__ == "__main__":
    wp.init()
    test_gae_matches_numpy_reference()
    test_loss_gradients_match_finite_differences()
    test_rollout_graph_matches_eager()
    test_training_improves_return()
    print("all PPO checks passed")
