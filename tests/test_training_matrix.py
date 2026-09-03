"""Every problem trains on every backend.

The project's claim is a matrix -- three RL problems, three ways of training
them -- and this is the test that holds it to that.  The matrix is *derived*
from the registry rather than written out, so a fourth environment or a fourth
backend is covered from the moment it is registered.

The budget is deliberately tiny, because what this asserts is that every pair
**runs**: it constructs, completes its iterations with finite metrics, and
evaluates.  Whether a pair **learns** is a separate and far more expensive
question, asserted in each problem's own test file for the seven pairs where a
short run is affordable.  The two that are not are SB3 on mountain car and on
the lander: SB3 pays a host round-trip per step, so those are minutes rather
than seconds and stay smoke-tested here.

Run with pytest, or directly:  python tests/test_training_matrix.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rl_common
import support

METRICS = ("entropy", "approx_kl", "clipfrac", "value_loss")


@pytest.mark.parametrize("backend", rl_common.BACKENDS)
@pytest.mark.parametrize("env_id", rl_common.env_ids())
def test_every_problem_runs_on_every_backend(env_id, backend):
    cfg = support.short_config(env_id, backend, iterations=2, num_envs=64, num_steps=16)
    trainer = rl_common.make_trainer(cfg)

    history = trainer.train(log_every=0)
    assert len(history) == cfg.num_iterations, f"{cfg.num_iterations} iterations asked for, {len(history)} ran"
    for stats in history:
        for key in METRICS:
            assert np.isfinite(stats[key]), f"{env_id}/{backend} iteration {stats['iteration']}: {key}={stats[key]}"

    result = trainer.evaluate(num_envs=16)
    assert np.isfinite(result["mean_return"]), result
    assert result["num_episodes"] == 16
    print(f"{env_id:12s} {backend:5s} {cfg.num_iterations} iters, greedy eval {result['mean_return']:8.1f}")


if __name__ == "__main__":
    for env in rl_common.env_ids():
        for be in rl_common.BACKENDS:
            test_every_problem_runs_on_every_backend(env, be)
    print("every problem runs on every backend")
