"""Backend-agnostic pieces of a training run: the loop skeleton, logging and
greedy evaluation.

A backend's trainer only has to provide ``iterate(lr) -> metrics`` (one rollout
plus one update) and, for evaluation, a greedy action function.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np

from .arrays import to_numpy
from .config import PPOConfig


def format_iteration(stats: dict) -> str:
    return (
        f"iter {stats['iteration']:4d}/{stats['num_iterations']}  step {stats['global_step']:>9,}  "
        f"return {stats['episodic_return']:8.1f}  len {stats['episodic_length']:6.1f}  "
        f"entropy {stats['entropy']:.3f}  kl {stats['approx_kl']:.4f}  "
        f"clipfrac {stats['clipfrac']:.3f}  v_loss {stats['value_loss']:8.2f}  "
        f"{stats['sps']:,.0f} steps/s"
    )


def run_training_loop(
    cfg: PPOConfig,
    iterate: Callable[[float], dict],
    *,
    log_every: int = 1,
    callback: Callable[[dict], None] | None = None,
) -> list[dict]:
    """Drive ``cfg.num_iterations`` iterations, handling annealing and logging.

    ``iterate(lr)`` runs one rollout + update and returns a metrics dict; the
    keys ``episodic_return`` / ``episodic_length`` / ``episodes`` describe the
    episodes that finished during it.
    """
    history = []
    global_step = 0
    start = time.time()

    for iteration in range(1, cfg.num_iterations + 1):
        lr = cfg.learning_rate_at(iteration)
        stats = dict(iterate(lr))
        global_step += cfg.batch_size
        elapsed = time.time() - start
        stats.update(
            iteration=iteration,
            num_iterations=cfg.num_iterations,
            global_step=global_step,
            lr=lr,
            sps=global_step / max(elapsed, 1e-9),
            elapsed=elapsed,
        )
        history.append(stats)
        if callback is not None:
            callback(stats)
        elif log_every and (iteration % log_every == 0 or iteration == cfg.num_iterations):
            print(format_iteration(stats))
    return history


def evaluate_greedy(env, act: Callable, *, max_steps: int) -> dict:
    """Run ``act`` until every environment has finished one episode.

    ``env`` follows the vector-env protocol of either backend and ``act`` maps
    that backend's observations to that backend's actions.
    """
    obs, _ = env.reset()
    n = env.num_envs
    alive = np.ones(n, dtype=bool)
    returns = np.zeros(n, dtype=np.float64)
    lengths = np.zeros(n, dtype=np.int64)

    for _ in range(max_steps):
        obs, reward, terminated, truncated, _ = env.step(act(obs))
        done = (to_numpy(terminated) + to_numpy(truncated)) > 0
        returns += alive * to_numpy(reward)
        lengths += alive
        alive &= ~done
        if not alive.any():
            break

    return {
        "mean_return": float(returns.mean()),
        "std_return": float(returns.std()),
        "min_return": float(returns.min()),
        "max_return": float(returns.max()),
        "mean_length": float(lengths.mean()),
        "num_episodes": int(n),
    }
