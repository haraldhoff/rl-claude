"""Warp kernels shared by the RL machinery (environment-agnostic).

Action sampling, rollout stores, GAE, advantage normalization, minibatch
gathers and the PPO losses all live here; nothing in this module knows what
environment it is training on.
"""

from __future__ import annotations

import warp as wp


# ---------------------------------------------------------------------------
# policy actions
# ---------------------------------------------------------------------------


def make_action_kernels(num_actions: int):
    """Kernels are generated per action count so their loops unroll."""

    @wp.kernel(enable_backward=False)
    def sample_actions(
        logits: wp.array2d(dtype=wp.float32),
        rng_states: wp.array(dtype=wp.uint32),
        actions: wp.array(dtype=wp.int32),
        log_probs: wp.array(dtype=wp.float32),
    ):
        i = wp.tid()
        # log-sum-exp for a numerically stable categorical distribution
        m = logits[i, 0]
        for a in range(1, wp.static(num_actions)):
            m = wp.max(m, logits[i, a])
        s = float(0.0)
        for a in range(wp.static(num_actions)):
            s += wp.exp(logits[i, a] - m)
        lse = m + wp.log(s)

        rng = rng_states[i]
        u = wp.randf(rng)
        rng_states[i] = rng

        # inverse-CDF sampling
        acc = float(0.0)
        choice = wp.static(num_actions) - 1
        for a in range(wp.static(num_actions)):
            acc += wp.exp(logits[i, a] - lse)
            if u < acc and choice == wp.static(num_actions) - 1:
                choice = a
        actions[i] = choice
        log_probs[i] = logits[i, choice] - lse

    @wp.kernel(enable_backward=False)
    def greedy_actions(logits: wp.array2d(dtype=wp.float32), actions: wp.array(dtype=wp.int32)):
        i = wp.tid()
        best = 0
        m = logits[i, 0]
        for a in range(1, wp.static(num_actions)):
            if logits[i, a] > m:
                m = logits[i, a]
                best = a
        actions[i] = best

    return sample_actions, greedy_actions


# ---------------------------------------------------------------------------
# PPO objective
# ---------------------------------------------------------------------------


def make_loss_kernels(num_actions: int):
    @wp.kernel
    def ppo_loss(
        logits: wp.array2d(dtype=wp.float32),  # (M, A)
        values: wp.array2d(dtype=wp.float32),  # (M, 1)
        actions: wp.array(dtype=wp.int32),
        old_log_probs: wp.array(dtype=wp.float32),
        advantages: wp.array(dtype=wp.float32),
        returns: wp.array(dtype=wp.float32),
        clip_coef: float,
        vf_coef: float,
        ent_coef: float,
        inv_batch: float,
        loss: wp.array(dtype=wp.float32),
    ):
        i = wp.tid()

        m = logits[i, 0]
        for a in range(1, wp.static(num_actions)):
            m = wp.max(m, logits[i, a])
        s = float(0.0)
        for a in range(wp.static(num_actions)):
            s += wp.exp(logits[i, a] - m)
        lse = m + wp.log(s)

        log_prob = logits[i, actions[i]] - lse
        ratio = wp.exp(log_prob - old_log_probs[i])
        adv = advantages[i]

        # clipped surrogate objective (negated: we minimize)
        pg = -wp.min(ratio * adv, wp.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * adv)

        diff = values[i, 0] - returns[i]
        vf = 0.5 * diff * diff

        entropy = float(0.0)
        for a in range(wp.static(num_actions)):
            lp = logits[i, a] - lse
            entropy -= wp.exp(lp) * lp

        wp.atomic_add(loss, 0, (pg + vf_coef * vf - ent_coef * entropy) * inv_batch)

    @wp.kernel(enable_backward=False)
    def ppo_metrics(
        logits: wp.array2d(dtype=wp.float32),
        values: wp.array2d(dtype=wp.float32),
        actions: wp.array(dtype=wp.int32),
        old_log_probs: wp.array(dtype=wp.float32),
        returns: wp.array(dtype=wp.float32),
        clip_coef: float,
        inv_batch: float,
        metrics: wp.array(dtype=wp.float32),  # [entropy, approx_kl, clipfrac, v_loss]
    ):
        i = wp.tid()
        m = logits[i, 0]
        for a in range(1, wp.static(num_actions)):
            m = wp.max(m, logits[i, a])
        s = float(0.0)
        for a in range(wp.static(num_actions)):
            s += wp.exp(logits[i, a] - m)
        lse = m + wp.log(s)

        entropy = float(0.0)
        for a in range(wp.static(num_actions)):
            lp = logits[i, a] - lse
            entropy -= wp.exp(lp) * lp

        log_ratio = logits[i, actions[i]] - lse - old_log_probs[i]
        ratio = wp.exp(log_ratio)
        approx_kl = (ratio - 1.0) - log_ratio  # Schulman's low-variance estimator
        clipped = float(0.0)
        if wp.abs(ratio - 1.0) > clip_coef:
            clipped = 1.0
        diff = values[i, 0] - returns[i]

        wp.atomic_add(metrics, 0, entropy * inv_batch)
        wp.atomic_add(metrics, 1, approx_kl * inv_batch)
        wp.atomic_add(metrics, 2, clipped * inv_batch)
        wp.atomic_add(metrics, 3, 0.5 * diff * diff * inv_batch)

    return ppo_loss, ppo_metrics


# ---------------------------------------------------------------------------
# rollout buffers
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def store_obs(src: wp.array2d(dtype=wp.float32), offset: wp.int32, dst: wp.array2d(dtype=wp.float32)):
    i, k = wp.tid()
    dst[offset + i, k] = src[i, k]


@wp.kernel(enable_backward=False)
def store_f32(src: wp.array(dtype=wp.float32), offset: wp.int32, dst: wp.array(dtype=wp.float32)):
    i = wp.tid()
    dst[offset + i] = src[i]


@wp.kernel(enable_backward=False)
def store_i32(src: wp.array(dtype=wp.int32), offset: wp.int32, dst: wp.array(dtype=wp.int32)):
    i = wp.tid()
    dst[offset + i] = src[i]


@wp.kernel(enable_backward=False)
def column_to_2d(src: wp.array2d(dtype=wp.float32), dst: wp.array2d(dtype=wp.float32)):
    """(T*N, 1) network output -> (T, N) time-major view."""
    t, n = wp.tid()
    dst[t, n] = src[t * dst.shape[1] + n, 0]


@wp.kernel(enable_backward=False)
def flatten_2d(src: wp.array2d(dtype=wp.float32), dst: wp.array(dtype=wp.float32)):
    """(T, N) -> (T*N,)"""
    t, n = wp.tid()
    dst[t * src.shape[1] + n] = src[t, n]


# ---------------------------------------------------------------------------
# advantages
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def gae_kernel(
    rewards: wp.array2d(dtype=wp.float32),  # (T, N)
    values: wp.array2d(dtype=wp.float32),
    boot_values: wp.array2d(dtype=wp.float32),  # V(s') of the pre-auto-reset next state
    terminated: wp.array2d(dtype=wp.float32),
    truncated: wp.array2d(dtype=wp.float32),
    gamma: float,
    gae_lambda: float,
    advantages: wp.array2d(dtype=wp.float32),
    returns: wp.array2d(dtype=wp.float32),
):
    n = wp.tid()
    adv = float(0.0)
    for t in range(rewards.shape[0] - 1, -1, -1):
        term = terminated[t, n]
        done = wp.max(term, truncated[t, n])
        # bootstrap through truncation, but not through termination
        next_value = boot_values[t, n] * (1.0 - term)
        delta = rewards[t, n] + gamma * next_value - values[t, n]
        adv = delta + gamma * gae_lambda * (1.0 - done) * adv
        advantages[t, n] = adv
        returns[t, n] = adv + values[t, n]


@wp.kernel(enable_backward=False)
def zero_kernel(x: wp.array(dtype=wp.float32)):
    """Zeroing through a kernel rather than ``array.zero_()``.

    Warp's memset path costs a fixed ~130 us per call on some drivers, which
    dominates a PPO minibatch (23 gradient arrays x 80 minibatches per
    iteration); a plain kernel launch is ~30x cheaper here.
    """
    x[wp.tid()] = 0.0


@wp.kernel(enable_backward=False)
def sum_and_sumsq(x: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    wp.atomic_add(out, 0, x[i])
    wp.atomic_add(out, 1, x[i] * x[i])


@wp.kernel(enable_backward=False)
def normalize(x: wp.array(dtype=wp.float32), moments: wp.array(dtype=wp.float32), count: float, eps: float):
    i = wp.tid()
    mean = moments[0] / count
    var = wp.max(moments[1] / count - mean * mean, 0.0)
    x[i] = (x[i] - mean) / (wp.sqrt(var) + eps)


# ---------------------------------------------------------------------------
# minibatch gathers
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def gather_obs(
    src: wp.array2d(dtype=wp.float32),
    idx: wp.array(dtype=wp.int32),
    offset: wp.int32,
    dst: wp.array2d(dtype=wp.float32),
):
    i, k = wp.tid()
    dst[i, k] = src[idx[offset + i], k]


@wp.kernel(enable_backward=False)
def gather_f32(
    src: wp.array(dtype=wp.float32),
    idx: wp.array(dtype=wp.int32),
    offset: wp.int32,
    dst: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    dst[i] = src[idx[offset + i]]


@wp.kernel(enable_backward=False)
def gather_i32(
    src: wp.array(dtype=wp.int32),
    idx: wp.array(dtype=wp.int32),
    offset: wp.int32,
    dst: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    dst[i] = src[idx[offset + i]]
