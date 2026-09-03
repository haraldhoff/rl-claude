"""PPO in JAX + Flax, matching the Warp backend's algorithm step for step.

A whole iteration -- rollout, GAE, and ``update_epochs`` passes over shuffled
minibatches -- is one jitted function: the rollout is a ``lax.scan`` over the
vectorized environment, GAE is a reverse ``lax.scan``, and the update is a scan
over minibatches inside a scan over epochs.  Only the per-iteration metrics
cross back to the host.

The pieces are built by the ``make_*`` factories below, which close over the
config and the network's ``apply``; :meth:`PPO._build_iteration` wires them
together.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import optax

from rl_common import PPOConfig, Trainer

from .agent import ActorCritic
from .vec_env import resolve_device, vec_reset, vec_step


class RunnerState(NamedTuple):
    params: Any
    opt_state: Any
    env_state: Any
    key: jax.Array


# ---------------------------------------------------------------------------
# the pieces of an iteration
# ---------------------------------------------------------------------------


def compute_gae(reward, value, boot_value, terminated, truncated, gamma: float, gae_lambda: float):
    """GAE over ``(T, N)`` arrays; the same recursion as ``warp_rl.kernels.gae_kernel``.

    Bootstrapping uses ``V(s') * (1 - terminated)``, so truncated episodes
    bootstrap and terminated ones do not, and the recursion is cut on either.
    """

    def step(adv, xs):
        reward_t, value_t, boot_t, terminated_t, truncated_t = xs
        done = jnp.maximum(terminated_t, truncated_t)
        delta = reward_t + gamma * boot_t * (1.0 - terminated_t) - value_t
        adv = delta + gamma * gae_lambda * (1.0 - done) * adv
        return adv, adv

    _, advantages = jax.lax.scan(
        step, jnp.zeros_like(reward[0]), (reward, value, boot_value, terminated, truncated), reverse=True
    )
    return advantages, advantages + value


def make_rollout_step(env, apply_fn, cfg: PPOConfig) -> Callable:
    """One step of every environment, as a ``lax.scan`` body."""
    step_env = functools.partial(vec_step, env, max_episode_steps=cfg.max_episode_steps, autoreset=True)
    carried = ("reward", "terminated", "truncated", "done", "done_return", "done_length")

    def rollout_step(carry, _):
        params, env_state, key = carry
        key, action_key = jax.random.split(key)

        obs = env_state.obs
        logits, value = apply_fn(params, obs)
        action = jax.random.categorical(action_key, logits)
        log_prob = jnp.take_along_axis(jax.nn.log_softmax(logits), action[:, None], axis=1)[:, 0]

        env_state, transition = step_env(env_state, action)
        step = {
            "obs": obs,
            "next_obs": transition["final_obs"],
            "action": action,
            "log_prob": log_prob,
            "value": value,
            **{name: transition[name] for name in carried},
        }
        return (params, env_state, key), step

    return rollout_step


def make_loss(apply_fn, cfg: PPOConfig) -> Callable:
    """The clipped surrogate objective, and the metrics we log alongside it."""

    def loss_fn(params, batch):
        logits, value = apply_fn(params, batch["obs"])
        log_probs = jax.nn.log_softmax(logits)
        log_prob = jnp.take_along_axis(log_probs, batch["action"][:, None], axis=1)[:, 0]

        log_ratio = log_prob - batch["log_prob"]
        ratio = jnp.exp(log_ratio)
        advantages = batch["advantage"]

        # clipped surrogate objective (negated: we minimize)
        pg = -jnp.minimum(ratio * advantages, jnp.clip(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef) * advantages)
        value_loss = 0.5 * jnp.square(value - batch["return"])
        entropy = -jnp.sum(jnp.exp(log_probs) * log_probs, axis=-1)

        loss = jnp.mean(pg + cfg.vf_coef * value_loss - cfg.ent_coef * entropy)
        metrics = {
            "entropy": jnp.mean(entropy),
            "approx_kl": jnp.mean((ratio - 1.0) - log_ratio),  # Schulman's estimator
            "clipfrac": jnp.mean((jnp.abs(ratio - 1.0) > cfg.clip_coef).astype(jnp.float32)),
            "value_loss": jnp.mean(value_loss),
        }
        return loss, metrics

    return loss_fn


def make_epoch(loss_fn, tx, cfg: PPOConfig) -> Callable:
    """One pass over the shuffled batch, as a ``lax.scan`` body."""
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    def minibatch_update(carry, minibatch):
        params, opt_state = carry
        (_, metrics), grads = grad_fn(params, minibatch)
        updates, opt_state = tx.update(grads, opt_state, params)
        return (optax.apply_updates(params, updates), opt_state), metrics

    def epoch(carry, _):
        params, opt_state, key, batch = carry
        key, perm_key = jax.random.split(key)
        perm = jax.random.permutation(perm_key, cfg.batch_size)
        minibatches = jax.tree.map(
            lambda x: x[perm].reshape((cfg.num_minibatches, cfg.minibatch_size) + x.shape[1:]), batch
        )
        (params, opt_state), metrics = jax.lax.scan(minibatch_update, (params, opt_state), minibatches)
        return (params, opt_state, key, batch), metrics

    return epoch


def set_learning_rate(opt_state, lr):
    """Push the annealed learning rate into the injected Adam hyperparameters.

    The ``optax.inject_hyperparams`` state is found by looking for the member
    that carries a ``learning_rate`` hyperparameter rather than by position, so
    reordering or extending the chain in :class:`PPO` cannot silently stop the
    annealing.  This runs under ``jit``, so the check costs one trace.
    """
    updated, found = [], False
    for state in opt_state:
        hyperparams = getattr(state, "hyperparams", None)
        if isinstance(hyperparams, dict) and "learning_rate" in hyperparams:
            state = state._replace(hyperparams={**hyperparams, "learning_rate": lr})
            found = True
        updated.append(state)
    if not found:
        raise ValueError("no optax.inject_hyperparams state carrying 'learning_rate' in the optimizer chain")
    return tuple(updated)


# ---------------------------------------------------------------------------
# trainer
# ---------------------------------------------------------------------------


class PPO(Trainer):
    def __init__(self, cfg: PPOConfig, *, device=None):
        self.cfg = cfg
        self.device = resolve_device(device)
        self.env_device = self.device  # make_env hands this to the vector env
        self.vec_env = self.make_env(cfg.num_envs)
        self.env = self.vec_env.env  # the functional single-env core

        self.agent = ActorCritic(
            self.vec_env.obs_dim, self.vec_env.num_actions, hidden=cfg.hidden, seed=cfg.seed, device=self.device
        )
        self.tx = optax.chain(
            optax.clip_by_global_norm(cfg.max_grad_norm),
            optax.inject_hyperparams(optax.adam)(learning_rate=cfg.learning_rate, eps=1e-8),
        )
        key, env_key = jax.random.split(jax.device_put(jax.random.PRNGKey(cfg.seed), self.device))
        # The whole runner state lives on self.device.  Only some of it gets
        # there on its own: leaves derived from a committed input inherit its
        # placement, but the ones built fresh and eagerly -- optax's step count
        # and injected learning rate, vec_reset's zeroed episode counters --
        # would land on JAX's default device instead, which is not ours
        # whenever --device asks for anything but the first one.
        self.runner = jax.device_put(
            RunnerState(
                params=self.agent.params,
                opt_state=self.tx.init(self.agent.params),
                env_state=vec_reset(self.env, env_key, cfg.num_envs),
                key=key,
            ),
            self.device,
        )
        self._iteration = jax.jit(self._build_iteration())

    def _build_iteration(self) -> Callable:
        cfg = self.cfg
        apply_fn = self.agent.net.apply
        rollout_step = make_rollout_step(self.env, apply_fn, cfg)
        epoch = make_epoch(make_loss(apply_fn, cfg), self.tx, cfg)

        def iteration(runner: RunnerState, lr):
            params, opt_state, env_state, key = runner
            opt_state = set_learning_rate(opt_state, lr)

            (params, env_state, key), traj = jax.lax.scan(
                rollout_step, (params, env_state, key), None, length=cfg.num_steps
            )

            # value of the pre-auto-reset next observation, for bootstrapping
            _, boot_values = apply_fn(params, traj["next_obs"])
            advantages, returns = compute_gae(
                traj["reward"],
                traj["value"],
                boot_values,
                traj["terminated"],
                traj["truncated"],
                cfg.gamma,
                cfg.gae_lambda,
            )
            if cfg.norm_adv:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            batch = {
                "obs": traj["obs"].reshape(cfg.batch_size, -1),
                "action": traj["action"].reshape(cfg.batch_size),
                "log_prob": traj["log_prob"].reshape(cfg.batch_size),
                "advantage": advantages.reshape(cfg.batch_size),
                "return": returns.reshape(cfg.batch_size),
            }
            (params, opt_state, key, _), metrics = jax.lax.scan(
                epoch, (params, opt_state, key, batch), None, length=cfg.update_epochs
            )

            stats = {name: value.mean() for name, value in metrics.items()}
            stats.update(
                episodes=traj["done"].sum(),
                return_sum=traj["done_return"].sum(),
                length_sum=traj["done_length"].sum(),
            )
            return RunnerState(params, opt_state, env_state, key), stats

        return iteration

    # -- training loop ------------------------------------------------------

    def iterate(self, lr: float) -> dict:
        """One rollout plus one update; returns this iteration's metrics."""
        self.runner, stats = self._iteration(self.runner, jnp.float32(lr))
        stats = {name: float(value) for name, value in stats.items()}
        episodes = int(stats.pop("episodes"))
        return_sum, length_sum = stats.pop("return_sum"), stats.pop("length_sum")
        return {
            **stats,
            "episodes": episodes,
            "episodic_return": return_sum / episodes if episodes else float("nan"),
            "episodic_length": length_sum / episodes if episodes else float("nan"),
        }

    def train(self, **kwargs) -> list[dict]:
        history = super().train(**kwargs)
        self.agent.params = self.runner.params  # keep the agent facade current
        return history

    # -- evaluation ---------------------------------------------------------

    def evaluate(self, **kwargs) -> dict:
        self.agent.params = self.runner.params  # evaluate what training produced
        return super().evaluate(**kwargs)
