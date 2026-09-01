"""PPO on top of any :class:`~warp_rl.vec_env.WarpVecEnv`.

Every per-sample operation -- action sampling, GAE, advantage normalization,
minibatch gathering and the clipped surrogate loss -- is a Warp kernel, so a
training iteration runs on the device with no host round-trips except the tiny
logging reads.  Gradients come from ``wp.Tape``; the networks and the Adam
optimizer come from warp-nn.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any, Callable

import numpy as np
import warp as wp
from warp_nn.optimizers import Adam

from . import kernels as K
from .models import ActorCritic
from .vec_env import seed_kernel


@dataclasses.dataclass
class PPOConfig:
    env_id: str = "cartpole"
    env_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
    num_envs: int = 256
    num_steps: int = 32
    total_timesteps: int = 500_000
    learning_rate: float = 1e-3
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 8
    update_epochs: int = 10
    clip_coef: float = 0.2
    ent_coef: float = 0.005
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    norm_adv: bool = True
    hidden: tuple[int, ...] = (64, 64)
    max_episode_steps: int = 500
    seed: int = 0
    use_graph: bool = True  # capture the rollout / update epochs as CUDA graphs

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.num_steps

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches

    @property
    def num_iterations(self) -> int:
        return max(1, self.total_timesteps // self.batch_size)


class PPO:
    def __init__(self, cfg: PPOConfig, *, device: str | wp.Device | None = None):
        from .registry import make  # local import: the registry imports the envs

        self.cfg = cfg
        self.device = wp.get_device(device)
        d = self.device

        self.env = make(
            cfg.env_id,
            cfg.num_envs,
            max_episode_steps=cfg.max_episode_steps,
            device=d,
            seed=cfg.seed,
            **cfg.env_kwargs,
        )
        self.agent = ActorCritic(
            self.env.obs_dim,
            self.env.num_actions,
            hidden=cfg.hidden,
            seed=cfg.seed,
            device=d,
        )
        self.optimizer = Adam(
            self.agent.parameters(),
            lr=cfg.learning_rate,
            max_norm=cfg.max_grad_norm,
            device=d,
            # we capture whole epochs ourselves; Adam must not nest a capture
            disable_graph=cfg.use_graph,
        )

        self.sample_actions, self.greedy_actions = K.make_action_kernels(self.env.num_actions)
        self.ppo_loss, self.ppo_metrics = K.make_loss_kernels(self.env.num_actions)

        T, N, B, M = cfg.num_steps, cfg.num_envs, cfg.batch_size, cfg.minibatch_size
        O = self.env.obs_dim
        self.obs_dim = O

        # rollout buffers (batch-major for the network, time-major for GAE)
        self.obs_buf = wp.zeros((B, O), dtype=wp.float32, device=d)
        self.next_obs_buf = wp.zeros((B, O), dtype=wp.float32, device=d)
        self.act_buf = wp.zeros(B, dtype=wp.int32, device=d)
        self.logp_buf = wp.zeros(B, dtype=wp.float32, device=d)
        self.adv_buf = wp.zeros(B, dtype=wp.float32, device=d)
        self.ret_buf = wp.zeros(B, dtype=wp.float32, device=d)

        self.rew_2d = wp.zeros((T, N), dtype=wp.float32, device=d)
        self.term_2d = wp.zeros((T, N), dtype=wp.float32, device=d)
        self.trunc_2d = wp.zeros((T, N), dtype=wp.float32, device=d)
        self.val_2d = wp.zeros((T, N), dtype=wp.float32, device=d)
        self.boot_2d = wp.zeros((T, N), dtype=wp.float32, device=d)
        self.adv_2d = wp.zeros((T, N), dtype=wp.float32, device=d)
        self.ret_2d = wp.zeros((T, N), dtype=wp.float32, device=d)

        # per-step scratch shared by the rollout kernels
        self.env_actions = wp.zeros(N, dtype=wp.int32, device=d)
        self.env_logps = wp.zeros(N, dtype=wp.float32, device=d)

        # minibatch scratch
        self.mb_obs = wp.zeros((M, O), dtype=wp.float32, device=d)
        self.mb_act = wp.zeros(M, dtype=wp.int32, device=d)
        self.mb_logp = wp.zeros(M, dtype=wp.float32, device=d)
        self.mb_adv = wp.zeros(M, dtype=wp.float32, device=d)
        self.mb_ret = wp.zeros(M, dtype=wp.float32, device=d)
        self.indices = wp.zeros(B, dtype=wp.int32, device=d)

        self.loss = wp.zeros(1, dtype=wp.float32, device=d, requires_grad=True)
        self.moments = wp.zeros(2, dtype=wp.float32, device=d)
        self.metrics = wp.zeros(4, dtype=wp.float32, device=d)

        # per-env RNG for action sampling (independent of the env's own RNG)
        self.rng_states = wp.zeros(N, dtype=wp.uint32, device=d)
        wp.launch(seed_kernel, dim=N, inputs=[cfg.seed + 12345, self.rng_states], device=d)

        self._np_rng = np.random.default_rng(cfg.seed)
        self._graph = None
        self._warm = False
        self._update_graph = None
        self._update_warm = False
        self._grad_views = None
        self._loss_view = self.loss.flatten()
        self.global_step = 0

    # -- rollout ------------------------------------------------------------

    def _rollout_eager(self) -> None:
        cfg = self.cfg
        N, O = cfg.num_envs, self.obs_dim
        for t in range(cfg.num_steps):
            offset = t * N
            wp.launch(K.store_obs, dim=(N, O), inputs=[self.env.obs, offset, self.obs_buf], device=self.device)

            logits = self.agent.policy(self.env.obs)
            wp.launch(
                self.sample_actions,
                dim=N,
                inputs=[logits, self.rng_states, self.env_actions, self.env_logps],
                device=self.device,
            )
            self.env.step(self.env_actions)

            wp.launch(K.store_i32, dim=N, inputs=[self.env_actions, offset, self.act_buf], device=self.device)
            wp.launch(K.store_f32, dim=N, inputs=[self.env_logps, offset, self.logp_buf], device=self.device)
            wp.launch(
                K.store_obs, dim=(N, O), inputs=[self.env.final_obs, offset, self.next_obs_buf], device=self.device
            )
            wp.copy(self.rew_2d[t], self.env.rewards)
            wp.copy(self.term_2d[t], self.env.terminated)
            wp.copy(self.trunc_2d[t], self.env.truncated)

    def rollout(self) -> None:
        if self._graph is not None:
            wp.capture_launch(self._graph)
            return
        if self.cfg.use_graph and self.device.is_cuda and self._warm:
            with wp.ScopedCapture(device=self.device) as capture:
                self._rollout_eager()
            self._graph = capture.graph
            wp.capture_launch(self._graph)
            return
        self._rollout_eager()
        self._warm = True

    # -- advantages ---------------------------------------------------------

    def compute_advantages(self) -> None:
        cfg = self.cfg
        T, N = cfg.num_steps, cfg.num_envs

        # NOTE: warp-nn caches one output array per (shape, dtype), so the two
        # value passes below share a buffer -- copy out before the second call.
        values = self.agent.value(self.obs_buf)
        wp.launch(K.column_to_2d, dim=(T, N), inputs=[values, self.val_2d], device=self.device)
        boot = self.agent.value(self.next_obs_buf)
        wp.launch(K.column_to_2d, dim=(T, N), inputs=[boot, self.boot_2d], device=self.device)

        wp.launch(
            K.gae_kernel,
            dim=N,
            inputs=[
                self.rew_2d,
                self.val_2d,
                self.boot_2d,
                self.term_2d,
                self.trunc_2d,
                cfg.gamma,
                cfg.gae_lambda,
                self.adv_2d,
                self.ret_2d,
            ],
            device=self.device,
        )
        wp.launch(K.flatten_2d, dim=(T, N), inputs=[self.adv_2d, self.adv_buf], device=self.device)
        wp.launch(K.flatten_2d, dim=(T, N), inputs=[self.ret_2d, self.ret_buf], device=self.device)

        if cfg.norm_adv:
            self.moments.zero_()
            wp.launch(K.sum_and_sumsq, dim=cfg.batch_size, inputs=[self.adv_buf, self.moments], device=self.device)
            wp.launch(
                K.normalize,
                dim=cfg.batch_size,
                inputs=[self.adv_buf, self.moments, float(cfg.batch_size), 1e-8],
                device=self.device,
            )

    # -- update -------------------------------------------------------------

    def _minibatch_step(self, start: int) -> None:
        cfg = self.cfg
        M, O = cfg.minibatch_size, self.obs_dim
        inv_m = 1.0 / float(M)
        num_updates = cfg.update_epochs * cfg.num_minibatches

        wp.launch(K.gather_obs, dim=(M, O), inputs=[self.obs_buf, self.indices, start, self.mb_obs], device=self.device)
        wp.launch(K.gather_i32, dim=M, inputs=[self.act_buf, self.indices, start, self.mb_act], device=self.device)
        wp.launch(K.gather_f32, dim=M, inputs=[self.logp_buf, self.indices, start, self.mb_logp], device=self.device)
        wp.launch(K.gather_f32, dim=M, inputs=[self.adv_buf, self.indices, start, self.mb_adv], device=self.device)
        wp.launch(K.gather_f32, dim=M, inputs=[self.ret_buf, self.indices, start, self.mb_ret], device=self.device)

        self._zero(self._loss_view)
        tape = wp.Tape()
        with tape:
            logits = self.agent.policy(self.mb_obs)
            values = self.agent.value(self.mb_obs)
            wp.launch(
                self.ppo_loss,
                dim=M,
                inputs=[
                    logits,
                    values,
                    self.mb_act,
                    self.mb_logp,
                    self.mb_adv,
                    self.mb_ret,
                    cfg.clip_coef,
                    cfg.vf_coef,
                    cfg.ent_coef,
                    inv_m,
                    self.loss,
                ],
                device=self.device,
            )
        tape.backward(loss=self.loss)
        wp.launch(
            self.ppo_metrics,
            dim=M,
            inputs=[
                logits,
                values,
                self.mb_act,
                self.mb_logp,
                self.mb_ret,
                cfg.clip_coef,
                inv_m / float(num_updates),
                self.metrics,
            ],
            device=self.device,
        )
        # the learning rate lives in a device array, so a captured step still
        # picks up the annealed value set by _set_lr()
        self.optimizer.step()

        if self._grad_views is None:
            # the arrays are stable across minibatches (warp-nn caches its
            # activations per shape), so resolve the gradient set once
            self._grad_views = [g.flatten() for g in tape.gradients.values() if isinstance(g, wp.array)]
        for view in self._grad_views:
            self._zero(view)

    def _zero(self, view: wp.array) -> None:
        wp.launch(K.zero_kernel, dim=view.shape[0], inputs=[view], device=self.device)

    def _epoch_eager(self) -> None:
        for start in range(0, self.cfg.batch_size, self.cfg.minibatch_size):
            self._minibatch_step(start)

    def _run_epoch(self) -> None:
        if self._update_graph is not None:
            wp.capture_launch(self._update_graph)
            return
        if self.cfg.use_graph and self.device.is_cuda and self._update_warm:
            with wp.ScopedCapture(device=self.device) as capture:
                self._epoch_eager()
            self._update_graph = capture.graph
            wp.capture_launch(self._update_graph)
            return
        self._epoch_eager()
        self._update_warm = True

    def _set_lr(self, lr: float) -> None:
        self.optimizer._lr.fill_(lr)  # noqa: SLF001 - warp-nn has no public setter

    def update(self, lr: float) -> dict[str, float]:
        cfg = self.cfg
        self.metrics.zero_()
        self._set_lr(lr)

        for _ in range(cfg.update_epochs):
            # a fresh shuffle per epoch; the graph reads the same index array
            self.indices.assign(self._np_rng.permutation(cfg.batch_size).astype(np.int32))
            self._run_epoch()

        entropy, approx_kl, clipfrac, v_loss = (float(v) for v in self.metrics.numpy())
        return {"entropy": entropy, "approx_kl": approx_kl, "clipfrac": clipfrac, "value_loss": v_loss}

    # -- training loop ------------------------------------------------------

    def train(self, *, log_every: int = 1, callback: Callable[[dict], None] | None = None) -> None:
        cfg = self.cfg
        self.env.reset(seed=cfg.seed)

        start = time.time()
        for it in range(1, cfg.num_iterations + 1):
            frac = 1.0 - (it - 1) / cfg.num_iterations
            lr = cfg.learning_rate * frac if cfg.anneal_lr else cfg.learning_rate

            self.rollout()
            self.global_step += cfg.batch_size
            self.compute_advantages()
            stats = self.update(lr)

            ep_return, ep_length, ep_count = self.env.pop_episode_stats()
            elapsed = time.time() - start
            stats.update(
                iteration=it,
                global_step=self.global_step,
                episodic_return=ep_return,
                episodic_length=ep_length,
                episodes=ep_count,
                lr=lr,
                sps=self.global_step / max(elapsed, 1e-9),
                elapsed=elapsed,
            )
            if callback is not None:
                callback(stats)
            elif log_every and (it % log_every == 0 or it == cfg.num_iterations):
                print(
                    f"iter {it:4d}/{cfg.num_iterations}  step {self.global_step:>9,}  "
                    f"return {ep_return:8.1f}  len {ep_length:6.1f}  "
                    f"entropy {stats['entropy']:.3f}  kl {stats['approx_kl']:.4f}  "
                    f"clipfrac {stats['clipfrac']:.3f}  v_loss {stats['value_loss']:8.2f}  "
                    f"{stats['sps']:,.0f} steps/s"
                )

    # -- evaluation ---------------------------------------------------------

    def evaluate(self, *, num_envs: int = 64, max_episode_steps: int | None = None, seed: int = 12345) -> dict:
        """Run the greedy (argmax) policy until every env finishes one episode."""
        from .registry import make

        cfg = self.cfg
        limit = max_episode_steps or cfg.max_episode_steps
        env = make(
            cfg.env_id,
            num_envs,
            max_episode_steps=limit,
            autoreset=False,
            device=self.device,
            seed=seed,
            **cfg.env_kwargs,
        )
        obs, _ = env.reset()
        actions = wp.zeros(num_envs, dtype=wp.int32, device=self.device)
        alive = np.ones(num_envs, dtype=bool)
        returns = np.zeros(num_envs, dtype=np.float32)
        lengths = np.zeros(num_envs, dtype=np.int64)

        for _ in range(limit):
            logits = self.agent.policy(obs)
            wp.launch(self.greedy_actions, dim=num_envs, inputs=[logits, actions], device=self.device)
            obs, reward, terminated, truncated, _ = env.step(actions)
            done = (terminated.numpy() + truncated.numpy()) > 0
            returns += alive * reward.numpy()
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
            "num_episodes": int(num_envs),
        }
