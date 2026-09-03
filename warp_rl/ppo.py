"""PPO on top of any :class:`~warp_rl.vec_env.WarpVecEnv`.

Every per-sample operation -- action sampling, GAE, advantage normalization,
minibatch gathering and the clipped surrogate loss -- is a Warp kernel, so a
training iteration runs on the device with no host round-trips except the tiny
logging reads.  Gradients come from ``wp.Tape``; the networks and the Adam
optimizer come from warp-nn.
"""

from __future__ import annotations

import numpy as np
import warp as wp
from warp_nn.optimizers import Adam

from rl_common import PPOConfig, Trainer

from . import kernels as K
from .agent import ActorCritic
from .vec_env import seed_kernel


class PPO(Trainer):
    def __init__(self, cfg: PPOConfig, *, device: str | wp.Device | None = None):
        self.cfg = cfg
        self.device = wp.get_device(device)
        d = self.device

        self.env_device = d
        self.env = self.make_env(cfg.num_envs)
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

        self.sample_actions, _ = K.make_action_kernels(self.env.num_actions)  # the trainer only samples
        self.ppo_loss, self.ppo_metrics = K.make_loss_kernels(self.env.num_actions)

        T, N, B, M = cfg.num_steps, cfg.num_envs, cfg.batch_size, cfg.minibatch_size
        O = self.env.obs_dim
        self.obs_dim = O

        def zeros(shape, dtype=wp.float32, requires_grad=False):
            return wp.zeros(shape, dtype=dtype, device=d, requires_grad=requires_grad)

        # rollout buffers: batch-major for the network...
        self.obs_buf = zeros((B, O))
        self.next_obs_buf = zeros((B, O))
        self.act_buf = zeros(B, wp.int32)
        self.logp_buf = zeros(B)
        self.adv_buf = zeros(B)
        self.ret_buf = zeros(B)

        # ... and time-major for the GAE recursion
        self.rew_2d = zeros((T, N))
        self.term_2d = zeros((T, N))
        self.trunc_2d = zeros((T, N))
        self.val_2d = zeros((T, N))
        self.boot_2d = zeros((T, N))
        self.adv_2d = zeros((T, N))
        self.ret_2d = zeros((T, N))

        # per-step scratch shared by the rollout kernels
        self.env_actions = zeros(N, wp.int32)
        self.env_logps = zeros(N)

        # minibatch scratch
        self.mb_obs = zeros((M, O))
        self.mb_act = zeros(M, wp.int32)
        self.mb_logp = zeros(M)
        self.mb_adv = zeros(M)
        self.mb_ret = zeros(M)
        self.indices = zeros(B, wp.int32)

        self.loss = zeros(1, requires_grad=True)
        self.moments = zeros(2)
        self.metrics = zeros(4)

        # per-env RNG for action sampling (independent of the env's own RNG)
        self.rng_states = zeros(N, wp.uint32)
        wp.launch(seed_kernel, dim=N, inputs=[cfg.seed + 12345, self.rng_states], device=d)

        self._np_rng = np.random.default_rng(cfg.seed)
        self._graph = None
        self._warm = False
        self._update_graph = None
        self._update_warm = False
        self._grad_views = None
        self._loss_view = self.loss.flatten()

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

    # -- one iteration ------------------------------------------------------

    def iterate(self, lr: float) -> dict:
        """One rollout plus one update; returns this iteration's metrics."""
        self.rollout()
        self.compute_advantages()
        stats = self.update(lr)
        ep_return, ep_length, ep_count = self.env.pop_episode_stats()
        stats.update(episodic_return=ep_return, episodic_length=ep_length, episodes=ep_count)
        return stats

    def train(self, **kwargs) -> list[dict]:
        self.env.reset(seed=self.cfg.seed)
        return super().train(**kwargs)
