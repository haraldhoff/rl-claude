"""Stable-Baselines3's PPO as a third backend.

The environment is ours (Warp or JAX, wrapped as an SB3 ``VecEnv``) and the
learner is SB3's reference PPO implementation, driven one iteration at a time by
the same shared training loop as the other two backends -- so the per-iteration
logging, the learning-rate annealing and the greedy evaluation are identical and
the only thing that changes is who computes the update.

``env_backend="gym"`` swaps in stock Gymnasium environments (``CartPole-v1``,
``MountainCar-v0``, ``LunarLander-v3``) instead, which gives a reference point:
same learner, upstream physics.

Two caveats when comparing runs. SB3 logs ``value_loss`` as the plain MSE and
``entropy_loss`` as the negated mean entropy, while our two backends log
``0.5 * MSE`` and the entropy itself, so the value-loss column is twice ours.
And SB3's Adam epsilon defaults to 1e-5 rather than 1e-8, which is not a detail
on sparse-reward tasks -- see :func:`sb3_rl.agent.policy_kwargs`.
"""

from __future__ import annotations

import numpy as np
import stable_baselines3 as sb3
from stable_baselines3.common.callbacks import BaseCallback

from rl_common import PPOConfig, Trainer, spec

from .agent import ActorCritic, policy_kwargs, resolve_device
from .vec_env import VecEnvAdapter

GYM_ENV_IDS = {"cartpole": "CartPole-v1", "lunarlander": "LunarLander-v3", "mountaincar": "MountainCar-v0"}


class ActionRepeat:
    """Hold each action for ``repeat`` frames of a Gymnasium environment.

    Only used by the ``env_backend="gym"`` baseline, so that stock
    ``MountainCar-v0`` sees the same decision rate our environment trains at.
    """

    def __init__(self, repeat: int):
        self.repeat = int(repeat)

    def __call__(self, env):
        import gymnasium as gym

        repeat = self.repeat

        class _Wrapper(gym.Wrapper):
            def step(self, action):
                total = 0.0
                for _ in range(repeat):
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    total += float(reward)
                    if terminated or truncated:
                        break
                return obs, total, terminated, truncated, info

        return _Wrapper(env)


class _EpisodeCollector(BaseCallback):
    """Collect the ``info["episode"]`` entries SB3 emits, per iteration."""

    def __init__(self):
        super().__init__()
        self.returns: list[float] = []
        self.lengths: list[float] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            episode = info.get("episode")
            if episode is not None:
                self.returns.append(float(episode["r"]))
                self.lengths.append(float(episode["l"]))
        return True

    def pop(self) -> tuple[float, float, int]:
        count = len(self.returns)
        if count == 0:
            return float("nan"), float("nan"), 0
        stats = (float(np.mean(self.returns)), float(np.mean(self.lengths)), count)
        self.returns.clear()
        self.lengths.clear()
        return stats


class PPO(Trainer):
    def __init__(self, cfg: PPOConfig, *, device=None):
        self.cfg = cfg
        self.env_backend = cfg.resolved_env_backend
        self.device = resolve_device(device)

        self.env_device = device
        if self.env_backend == "gym":
            self.env = None
            self.venv = self._make_gym_vec_env()
        else:
            self.env = self.make_env(cfg.num_envs)
            self.venv = VecEnvAdapter(self.env)

        self.model = sb3.PPO(
            "MlpPolicy",
            self.venv,
            n_steps=cfg.num_steps,
            batch_size=cfg.minibatch_size,
            n_epochs=cfg.update_epochs,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            clip_range=cfg.clip_coef,
            ent_coef=cfg.ent_coef,
            vf_coef=cfg.vf_coef,
            max_grad_norm=cfg.max_grad_norm,
            normalize_advantage=cfg.norm_adv,
            learning_rate=cfg.learning_rate,
            policy_kwargs=policy_kwargs(cfg.hidden),
            seed=cfg.seed,
            device=self.device,
            verbose=0,
        )
        self.agent = ActorCritic(
            spec(cfg.env_id).obs_dim, spec(cfg.env_id).num_actions, hidden=cfg.hidden, device=device
        )
        self.agent.policy = self.model.policy  # the facade views the live policy
        self._episodes = _EpisodeCollector()
        self.global_step = 0

    def _make_gym_vec_env(self):
        """Stock Gymnasium environments, for a same-learner reference run."""
        from stable_baselines3.common.env_util import make_vec_env

        cfg = self.cfg
        repeat = int(cfg.env_kwargs.get("action_repeat", 1))
        wrappers = ActionRepeat(repeat) if repeat > 1 else None
        return make_vec_env(
            GYM_ENV_IDS[cfg.env_id],
            n_envs=cfg.num_envs,
            seed=cfg.seed,
            wrapper_class=wrappers,
            env_kwargs=dict(max_episode_steps=cfg.max_episode_steps * repeat),
        )

    # -- training loop ------------------------------------------------------

    def iterate(self, lr: float) -> dict:
        """One SB3 rollout plus one SB3 update, driven by the shared loop."""
        # the shared loop owns the annealing; hand the value to SB3 per iteration
        self.model.lr_schedule = lambda _progress, value=lr: value
        self.model.learn(
            total_timesteps=self.cfg.batch_size,
            reset_num_timesteps=False,
            callback=self._episodes,
            progress_bar=False,
        )

        logged = self.model.logger.name_to_value
        ep_return, ep_length, episodes = self._episodes.pop()
        return {
            "entropy": -float(logged.get("train/entropy_loss", np.nan)),
            "approx_kl": float(logged.get("train/approx_kl", np.nan)),
            "clipfrac": float(logged.get("train/clip_fraction", np.nan)),
            "value_loss": float(logged.get("train/value_loss", np.nan)),
            "episodic_return": ep_return,
            "episodic_length": ep_length,
            "episodes": episodes,
        }

    # -- evaluation ---------------------------------------------------------

    def evaluate(self, *, num_envs: int = 64, max_episode_steps: int | None = None, seed: int = 12345) -> dict:
        """Greedy evaluation, in the environment family this model trained on."""
        if self.env_backend == "gym":
            return self._evaluate_gym(num_envs, max_episode_steps or self.cfg.max_episode_steps, seed)
        return super().evaluate(num_envs=num_envs, max_episode_steps=max_episode_steps, seed=seed)

    def _evaluate_gym(self, episodes: int, limit: int, seed: int) -> dict:
        import gymnasium as gym

        cfg = self.cfg
        repeat = int(cfg.env_kwargs.get("action_repeat", 1))
        env = gym.make(GYM_ENV_IDS[cfg.env_id], max_episode_steps=limit * repeat)
        returns, lengths = [], []
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            total, steps, done, action = 0.0, 0, False, 0
            while not done:
                if steps % repeat == 0:
                    action = int(self.agent.act_numpy(np.asarray(obs, dtype=np.float32))[0])
                obs, reward, terminated, truncated, _ = env.step(action)
                total += float(reward)
                steps += 1
                done = terminated or truncated
            returns.append(total)
            lengths.append(steps)
        env.close()
        returns = np.asarray(returns)
        return {
            "mean_return": float(returns.mean()),
            "std_return": float(returns.std()),
            "min_return": float(returns.min()),
            "max_return": float(returns.max()),
            "mean_length": float(np.mean(lengths)),
            "num_episodes": int(episodes),
        }

