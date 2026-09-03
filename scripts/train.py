"""Train PPO on one of the three backends.

    python scripts/train.py                                  # cartpole, warp backend
    python scripts/train.py --backend jax --env cartpole     # the same run in JAX + Flax
    python scripts/train.py --backend sb3 --env cartpole     # ... or SB3's reference PPO
    python scripts/train.py --backend sb3 --env-backend gym  # ... SB3 on stock Gymnasium
    python scripts/train.py --env lunarlander --gym-eval 10  # cross-check in real Gymnasium
"""

from __future__ import annotations

import argparse
import os
import sys

# the repo root, so `python scripts/train.py` works from a bare checkout and
# not only from an installed package -- same bootstrap as tools/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from rl_common import cli, make_trainer, spec

GYM_ENV_IDS = {"cartpole": "CartPole-v1", "lunarlander": "LunarLander-v3", "mountaincar": "MountainCar-v0"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_arguments(parser)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--eval-envs", type=int, default=128)
    parser.add_argument("--gym-eval", type=int, default=0, metavar="N", help="also run N episodes in Gymnasium")
    parser.add_argument("--save", type=str, default=None, help="path to save the trained weights")
    return parser.parse_args()


def gym_eval(agent, env_id: str, episodes: int, max_episode_steps: int, seed: int, repeat: int = 1) -> None:
    """Cross-check: run the trained policy inside the real Gymnasium environment."""
    import gymnasium as gym

    gym_id = GYM_ENV_IDS[env_id]
    try:
        env = gym.make(gym_id, max_episode_steps=max_episode_steps)
    except Exception as exc:  # e.g. LunarLander without box2d installed
        print(f"skipping Gymnasium cross-check ({gym_id}): {exc}")
        return

    returns = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        total, done, step, action = 0.0, False, 0, 0
        while not done:
            if step % repeat == 0:  # match the agent's decision rate
                action = int(agent.act_numpy(obs)[0])
            obs, reward, terminated, truncated, _ = env.step(action)
            step += 1
            total += float(reward)
            done = terminated or truncated
        returns.append(total)
    env.close()
    print(f"gymnasium {gym_id} ({episodes} episodes): mean return {np.mean(returns):.1f} +/- {np.std(returns):.1f}")


def main() -> None:
    args = parse_args()
    cfg = cli.config_from_args(args)

    trainer = make_trainer(cfg, device=args.device)
    print(
        f"{cfg.env_id} on {cfg.backend} learner + {cfg.resolved_env_backend} env ({trainer.device}) | "
        f"{cfg.num_envs} envs x {cfg.num_steps} steps = {cfg.batch_size} batch | "
        f"{cfg.num_iterations} iterations | minibatch {cfg.minibatch_size}"
    )
    trainer.train(log_every=args.log_every)

    result = trainer.evaluate(num_envs=args.eval_envs)
    verdict = "  <- solved" if result["mean_return"] >= spec(cfg.env_id).solved_return else ""
    print(
        f"greedy evaluation ({result['num_episodes']} episodes): "
        f"mean return {result['mean_return']:.1f} +/- {result['std_return']:.1f} "
        f"(min {result['min_return']:.0f}, max {result['max_return']:.0f}, "
        f"mean length {result['mean_length']:.0f}){verdict}"
    )

    if args.save:
        trainer.agent.save(args.save)
        print(f"saved weights to {args.save}")

    if args.gym_eval:
        repeat = int(cfg.env_kwargs.get("action_repeat", 1))
        gym_eval(trainer.agent, cfg.env_id, args.gym_eval, cfg.max_episode_steps * repeat, args.seed, repeat=repeat)


if __name__ == "__main__":
    main()
