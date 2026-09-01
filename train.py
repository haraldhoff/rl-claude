"""Train PPO on a Warp environment.

    python train.py                          # cartpole with the recommended settings
    python train.py --env lunarlander        # ... or the lander
    python train.py --env cartpole --gym-eval 10   # cross-check in real Gymnasium
    python train.py --num-envs 1024 --total-timesteps 2000000
"""

from __future__ import annotations

import argparse

import numpy as np
import warp as wp

from warp_rl import PPO, default_config, env_ids, spec


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", type=str, default="cartpole", help=f"environment id: {', '.join(env_ids())}")
    # every override defaults to None so the environment's recommended value wins
    p.add_argument("--num-envs", type=int, default=None)
    p.add_argument("--num-steps", type=int, default=None)
    p.add_argument("--total-timesteps", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--gae-lambda", type=float, default=None)
    p.add_argument("--num-minibatches", type=int, default=None)
    p.add_argument("--update-epochs", type=int, default=None)
    p.add_argument("--clip-coef", type=float, default=None)
    p.add_argument("--ent-coef", type=float, default=None)
    p.add_argument("--vf-coef", type=float, default=None)
    p.add_argument("--max-grad-norm", type=float, default=None)
    p.add_argument("--max-episode-steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None, help="warp device, e.g. cuda:0 or cpu")
    p.add_argument("--no-anneal-lr", action="store_true")
    p.add_argument("--no-graph", action="store_true", help="disable CUDA graph capture")
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--eval-envs", type=int, default=128)
    p.add_argument("--gym-eval", type=int, default=0, metavar="N", help="also run N episodes in Gymnasium")
    p.add_argument("--save", type=str, default=None, help="path to save the trained weights (.npz)")
    return p.parse_args()


GYM_ENV_IDS = {"cartpole": "CartPole-v1", "lunarlander": "LunarLander-v3", "mountaincar": "MountainCar-v0"}


def gym_eval(agent, env_id: str, episodes: int, max_episode_steps: int, seed: int, repeat: int = 1) -> None:
    """Cross-check: run the Warp-trained policy inside the real Gymnasium env."""
    import gymnasium as gym

    from warp_rl.kernels import make_action_kernels

    gym_id = GYM_ENV_IDS[env_id]
    _, greedy = make_action_kernels(agent.num_actions)
    try:
        env = gym.make(gym_id, max_episode_steps=max_episode_steps)
    except Exception as exc:  # e.g. LunarLander without box2d installed
        print(f"skipping Gymnasium cross-check ({gym_id}): {exc}")
        return
    obs_arr = wp.zeros((1, agent.obs_dim), dtype=wp.float32, device=agent.device)
    act_arr = wp.zeros(1, dtype=wp.int32, device=agent.device)

    returns = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        total, done, step = 0.0, False, 0
        while not done:
            if step % repeat == 0:  # match the agent's decision rate
                obs_arr.assign(np.asarray(obs, dtype=np.float32).reshape(1, -1))
                logits = agent.policy(obs_arr)
                wp.launch(greedy, dim=1, inputs=[logits, act_arr], device=agent.device)
            obs, reward, terminated, truncated, _ = env.step(int(act_arr.numpy()[0]))
            step += 1
            total += float(reward)
            done = terminated or truncated
        returns.append(total)
    env.close()
    print(f"gymnasium {gym_id} ({episodes} episodes): mean return {np.mean(returns):.1f} +/- {np.std(returns):.1f}")


def main() -> None:
    args = parse_args()
    wp.init()

    cfg = default_config(
        args.env,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        anneal_lr=False if args.no_anneal_lr else None,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        num_minibatches=args.num_minibatches,
        update_epochs=args.update_epochs,
        clip_coef=args.clip_coef,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        max_episode_steps=args.max_episode_steps,
        seed=args.seed,
        use_graph=False if args.no_graph else None,
    )

    trainer = PPO(cfg, device=args.device)
    print(
        f"{cfg.env_id} on {trainer.device} | {cfg.num_envs} envs x {cfg.num_steps} steps = {cfg.batch_size} batch "
        f"| {cfg.num_iterations} iterations | minibatch {cfg.minibatch_size}"
    )
    trainer.train(log_every=args.log_every)

    result = trainer.evaluate(num_envs=args.eval_envs)
    solved = spec(cfg.env_id).solved_return
    verdict = "  <- solved" if result["mean_return"] >= solved else ""
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
        gym_eval(
            trainer.agent, cfg.env_id, args.gym_eval, cfg.max_episode_steps * repeat, args.seed, repeat=repeat
        )


if __name__ == "__main__":
    main()
