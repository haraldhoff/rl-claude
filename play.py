"""Watch a policy play a Warp environment.

    python play.py                                   # train cartpole, then open a window
    python play.py --env lunarlander --weights lander.npz
    python play.py --num-render 9 --gif media/grid.gif
    python play.py --random --stochastic              # what an untrained policy looks like
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import warp as wp

from warp_rl import PPO, ActorCritic, default_config, env_ids, make, make_renderer, spec
from warp_rl.kernels import make_action_kernels
from warp_rl.vec_env import seed_kernel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", type=str, default="cartpole", help=f"environment id: {', '.join(env_ids())}")
    p.add_argument("--weights", type=str, default=None, help="npz saved by train.py --save")
    p.add_argument("--train-steps", type=int, default=None, help="steps to train first when no weights are given")
    p.add_argument("--random", action="store_true", help="play an untrained policy instead")
    p.add_argument("--stochastic", action="store_true", help="sample actions instead of taking the argmax")
    p.add_argument("--episodes", type=int, default=3, help="episodes to play (counted on the first env)")
    p.add_argument("--num-render", type=int, default=1, help="how many environments to show side by side")
    p.add_argument("--cols", type=int, default=None)
    p.add_argument("--tile", type=int, nargs=2, default=None, metavar=("W", "H"), help="tile size in pixels")
    p.add_argument("--max-episode-steps", type=int, default=None)
    p.add_argument("--fps", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--gif", type=str, default=None, help="record to this .gif/.mp4 instead of opening a window")
    p.add_argument("--gif-every", type=int, default=1, help="record every Nth frame (2 = half the frames)")
    return p.parse_args()


def get_agent(args, cfg) -> ActorCritic:
    device = wp.get_device(args.device)
    env_spec = spec(args.env)
    obs_dim, num_actions = env_spec.env_cls.obs_dim, env_spec.env_cls.num_actions
    if args.random:
        print("playing an untrained (randomly initialized) policy")
        return ActorCritic(obs_dim, num_actions, hidden=cfg.hidden, seed=args.seed, device=device)
    if args.weights:
        agent = ActorCritic(obs_dim, num_actions, hidden=cfg.hidden, seed=args.seed, device=device)
        agent.load(args.weights)
        print(f"loaded weights from {args.weights}")
        return agent
    print(f"no weights given -- training {cfg.env_id} for {cfg.total_timesteps:,} steps first")
    trainer = PPO(cfg, device=device)
    trainer.train(log_every=10)
    print(f"greedy evaluation: {trainer.evaluate(num_envs=64)['mean_return']:.1f}")
    return trainer.agent


def main() -> None:
    args = parse_args()
    wp.init()

    cfg = default_config(
        args.env,
        total_timesteps=args.train_steps,
        max_episode_steps=args.max_episode_steps,
        seed=args.seed,
    )
    agent = get_agent(args, cfg)
    device = agent.device
    n = max(1, args.num_render)
    fps = args.fps or spec(args.env).render_fps

    # environments that train with an action repeat are played back at the
    # physics rate: same trajectory, but every frame is drawn
    repeat = int(cfg.env_kwargs.get("action_repeat", 1))
    env_kwargs = {k: v for k, v in cfg.env_kwargs.items() if k != "action_repeat"}
    env = make(
        args.env,
        n,
        max_episode_steps=cfg.max_episode_steps * repeat,
        autoreset=True,
        device=device,
        seed=args.seed,
        **env_kwargs,
    )
    env.reset()

    sample, greedy = make_action_kernels(env.num_actions)
    actions = wp.zeros(n, dtype=wp.int32, device=device)
    rng_states = wp.zeros(n, dtype=wp.uint32, device=device)
    log_probs = wp.zeros(n, dtype=wp.float32, device=device)
    if args.stochastic:
        wp.launch(seed_kernel, dim=n, inputs=[args.seed + 999, rng_states], device=device)

    tile = tuple(args.tile) if args.tile else None
    renderer = make_renderer(
        args.env,
        env,
        mode="rgb_array" if args.gif else "human",
        num_render=n,
        cols=args.cols,
        tile_size=tile,
        fps=fps,
        caption=f"Warp {args.env} -- press ESC to quit",
    )

    frames = []
    episodes_done = 0
    returns = []
    step = 0
    max_steps = args.episodes * cfg.max_episode_steps * repeat + 10

    frame = renderer.render()  # initial state
    if frame is not None:
        frames.append(frame)

    while episodes_done < args.episodes and step < max_steps and not renderer.closed:
        if step % repeat == 0:  # otherwise hold the previous action
            logits = agent.policy(env.obs)
            if args.stochastic:
                wp.launch(sample, dim=n, inputs=[logits, rng_states, actions, log_probs], device=device)
            else:
                wp.launch(greedy, dim=n, inputs=[logits, actions], device=device)

        # the auto-reset clears ep_return, so accumulate env 0's return here
        running = float(env.ep_return.numpy()[0])
        _, reward, terminated, truncated, _ = env.step(actions)
        step += 1
        if terminated.numpy()[0] + truncated.numpy()[0] > 0:
            episodes_done += 1
            returns.append(running + float(reward.numpy()[0]))
            print(f"episode {episodes_done}: return {returns[-1]:.1f}")

        frame = renderer.render()
        if frame is not None and step % args.gif_every == 0:
            frames.append(frame)

    renderer.close()
    if returns:
        print(f"{len(returns)} episodes on env 0: mean return {np.mean(returns):.1f}")

    if args.gif and frames:
        import imageio.v2 as imageio

        directory = os.path.dirname(os.path.abspath(args.gif))
        os.makedirs(directory, exist_ok=True)
        if args.gif.lower().endswith(".gif"):
            imageio.mimsave(args.gif, frames, duration=1000.0 * args.gif_every / fps, loop=0)
        else:
            imageio.mimsave(args.gif, frames, fps=fps / args.gif_every)
        print(f"wrote {len(frames)} frames to {args.gif} ({os.path.getsize(args.gif) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
