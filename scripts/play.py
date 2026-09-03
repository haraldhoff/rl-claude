"""Watch a policy play, on any backend.

    python scripts/play.py                                        # train cartpole, then open a window
    python scripts/play.py --env lunarlander --weights weights/lunarlander.npz
    python scripts/play.py --backend jax --env mountaincar --weights weights/jax/mountaincar.npz
    python scripts/play.py --num-render 9 --gif media/grid.gif
    python scripts/play.py --random --stochastic                  # an untrained policy
"""

from __future__ import annotations

import argparse
import os
import sys

# the repo root, so `python scripts/play.py` works from a bare checkout and
# not only from an installed package -- same bootstrap as tools/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from rl_common import cli, make, make_agent, make_renderer, make_trainer, spec, to_numpy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_arguments(parser, overrides=False)
    parser.add_argument("--weights", type=str, default=None, help="weights saved by train.py --save")
    parser.add_argument("--train-steps", type=int, default=None, help="steps to train when no weights are given")
    parser.add_argument("--random", action="store_true", help="play an untrained policy instead")
    parser.add_argument("--stochastic", action="store_true", help="sample actions instead of taking the argmax")
    parser.add_argument("--episodes", type=int, default=3, help="episodes to play (counted on the first env)")
    parser.add_argument("--num-render", type=int, default=1, help="how many environments to show side by side")
    parser.add_argument("--cols", type=int, default=None)
    parser.add_argument("--tile", type=int, nargs=2, default=None, metavar=("W", "H"), help="tile size in pixels")
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--gif", type=str, default=None, help="record to this .gif/.mp4 instead of opening a window")
    parser.add_argument("--gif-every", type=int, default=1, help="record every Nth frame (2 = half the frames)")
    return parser.parse_args()


def get_agent(args, cfg):
    """An untrained, a loaded, or a freshly trained policy."""
    if args.random:
        print("playing an untrained (randomly initialized) policy")
        return make_agent(args.env, backend=args.backend, hidden=cfg.hidden, seed=args.seed, device=args.device)
    if args.weights:
        agent = make_agent(args.env, backend=args.backend, hidden=cfg.hidden, seed=args.seed, device=args.device)
        agent.load(args.weights)
        print(f"loaded weights from {args.weights}")
        return agent
    print(f"no weights given -- training {cfg.env_id} on {cfg.backend} for {cfg.total_timesteps:,} steps first")
    trainer = make_trainer(cfg, device=args.device)
    trainer.train(log_every=10)
    print(f"greedy evaluation: {trainer.evaluate(num_envs=64)['mean_return']:.1f}")
    return trainer.agent


def write_recording(path: str, frames: list, fps: float, every: int) -> None:
    import imageio.v2 as imageio

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if path.lower().endswith(".gif"):
        imageio.mimsave(path, frames, duration=1000.0 * every / fps, loop=0)
    else:
        imageio.mimsave(path, frames, fps=fps / every)
    print(f"wrote {len(frames)} frames to {path} ({os.path.getsize(path) / 1e6:.1f} MB)")


def main() -> None:
    args = parse_args()
    cfg = cli.config_from_args(
        args, total_timesteps=args.train_steps, max_episode_steps=args.max_episode_steps
    )
    agent = get_agent(args, cfg)
    num_envs = max(1, args.num_render)
    fps = args.fps or spec(args.env).render_fps

    # environments that train with an action repeat are played back at the
    # physics rate: same trajectory, but every frame is drawn
    repeat = int(cfg.env_kwargs.get("action_repeat", 1))
    env_kwargs = {k: v for k, v in cfg.env_kwargs.items() if k != "action_repeat"}
    env = make(
        args.env,
        num_envs,
        backend=cfg.resolved_env_backend,
        max_episode_steps=cfg.max_episode_steps * repeat,
        device=args.device,
        seed=args.seed,
        **env_kwargs,
    )
    obs, _ = env.reset()

    renderer = make_renderer(
        args.env,
        env,
        mode="rgb_array" if args.gif else "human",
        num_render=num_envs,
        cols=args.cols,
        tile_size=tuple(args.tile) if args.tile else None,
        fps=fps,
        caption=f"{args.env} ({args.backend}) -- press ESC to quit",
    )

    frames = []
    returns = []
    step = 0
    actions = None
    max_steps = args.episodes * cfg.max_episode_steps * repeat + 10

    frame = renderer.render()  # initial state
    if frame is not None:
        frames.append(frame)

    while len(returns) < args.episodes and step < max_steps and not renderer.closed:
        if step % repeat == 0:  # otherwise hold the previous action
            actions = agent.act(obs, stochastic=args.stochastic)

        # the auto-reset clears ep_return, so accumulate env 0's return here
        running = float(renderer.state["ep_return"][0]) if renderer.state else 0.0
        obs, reward, terminated, truncated, _ = env.step(actions)
        step += 1
        if to_numpy(terminated)[0] + to_numpy(truncated)[0] > 0:
            returns.append(running + float(to_numpy(reward)[0]))
            print(f"episode {len(returns)}: return {returns[-1]:.1f}")

        frame = renderer.render()
        if frame is not None and step % args.gif_every == 0:
            frames.append(frame)

    renderer.close()
    if returns:
        print(f"{len(returns)} episodes on env 0: mean return {np.mean(returns):.1f}")
    if args.gif and frames:
        write_recording(args.gif, frames, fps, args.gif_every)


if __name__ == "__main__":
    main()
