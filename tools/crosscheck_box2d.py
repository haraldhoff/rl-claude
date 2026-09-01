"""Cross-check the Warp lander against the real Box2D LunarLander-v3.

Two directions, both needing ``gymnasium[box2d]``:

1. fly Gymnasium's own heuristic controller in *both* environments -- if the
   Warp dynamics are a fair stand-in, it should score about the same in each;
2. fly a Warp-trained policy in Box2D (zero-shot transfer), which measures what
   the dynamics gap costs.

    python tools/crosscheck_box2d.py --weights weights/lunarlander.npz --episodes 64
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import warp as wp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"))

import warp_rl  # noqa: E402
from test_lunar_lander import heuristic  # noqa: E402
from warp_rl.kernels import make_action_kernels  # noqa: E402


def run_warp(policy, episodes: int, seed: int) -> tuple[float, float, int]:
    env = warp_rl.make("lunarlander", episodes, autoreset=False, seed=seed)
    obs, _ = env.reset()
    alive = np.ones(episodes, dtype=bool)
    returns = np.zeros(episodes)
    last = np.zeros(episodes)
    for _ in range(env.max_episode_steps):
        obs, reward, terminated, truncated, _ = env.step(policy(obs.numpy()))
        done = (terminated.numpy() + truncated.numpy()) > 0
        returns += alive * reward.numpy()
        last = np.where(alive & done, reward.numpy(), last)
        alive &= ~done
        if not alive.any():
            break
    return float(returns.mean()), float(returns.std()), int((last > 50).sum())


def run_gym(policy, episodes: int, seed: int) -> tuple[float, float, int]:
    import gymnasium as gym

    env = gym.make("LunarLander-v3")
    returns = []
    landed = 0
    for ep in range(episodes):
        s, _ = env.reset(seed=seed + ep)
        total, done = 0.0, False
        while not done:
            action = int(policy(np.asarray(s, dtype=np.float64)[None])[0])
            s, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            done = terminated or truncated
        returns.append(total)
        landed += int(total > 200)
    env.close()
    return float(np.mean(returns)), float(np.std(returns)), landed


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", type=str, default="weights/lunarlander.npz")
    p.add_argument("--episodes", type=int, default=64)
    p.add_argument("--seed", type=int, default=1000)
    args = p.parse_args()
    wp.init()

    print(f"{'':34s}  {'mean':>8s}  {'std':>6s}  landed")
    for name, runner in (("warp", run_warp), ("box2d", run_gym)):
        mean, std, landed = runner(heuristic, args.episodes, args.seed)
        print(f"gymnasium heuristic in {name:<11s}  {mean:8.1f}  {std:6.1f}  {landed}/{args.episodes}")

    if os.path.exists(args.weights):
        agent = warp_rl.ActorCritic(8, 4, hidden=(128, 128), device=wp.get_device())
        agent.load(args.weights)
        _, greedy = make_action_kernels(4)
        obs_arr = wp.zeros((1, 8), dtype=wp.float32, device=agent.device)
        act_arr = wp.zeros(1, dtype=wp.int32, device=agent.device)

        def policy(states: np.ndarray) -> np.ndarray:
            out = np.zeros(len(states), dtype=np.int32)
            for i, s in enumerate(states):
                obs_arr.assign(np.asarray(s, dtype=np.float32).reshape(1, 8))
                wp.launch(greedy, dim=1, inputs=[agent.policy(obs_arr), act_arr], device=agent.device)
                out[i] = act_arr.numpy()[0]
            return out

        mean, std, landed = run_gym(policy, args.episodes, args.seed)
        print(f"{'warp-trained policy in box2d':34s}  {mean:8.1f}  {std:6.1f}  {landed}/{args.episodes}")
    else:
        print(f"(no weights at {args.weights}; run train.py --env lunarlander --save {args.weights})")


if __name__ == "__main__":
    main()
