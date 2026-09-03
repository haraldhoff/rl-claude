"""Cross-check the Warp lander against the real Box2D LunarLander-v3.

Two directions, both needing ``gymnasium[box2d]``:

1. fly Gymnasium's own heuristic controller in *both* environments -- if our
   dynamics are a fair stand-in, it should score about the same in each;
2. fly a policy trained here in Box2D (zero-shot transfer), which measures what
   the dynamics gap costs.

    python tools/crosscheck_box2d.py --weights weights/lunarlander.npz --episodes 64
    python tools/crosscheck_box2d.py --backend jax --weights weights/jax/lunarlander.npz
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"))

import rl_common
from rl_common import to_numpy
from test_lunar_lander import heuristic


def run_backend(policy, episodes: int, seed: int, backend: str) -> tuple[float, float, int]:
    env = rl_common.make("lunarlander", episodes, backend=backend, autoreset=False, seed=seed)
    obs, _ = env.reset()
    alive = np.ones(episodes, dtype=bool)
    returns = np.zeros(episodes)
    last = np.zeros(episodes)
    for _ in range(env.max_episode_steps):
        obs, reward, terminated, truncated, _ = env.step(policy(to_numpy(obs)))
        done = (to_numpy(terminated) + to_numpy(truncated)) > 0
        returns += alive * to_numpy(reward)
        last = np.where(alive & done, to_numpy(reward), last)
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
    p.add_argument("--backend", type=str, default="warp", choices=rl_common.BACKENDS)
    p.add_argument("--weights", type=str, default="weights/lunarlander.npz")
    p.add_argument("--episodes", type=int, default=64)
    p.add_argument("--seed", type=int, default=1000)
    args = p.parse_args()

    print(f"{'':34s}  {'mean':>8s}  {'std':>6s}  landed")
    for name in (args.backend, "box2d"):
        if name == "box2d":
            mean, std, landed = run_gym(heuristic, args.episodes, args.seed)
        else:
            mean, std, landed = run_backend(heuristic, args.episodes, args.seed, name)
        print(f"gymnasium heuristic in {name:<11s}  {mean:8.1f}  {std:6.1f}  {landed}/{args.episodes}")

    if os.path.exists(args.weights):
        agent = rl_common.make_agent("lunarlander", backend=args.backend)
        agent.load(args.weights)
        mean, std, landed = run_gym(lambda states: agent.act_numpy(states), args.episodes, args.seed)
        label = f"{args.backend}-trained policy in box2d"
        print(f"{label:34s}  {mean:8.1f}  {std:6.1f}  {landed}/{args.episodes}")
    else:
        print(f"(no weights at {args.weights}; run train.py --env lunarlander --save {args.weights})")


if __name__ == "__main__":
    main()
