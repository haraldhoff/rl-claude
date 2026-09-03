"""Convert saved weights between the Warp and JAX backends.

The two backends store the same two MLPs under different names -- warp-nn's
``policy.0.weight`` / ``policy.0.bias`` (shape ``(out, in)`` / ``(out, 1)``)
versus Flax's ``params/policy/Dense_0/kernel`` / ``bias`` (``(in, out)`` /
``(out,)``) -- so a checkpoint is portable with a transpose and a rename.

    python tools/convert_weights.py --env lunarlander --from warp --to jax \\
        weights/lunarlander.npz weights/jax/lunarlander.npz
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rl_common

# Sequential indices of the Linear layers in the Warp model (Linear, Tanh, ...)
_WARP_LINEAR_INDICES = (0, 2, 4, 6, 8)


def warp_to_jax(env_id: str, source: str, destination: str, hidden=None) -> None:
    warp_agent = rl_common.make_agent(env_id, backend="warp", hidden=hidden)
    jax_agent = rl_common.make_agent(env_id, backend="jax", hidden=hidden)
    warp_agent.load(source)

    state = {k: v.numpy() for k, v in warp_agent.state_dict().items()}
    params = jax_agent.params["params"]
    for net in ("policy", "value"):
        dense = 0
        for layer in _WARP_LINEAR_INDICES:
            key = f"{net}.{layer}.weight"
            if key not in state:
                continue
            params[net][f"Dense_{dense}"]["kernel"] = state[key].T
            params[net][f"Dense_{dense}"]["bias"] = state[f"{net}.{layer}.bias"].reshape(-1)
            dense += 1
    jax_agent.save(destination)


def jax_to_warp(env_id: str, source: str, destination: str, hidden=None) -> None:
    import warp as wp

    warp_agent = rl_common.make_agent(env_id, backend="warp", hidden=hidden)
    jax_agent = rl_common.make_agent(env_id, backend="jax", hidden=hidden)
    jax_agent.load(source)

    params = jax_agent.params["params"]
    state = warp_agent.state_dict()
    for net in ("policy", "value"):
        dense = 0
        for layer in _WARP_LINEAR_INDICES:
            key = f"{net}.{layer}.weight"
            if key not in state:
                continue
            # Stage through host numpy.  Handing wp.array() a JAX *device*
            # array wraps it by pointer, and the temporary that pointer refers
            # to is unreferenced the moment this statement ends -- JAX then
            # reuses the address for the next allocation of that shape while
            # Warp's copy is still queued.  That silently corrupted every
            # parameter whose shape occurs in both the policy and the value
            # net, which on cartpole is four of the six weight tensors, and
            # was unreachable while jaxlib was CPU-only.
            kernel = np.ascontiguousarray(np.asarray(params[net][f"Dense_{dense}"]["kernel"]).T)
            bias = np.ascontiguousarray(np.asarray(params[net][f"Dense_{dense}"]["bias"]).reshape(-1, 1))
            wp.copy(state[key], wp.array(kernel, dtype=wp.float32, device=state[key].device))
            wp.copy(
                state[f"{net}.{layer}.bias"],
                wp.array(bias, dtype=wp.float32, device=state[key].device),
            )
            dense += 1
    warp_agent.save(destination)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source")
    p.add_argument("destination")
    p.add_argument("--env", type=str, required=True, help=f"environment id: {', '.join(rl_common.env_ids())}")
    p.add_argument("--from", dest="source_backend", type=str, required=True, choices=rl_common.BACKENDS)
    p.add_argument("--to", dest="target_backend", type=str, required=True, choices=rl_common.BACKENDS)
    args = p.parse_args()

    if args.source_backend == args.target_backend:
        raise SystemExit("--from and --to must differ")
    os.makedirs(os.path.dirname(os.path.abspath(args.destination)), exist_ok=True)

    convert = warp_to_jax if args.source_backend == "warp" else jax_to_warp
    convert(args.env, args.source, args.destination)
    print(f"converted {args.source} ({args.source_backend}) -> {args.destination} ({args.target_backend})")


if __name__ == "__main__":
    main()
