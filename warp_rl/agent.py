"""Policy and value networks in warp-nn, and the Warp backend's agent."""

from __future__ import annotations

import numpy as np
import warp as wp
from warp_nn import nn

from rl_common.agent import Agent


def _orthogonal(shape: tuple[int, int], gain: float, rng: np.random.Generator) -> np.ndarray:
    """Orthogonal initialization (the standard PPO choice)."""
    a = rng.normal(size=shape).astype(np.float64)
    q, r = np.linalg.qr(a.T if shape[0] < shape[1] else a)
    q = q * np.sign(np.diag(r))
    if shape[0] < shape[1]:
        q = q.T
    return (gain * q).astype(np.float32)


def _init_layer(layer: nn.Linear, gain: float, rng: np.random.Generator) -> None:
    w = _orthogonal((layer.out_features, layer.in_features), gain, rng)
    wp.copy(layer.weight.data, wp.array(w, dtype=wp.float32, device=layer.device))
    if layer.bias is not None:
        layer.bias.data.zero_()


def mlp(
    in_features: int,
    out_features: int,
    *,
    hidden: tuple[int, ...] = (64, 64),
    out_gain: float = 0.01,
    rng: np.random.Generator | None = None,
    device: str | wp.Device | None = None,
) -> nn.Sequential:
    """Tanh MLP with orthogonal init (gain sqrt(2) hidden, ``out_gain`` head)."""
    rng = np.random.default_rng() if rng is None else rng
    layers: list[nn.Module] = []
    sizes = (in_features, *hidden)
    for i in range(len(hidden)):
        layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.Tanh()]
    layers.append(nn.Linear(sizes[-1], out_features))

    net = nn.Sequential(*layers)
    if device is not None:
        net = net.to(device)

    gains = [np.sqrt(2.0)] * len(hidden) + [out_gain]
    linears = [m for m in net.modules() if isinstance(m, nn.Linear)]
    for layer, gain in zip(linears, gains):
        _init_layer(layer, gain, rng)
    return net


class ActorCritic(Agent):
    """Separate actor and critic MLPs (the CleanRL CartPole configuration)."""

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        *,
        hidden: tuple[int, ...] = (64, 64),
        seed: int = 0,
        device: str | wp.Device | None = None,
    ):
        rng = np.random.default_rng(seed)
        self.device = wp.get_device(device)
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.policy = mlp(obs_dim, num_actions, hidden=hidden, out_gain=0.01, rng=rng, device=self.device)
        self.value = mlp(obs_dim, 1, hidden=hidden, out_gain=1.0, rng=rng, device=self.device)
        self._action_seed = seed + 12345
        self._action_cache: dict[int, tuple] = {}

    def act(self, obs: wp.array, *, stochastic: bool = False) -> wp.array:
        """Actions for a batch of observations (argmax, or sampled)."""
        from .kernels import make_action_kernels
        from .vec_env import seed_kernel

        n = obs.shape[0]
        if n not in self._action_cache:
            sample, greedy = make_action_kernels(self.num_actions)
            actions = wp.zeros(n, dtype=wp.int32, device=self.device)
            log_probs = wp.zeros(n, dtype=wp.float32, device=self.device)
            rng_states = wp.zeros(n, dtype=wp.uint32, device=self.device)
            wp.launch(seed_kernel, dim=n, inputs=[self._action_seed, rng_states], device=self.device)
            self._action_cache[n] = (sample, greedy, actions, log_probs, rng_states)
        sample, greedy, actions, log_probs, rng_states = self._action_cache[n]

        logits = self.policy(obs)
        if stochastic:
            wp.launch(sample, dim=n, inputs=[logits, rng_states, actions, log_probs], device=self.device)
        else:
            wp.launch(greedy, dim=n, inputs=[logits, actions], device=self.device)
        return actions

    def parameters(self) -> list[wp.array]:
        return self.policy.parameters() + self.value.parameters()

    def act_numpy(self, obs, *, stochastic: bool = False) -> np.ndarray:
        """``act`` for host observations (used by the Gymnasium cross-check)."""
        obs = np.ascontiguousarray(np.asarray(obs, dtype=np.float32)).reshape(-1, self.obs_dim)
        key = ("host_obs", obs.shape[0])
        if key not in self._action_cache:
            self._action_cache[key] = wp.zeros(obs.shape, dtype=wp.float32, device=self.device)
        buffer = self._action_cache[key]
        buffer.assign(obs)
        return self.act(buffer, stochastic=stochastic).numpy()

    def state_dict(self) -> dict[str, wp.array]:
        out = {}
        for k, v in self.policy.state_dict().items():
            out[f"policy.{k}"] = v
        for k, v in self.value.state_dict().items():
            out[f"value.{k}"] = v
        return out

    def save(self, path: str) -> None:
        np.savez(path, **{k: v.numpy() for k, v in self.state_dict().items()})

    def load(self, path: str) -> None:
        data = np.load(path)
        for k, v in self.state_dict().items():
            wp.copy(v, wp.array(data[k], dtype=v.dtype, device=v.device))


class _ActionScratch:
    """Per-batch-size kernels and buffers behind :meth:`ActorCritic.act`."""

    def __init__(self, size: int, obs_dim: int, num_actions: int, seed: int, device):
        from .kernels import make_action_kernels
        from .vec_env import seed_kernel

        self.size = size
        self.sample, self.greedy = make_action_kernels(num_actions)
        self.actions = wp.zeros(size, dtype=wp.int32, device=device)
        self.log_probs = wp.zeros(size, dtype=wp.float32, device=device)
        self.obs = wp.zeros((size, obs_dim), dtype=wp.float32, device=device)
        self.rng_states = wp.zeros(size, dtype=wp.uint32, device=device)
        wp.launch(seed_kernel, dim=size, inputs=[seed, self.rng_states], device=device)
