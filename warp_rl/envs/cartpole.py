"""CartPole-v1 in Warp kernels.

Physics and constants are a faithful port of ``gymnasium/envs/classic_control/
cartpole.py`` (``euler`` integrator), so a Warp env stepped with the same
actions from the same state reproduces Gymnasium up to float32 rounding.
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

from ..render import TiledRenderer
from ..vec_env import WarpVecEnv


# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------


@wp.struct
class CartPoleParams:
    """Physical constants of the cart-pole system (passed to every kernel)."""

    gravity: float
    masscart: float
    masspole: float
    total_mass: float
    length: float  # actually half the pole's length
    polemass_length: float
    force_mag: float
    tau: float  # seconds between state updates
    theta_threshold: float
    x_threshold: float
    reset_bound: float


def default_params() -> CartPoleParams:
    p = CartPoleParams()
    p.gravity = 9.8
    p.masscart = 1.0
    p.masspole = 0.1
    p.total_mass = p.masscart + p.masspole
    p.length = 0.5
    p.polemass_length = p.masspole * p.length
    p.force_mag = 10.0
    p.tau = 0.02
    p.theta_threshold = 12.0 * 2.0 * np.pi / 360.0
    p.x_threshold = 2.4
    p.reset_bound = 0.05
    return p


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def reset_kernel(
    params: CartPoleParams,
    needs_reset: wp.array(dtype=wp.int32),
    rng_states: wp.array(dtype=wp.uint32),
    obs: wp.array2d(dtype=wp.float32),
):
    i = wp.tid()
    if needs_reset[i] == 0:
        return
    rng = rng_states[i]
    b = params.reset_bound
    for k in range(4):
        obs[i, k] = wp.randf(rng, -b, b)
    rng_states[i] = rng


@wp.kernel(enable_backward=False)
def step_kernel(
    params: CartPoleParams,
    actions: wp.array(dtype=wp.int32),
    obs: wp.array2d(dtype=wp.float32),  # in: s_t, out: s_t+1 (before auto-reset)
    final_obs: wp.array2d(dtype=wp.float32),
    rewards: wp.array(dtype=wp.float32),
    terminated: wp.array(dtype=wp.float32),
    truncated: wp.array(dtype=wp.float32),
):
    i = wp.tid()

    x = obs[i, 0]
    x_dot = obs[i, 1]
    theta = obs[i, 2]
    theta_dot = obs[i, 3]

    force = -params.force_mag
    if actions[i] == 1:
        force = params.force_mag

    costheta = wp.cos(theta)
    sintheta = wp.sin(theta)

    temp = (force + params.polemass_length * theta_dot * theta_dot * sintheta) / params.total_mass
    thetaacc = (params.gravity * sintheta - costheta * temp) / (
        params.length * (4.0 / 3.0 - params.masspole * costheta * costheta / params.total_mass)
    )
    xacc = temp - params.polemass_length * thetaacc * costheta / params.total_mass

    # explicit Euler, in Gymnasium's update order
    x = x + params.tau * x_dot
    x_dot = x_dot + params.tau * xacc
    theta = theta + params.tau * theta_dot
    theta_dot = theta_dot + params.tau * thetaacc

    term = 0.0
    if x < -params.x_threshold or x > params.x_threshold:
        term = 1.0
    if theta < -params.theta_threshold or theta > params.theta_threshold:
        term = 1.0

    rewards[i] = 1.0
    terminated[i] = term
    truncated[i] = 0.0

    obs[i, 0] = x
    obs[i, 1] = x_dot
    obs[i, 2] = theta
    obs[i, 3] = theta_dot
    for k in range(4):
        final_obs[i, k] = obs[i, k]


# ---------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------


class CartPoleVectorEnv(WarpVecEnv):
    """``num_envs`` independent CartPole-v1 environments stepped in lockstep."""

    env_id = "cartpole"
    obs_dim = 4
    num_actions = 2

    def __init__(self, num_envs: int = 1, *, max_episode_steps: int = 500, **kwargs):
        self.params = default_params()
        super().__init__(num_envs, max_episode_steps=max_episode_steps, **kwargs)

    def observation_high(self) -> np.ndarray:
        return np.array(
            [self.params.x_threshold * 2.0, np.inf, self.params.theta_threshold * 2.0, np.inf],
            dtype=np.float32,
        )

    def _reset(self) -> None:
        wp.launch(
            reset_kernel,
            dim=self.num_envs,
            inputs=[self.params, self.needs_reset, self.rng_states, self.obs],
            device=self.device,
        )

    def _step(self, actions: wp.array) -> None:
        wp.launch(
            step_kernel,
            dim=self.num_envs,
            inputs=[
                self.params,
                actions,
                self.obs,
                self.final_obs,
                self.rewards,
                self.terminated,
                self.truncated,
            ],
            device=self.device,
        )

    def set_state(self, state) -> None:
        """Overwrite the raw state of every environment (shape ``(num_envs, 4)``)."""
        arr = np.ascontiguousarray(np.asarray(state, dtype=np.float32)).reshape(self.num_envs, 4)
        self.obs.assign(arr)
        self.steps.zero_()
        self.ep_return.zero_()
        self.ep_length.zero_()


# ---------------------------------------------------------------------------
# renderer
# ---------------------------------------------------------------------------

_CART_W, _CART_H = 50.0, 30.0
_POLE_W = 10.0
_CART_Y = 300.0  # cart centre line, measured from the top of a 400px tile

_CART = (0, 0, 0)
_POLE = (202, 152, 101)
_AXLE = (129, 132, 203)
_TRACK = (0, 0, 0)
_LIMIT = (220, 90, 90)


class CartPoleRenderer(TiledRenderer):
    """The classic-control cart-pole picture, drawn from the device state."""

    def setup(self) -> None:
        import pygame  # noqa: F401  (imported by the base, kept local for clarity)

        env = self.env
        # a little margin beyond +/- x_threshold so the limit markers are visible
        self.scale = self.tile_w / (env.params.x_threshold * 2.0 * 1.15)
        self.pole_len = self.scale * (2.0 * env.params.length)
        self.cart_y = _CART_Y * (self.tile_h / 400.0)

    def stats_label(self, index: int) -> str:
        x, _, theta, _ = self.states[index]
        return (
            f"env {index}  t={int(self.steps[index]):3d}  R={self.returns[index]:.0f}  "
            f"x={x:+.2f}  th={math.degrees(theta):+5.1f}"
        )

    def draw_tile(self, origin: tuple[float, float], index: int) -> None:
        import pygame

        ox, oy = origin
        surf = self.surface
        x, _, theta, _ = (float(v) for v in self.states[index])

        cart_x = ox + self.tile_w / 2.0 + x * self.scale
        cart_y = oy + self.cart_y
        cart_w, cart_h = _CART_W * self.k, _CART_H * self.k
        pole_w = max(2.0, _POLE_W * self.k)

        # track and the +/- x_threshold limits that end an episode
        pygame.draw.line(surf, _TRACK, (ox, cart_y), (ox + self.tile_w, cart_y), max(1, int(2 * self.k)))
        for sign in (-1.0, 1.0):
            lx = ox + self.tile_w / 2.0 + sign * self.env.params.x_threshold * self.scale
            pygame.draw.line(
                surf, _LIMIT, (lx, cart_y - 20 * self.k), (lx, cart_y + 20 * self.k), max(1, int(2 * self.k))
            )

        pygame.draw.rect(surf, _CART, pygame.Rect(cart_x - cart_w / 2, cart_y - cart_h / 2, cart_w, cart_h))

        # pole as a rotated quad: +theta tilts it to the right
        axle = (cart_x, cart_y - cart_h / 4.0)
        along = (math.sin(theta), -math.cos(theta))
        across = (math.cos(theta) * pole_w / 2.0, math.sin(theta) * pole_w / 2.0)
        tip = (axle[0] + along[0] * self.pole_len, axle[1] + along[1] * self.pole_len)
        pygame.draw.polygon(
            surf,
            _POLE,
            [
                (axle[0] - across[0], axle[1] - across[1]),
                (axle[0] + across[0], axle[1] + across[1]),
                (tip[0] + across[0], tip[1] + across[1]),
                (tip[0] - across[0], tip[1] - across[1]),
            ],
        )
        pygame.draw.circle(surf, _AXLE, axle, pole_w / 2.0)
