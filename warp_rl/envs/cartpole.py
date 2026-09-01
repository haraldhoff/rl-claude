"""CartPole-v1 in Warp kernels.

Physics and constants are a faithful port of ``gymnasium/envs/classic_control/
cartpole.py`` (``euler`` integrator), so a Warp env stepped with the same
actions from the same state reproduces Gymnasium up to float32 rounding.  The
constants live in :mod:`rl_common.specs.cartpole`, shared with the JAX backend.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from rl_common.specs import cartpole as spec

from ..vec_env import WarpVecEnv


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
    p.gravity = spec.GRAVITY
    p.masscart = spec.MASSCART
    p.masspole = spec.MASSPOLE
    p.total_mass = spec.TOTAL_MASS
    p.length = spec.LENGTH
    p.polemass_length = spec.POLEMASS_LENGTH
    p.force_mag = spec.FORCE_MAG
    p.tau = spec.TAU
    p.theta_threshold = spec.THETA_THRESHOLD
    p.x_threshold = spec.X_THRESHOLD
    p.reset_bound = spec.RESET_BOUND
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
    obs_dim = spec.OBS_DIM
    num_actions = spec.NUM_ACTIONS

    def __init__(self, num_envs: int = 1, *, max_episode_steps: int = spec.MAX_EPISODE_STEPS, **kwargs):
        self.params = default_params()
        super().__init__(num_envs, max_episode_steps=max_episode_steps, **kwargs)

    def observation_high(self) -> np.ndarray:
        return np.array([spec.X_THRESHOLD * 2.0, np.inf, spec.THETA_THRESHOLD * 2.0, np.inf], dtype=np.float32)

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
