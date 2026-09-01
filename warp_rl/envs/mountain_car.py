"""MountainCar-v0 in Warp kernels.

A line-by-line port of ``gymnasium/envs/classic_control/mountain_car.py``: the
same update rule, the same clipping, the same ``-1`` per step and the same
termination test, so a Warp env stepped with the same actions from the same
state reproduces Gymnasium exactly (float32 rounding aside).

Actions: 0 = push left, 1 = do nothing, 2 = push right.

The one addition is ``action_repeat``: holding an action for several physics
steps.  With the default of 1 this is exactly MountainCar-v0; with 8 (what the
registry trains on) the agent decides every 8 frames, which is the difference
between an unsolvable exploration problem and a solvable one -- uniformly random
actions never reach the flag (0 of 4096 episodes), while random *held* actions
resonate up the hill in ~1.3% of them.  Rewards still count physics steps, so a
return is directly comparable to MountainCar-v0's.
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

MIN_POSITION = -1.2
MAX_POSITION = 0.6
MAX_SPEED = 0.07
GOAL_POSITION = 0.5
GOAL_VELOCITY = 0.0
FORCE = 0.001
GRAVITY = 0.0025
RESET_LOW = -0.6
RESET_HIGH = -0.4


@wp.struct
class MountainCarParams:
    min_position: float
    max_position: float
    max_speed: float
    goal_position: float
    goal_velocity: float
    force: float
    gravity: float
    reset_low: float
    reset_high: float


def default_params() -> MountainCarParams:
    p = MountainCarParams()
    p.min_position = MIN_POSITION
    p.max_position = MAX_POSITION
    p.max_speed = MAX_SPEED
    p.goal_position = GOAL_POSITION
    p.goal_velocity = GOAL_VELOCITY
    p.force = FORCE
    p.gravity = GRAVITY
    p.reset_low = RESET_LOW
    p.reset_high = RESET_HIGH
    return p


def height(x):
    """Terrain profile, as Gymnasium draws it."""
    return np.sin(3.0 * np.asarray(x)) * 0.45 + 0.55


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def reset_kernel(
    params: MountainCarParams,
    needs_reset: wp.array(dtype=wp.int32),
    rng_states: wp.array(dtype=wp.uint32),
    obs: wp.array2d(dtype=wp.float32),
):
    i = wp.tid()
    if needs_reset[i] == 0:
        return
    rng = rng_states[i]
    obs[i, 0] = wp.randf(rng, params.reset_low, params.reset_high)
    obs[i, 1] = 0.0
    rng_states[i] = rng


@wp.kernel(enable_backward=False)
def step_kernel(
    params: MountainCarParams,
    repeat: wp.int32,
    actions: wp.array(dtype=wp.int32),
    obs: wp.array2d(dtype=wp.float32),  # in: s_t, out: s_t+1 (before auto-reset)
    final_obs: wp.array2d(dtype=wp.float32),
    rewards: wp.array(dtype=wp.float32),
    terminated: wp.array(dtype=wp.float32),
    truncated: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    position = obs[i, 0]
    velocity = obs[i, 1]

    term = float(0.0)
    reward = float(0.0)
    for _ in range(repeat):
        if term == 0.0:  # the episode ends the instant the flag is reached
            velocity += float(actions[i] - 1) * params.force + wp.cos(3.0 * position) * (-params.gravity)
            velocity = wp.clamp(velocity, -params.max_speed, params.max_speed)
            position = wp.clamp(position + velocity, params.min_position, params.max_position)
            if position <= params.min_position and velocity < 0.0:
                velocity = 0.0
            reward -= 1.0
            if position >= params.goal_position and velocity >= params.goal_velocity:
                term = 1.0

    rewards[i] = reward
    terminated[i] = term
    truncated[i] = 0.0

    obs[i, 0] = position
    obs[i, 1] = velocity
    final_obs[i, 0] = position
    final_obs[i, 1] = velocity


# ---------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------


class MountainCarVectorEnv(WarpVecEnv):
    """``num_envs`` independent MountainCar-v0 environments stepped in lockstep.

    The reward is Gymnasium's: ``-1`` every step until the flag is reached, which
    makes this a pure exploration problem -- see the registry for the settings
    that get PPO over the hill.
    """

    env_id = "mountaincar"
    obs_dim = 2
    num_actions = 3

    def __init__(self, num_envs: int = 1, *, max_episode_steps: int = 200, action_repeat: int = 1, **kwargs):
        self.params = default_params()
        self.action_repeat = int(action_repeat)
        super().__init__(num_envs, max_episode_steps=max_episode_steps, **kwargs)

    @property
    def physics_steps_per_episode(self) -> int:
        """The MountainCar-v0 time limit is 200 *physics* steps."""
        return self.max_episode_steps * self.action_repeat

    def observation_high(self) -> np.ndarray:
        return np.array([MAX_POSITION, MAX_SPEED], dtype=np.float32)

    def _build_spaces(self) -> None:
        super()._build_spaces()
        try:  # the position range is not symmetric, unlike the default box
            from gymnasium import spaces
        except Exception:  # pragma: no cover
            return
        low = np.array([MIN_POSITION, -MAX_SPEED], dtype=np.float32)
        high = np.array([MAX_POSITION, MAX_SPEED], dtype=np.float32)
        self.single_observation_space = spaces.Box(low, high, dtype=np.float32)
        self.observation_space = spaces.Box(
            np.tile(low, (self.num_envs, 1)), np.tile(high, (self.num_envs, 1)), dtype=np.float32
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
                self.action_repeat,
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
        """Overwrite ``(position, velocity)`` for every environment."""
        arr = np.ascontiguousarray(np.asarray(state, dtype=np.float32)).reshape(self.num_envs, 2)
        self.obs.assign(arr)
        self.steps.zero_()
        self.ep_return.zero_()
        self.ep_length.zero_()


# ---------------------------------------------------------------------------
# renderer
# ---------------------------------------------------------------------------

_HILL = (60, 60, 70)
_GROUND = (215, 215, 225)
_CAR = (20, 20, 30)
_WHEEL = (128, 128, 128)
_FLAG_POLE = (20, 20, 30)
_FLAG = (204, 204, 0)

_CAR_W, _CAR_H = 40.0, 20.0
_CLEARANCE = 10.0


class MountainCarRenderer(TiledRenderer):
    """The classic-control mountain car picture, drawn from the device state."""

    def setup(self) -> None:
        self.scale = self.tile_w / (MAX_POSITION - MIN_POSITION)
        self.xs = np.linspace(MIN_POSITION, MAX_POSITION, 100)
        self.ys = height(self.xs)

    def stats_label(self, index: int) -> str:
        position, velocity = self.states[index]
        return (
            f"env {index}  t={int(self.steps[index]):3d}  R={self.returns[index]:.0f}  "
            f"x={position:+.2f}  v={velocity:+.3f}"
        )

    def _to_screen(self, origin, x, y) -> tuple[float, float]:
        # y is measured up from the bottom of the tile, in "height" units
        return (
            float(origin[0] + (x - MIN_POSITION) * self.scale),
            float(origin[1] + self.tile_h - y * self.scale),
        )

    def draw_tile(self, origin: tuple[float, float], index: int) -> None:
        import pygame

        surf = self.surface
        hill = [self._to_screen(origin, x, y) for x, y in zip(self.xs, self.ys)]
        pygame.draw.polygon(
            surf,
            _GROUND,
            [self._to_screen(origin, MIN_POSITION, 0.0), *hill, self._to_screen(origin, MAX_POSITION, 0.0)],
        )
        pygame.draw.lines(surf, _HILL, False, hill, max(1, int(2 * self.k)))

        # flag on the goal
        flag_base = self._to_screen(origin, GOAL_POSITION, float(height(GOAL_POSITION)))
        flag_top = (flag_base[0], flag_base[1] - 50 * self.k)
        pygame.draw.line(surf, _FLAG_POLE, flag_base, flag_top, max(1, int(2 * self.k)))
        pygame.draw.polygon(
            surf,
            _FLAG,
            [flag_top, (flag_top[0] + 25 * self.k, flag_top[1] + 10 * self.k), (flag_top[0], flag_top[1] + 20 * self.k)],
        )

        # car: a box rotated to the slope, sitting `clearance` above the curve
        position = float(self.states[index, 0])
        angle = math.cos(3.0 * position)  # Gymnasium rotates by cos(3x) directly
        c, s = math.cos(angle), math.sin(angle)
        base = self._to_screen(origin, position, float(height(position)))
        base = (base[0], base[1] - _CLEARANCE * self.k)

        def place(px, py):
            px, py = px * self.k, py * self.k
            return (base[0] + c * px - s * py, base[1] - (s * px + c * py))

        body = [place(x, y) for x, y in ((-_CAR_W / 2, 0.0), (-_CAR_W / 2, _CAR_H), (_CAR_W / 2, _CAR_H), (_CAR_W / 2, 0.0))]
        pygame.draw.polygon(surf, _CAR, body)
        for wheel_x in (-_CAR_W / 4, _CAR_W / 4):
            pygame.draw.circle(surf, _WHEEL, place(wheel_x, 0.0), max(2.0, _CAR_H / 2.5 * self.k))
