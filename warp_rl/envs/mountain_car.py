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

import numpy as np
import warp as wp

from rl_common.specs import mountain_car as spec

from ..vec_env import WarpVecEnv


# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------


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
    p.min_position = spec.MIN_POSITION
    p.max_position = spec.MAX_POSITION
    p.max_speed = spec.MAX_SPEED
    p.goal_position = spec.GOAL_POSITION
    p.goal_velocity = spec.GOAL_VELOCITY
    p.force = spec.FORCE
    p.gravity = spec.GRAVITY
    p.reset_low = spec.RESET_LOW
    p.reset_high = spec.RESET_HIGH
    return p


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
    obs_dim = spec.OBS_DIM
    num_actions = spec.NUM_ACTIONS

    def __init__(
        self,
        num_envs: int = 1,
        *,
        max_episode_steps: int = spec.MAX_EPISODE_STEPS,
        action_repeat: int = 1,
        **kwargs,
    ):
        self.params = default_params()
        self.action_repeat = int(action_repeat)
        super().__init__(num_envs, max_episode_steps=max_episode_steps, **kwargs)

    @property
    def physics_steps_per_episode(self) -> int:
        """The MountainCar-v0 time limit is 200 *physics* steps."""
        return self.max_episode_steps * self.action_repeat

    def observation_high(self) -> np.ndarray:
        return np.array([spec.MAX_POSITION, spec.MAX_SPEED], dtype=np.float32)

    def _build_spaces(self) -> None:
        super()._build_spaces()
        try:  # the position range is not symmetric, unlike the default box
            from gymnasium import spaces
        except Exception:  # pragma: no cover
            return
        low = np.array([spec.MIN_POSITION, -spec.MAX_SPEED], dtype=np.float32)
        high = np.array([spec.MAX_POSITION, spec.MAX_SPEED], dtype=np.float32)
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
