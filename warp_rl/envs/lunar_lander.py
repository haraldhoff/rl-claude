"""LunarLander in Warp kernels.

The task definition -- terrain generation, observation vector, reward shaping,
engine impulses, termination rules and the 4 discrete actions -- is ported
line-by-line from ``gymnasium/envs/box2d/lunar_lander.py`` (LunarLander-v3,
discrete).  The *dynamics* are not Box2D: instead of a constraint solver with
articulated legs, the lander is a single rigid body (hull + welded legs, with
the mass, centre of mass and moment of inertia computed from the very same
polygons and densities) that touches the ground through two leg contact points
resolved with a penalty spring-damper and Coulomb friction.  Trajectories
therefore differ from Box2D's, but the control problem -- and every number the
agent sees -- is the same.

Deviations worth knowing:

* legs are rigid, not sprung revolute joints;
* contacts are compliant (a few millimetres of penetration under weight)
  instead of hard constraints, with friction ``mu = 0.5`` rather than Box2D's
  slippery 0.1 (documented in :data:`CONTACT_FRICTION`);
* "the lander came to rest" replaces Box2D's sleep test: velocities below
  ``SLEEP_V`` / ``SLEEP_W`` for half a second with no engine firing;
* ``reset()`` does not run Gymnasium's extra zero-action step; the shaping
  baseline is seeded from the initial state instead, so the first reward is a
  normal shaping delta;
* no wind/turbulence and no continuous action space.
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

from rl_common.specs import lunar_lander as spec

from ..vec_env import WarpVecEnv


# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------


@wp.struct
class LanderParams:
    gravity: float
    dt: float
    substeps: int
    mass: float
    inertia: float
    com: wp.vec2
    main_power: float
    side_power: float
    contact_k: float
    contact_c: float
    contact_mu: float
    contact_kt: float
    crash_speed: float
    world_w: float
    world_h: float
    helipad_y: float
    chunk_dx: float
    leg_down: float
    initial_random: float
    sleep_v: float
    sleep_w: float
    sleep_steps: int


def default_params() -> LanderParams:
    p = LanderParams()
    p.gravity = spec.GRAVITY
    p.dt = 1.0 / spec.FPS
    p.substeps = spec.SUBSTEPS
    p.mass = spec.MASS
    p.inertia = spec.INERTIA
    p.com = wp.vec2(float(spec.COM[0]), float(spec.COM[1]))
    p.main_power = spec.MAIN_ENGINE_POWER
    p.side_power = spec.SIDE_ENGINE_POWER
    p.contact_k = spec.CONTACT_STIFFNESS
    p.contact_c = spec.CONTACT_DAMPING
    p.contact_mu = spec.CONTACT_FRICTION
    p.contact_kt = spec.CONTACT_TANGENT_DAMPING
    p.crash_speed = spec.CRASH_SPEED
    p.world_w = spec.WORLD_W
    p.world_h = spec.WORLD_H
    p.helipad_y = spec.HELIPAD_Y
    p.chunk_dx = spec.CHUNK_DX
    p.leg_down = spec.LEG_DOWN / spec.SCALE
    p.initial_random = spec.INITIAL_RANDOM
    p.sleep_v = spec.SLEEP_V
    p.sleep_w = spec.SLEEP_W
    p.sleep_steps = spec.SLEEP_STEPS
    return p


# ---------------------------------------------------------------------------
# kernel helpers
# ---------------------------------------------------------------------------


@wp.func
def rotate(angle: float, v: wp.vec2) -> wp.vec2:
    c = wp.cos(angle)
    s = wp.sin(angle)
    return wp.vec2(c * v[0] - s * v[1], s * v[0] + c * v[1])


@wp.func
def terrain_segment(x: float, chunk_dx: float, n: int) -> int:
    idx = int(wp.floor(x / chunk_dx))
    return wp.clamp(idx, 0, n - 2)


@wp.func
def ground_height(terrain: wp.array2d(dtype=wp.float32), i: int, x: float, chunk_dx: float) -> float:
    idx = terrain_segment(x, chunk_dx, terrain.shape[1])
    t = wp.clamp(x / chunk_dx - float(idx), 0.0, 1.0)
    return terrain[i, idx] * (1.0 - t) + terrain[i, idx + 1] * t


@wp.func
def ground_normal(terrain: wp.array2d(dtype=wp.float32), i: int, x: float, chunk_dx: float) -> wp.vec2:
    idx = terrain_segment(x, chunk_dx, terrain.shape[1])
    dy = terrain[i, idx + 1] - terrain[i, idx]
    return wp.normalize(wp.vec2(-dy, chunk_dx))


@wp.func
def write_observation(
    params: LanderParams,
    i: int,
    com: wp.vec2,
    vel: wp.vec2,
    angle: float,
    omega: float,
    c0: float,
    c1: float,
    obs: wp.array2d(dtype=wp.float32),
) -> float:
    """Write the 8-dim observation and return the reward-shaping potential."""
    origin = com - rotate(angle, params.com)
    half_w = params.world_w * 0.5
    half_h = params.world_h * 0.5

    s0 = (origin[0] - half_w) / half_w
    s1 = (origin[1] - (params.helipad_y + params.leg_down)) / half_h
    s2 = vel[0] * half_w / 50.0
    s3 = vel[1] * half_h / 50.0
    s4 = angle
    s5 = 20.0 * omega / 50.0

    obs[i, 0] = s0
    obs[i, 1] = s1
    obs[i, 2] = s2
    obs[i, 3] = s3
    obs[i, 4] = s4
    obs[i, 5] = s5
    obs[i, 6] = c0
    obs[i, 7] = c1

    return (
        -100.0 * wp.sqrt(s0 * s0 + s1 * s1)
        - 100.0 * wp.sqrt(s2 * s2 + s3 * s3)
        - 100.0 * wp.abs(s4)
        + 10.0 * c0
        + 10.0 * c1
    )


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def reset_kernel(
    params: LanderParams,
    needs_reset: wp.array(dtype=wp.int32),
    rng_states: wp.array(dtype=wp.uint32),
    raw_terrain: wp.array2d(dtype=wp.float32),  # (N, CHUNKS+1) scratch
    terrain: wp.array2d(dtype=wp.float32),  # (N, CHUNKS)
    body: wp.array2d(dtype=wp.float32),  # (N, 6): com x, y, vx, vy, angle, omega
    contacts: wp.array2d(dtype=wp.float32),
    engine: wp.array2d(dtype=wp.float32),
    prev_shaping: wp.array(dtype=wp.float32),
    sleep_counter: wp.array(dtype=wp.int32),
    obs: wp.array2d(dtype=wp.float32),
):
    i = wp.tid()
    if needs_reset[i] == 0:
        return
    rng = rng_states[i]

    # terrain: CHUNKS+1 random heights, a flat helipad in the middle, then the
    # same 0.33-weighted smoothing Gymnasium uses (0.33, not 1/3)
    n_raw = raw_terrain.shape[1]
    for j in range(n_raw):
        raw_terrain[i, j] = wp.randf(rng, 0.0, params.world_h * 0.5)
    for j in range(3, 8):
        raw_terrain[i, j] = params.helipad_y
    for j in range(terrain.shape[1]):
        jm = j - 1
        if jm < 0:
            jm = n_raw - 1
        terrain[i, j] = 0.33 * (raw_terrain[i, jm] + raw_terrain[i, j] + raw_terrain[i, j + 1])

    # the lander starts at the top centre with a random shove, which Gymnasium
    # applies as a one-frame force to the centre of mass
    kick = params.initial_random * params.dt / params.mass
    origin = wp.vec2(params.world_w * 0.5, params.world_h)
    com = origin + params.com
    vel = wp.vec2(wp.randf(rng, -1.0, 1.0) * kick, wp.randf(rng, -1.0, 1.0) * kick)
    rng_states[i] = rng

    body[i, 0] = com[0]
    body[i, 1] = com[1]
    body[i, 2] = vel[0]
    body[i, 3] = vel[1]
    body[i, 4] = 0.0
    body[i, 5] = 0.0
    contacts[i, 0] = 0.0
    contacts[i, 1] = 0.0
    engine[i, 0] = 0.0
    engine[i, 1] = 0.0
    sleep_counter[i] = 0

    prev_shaping[i] = write_observation(params, i, com, vel, 0.0, 0.0, 0.0, 0.0, obs)


@wp.kernel(enable_backward=False)
def refresh_kernel(
    params: LanderParams,
    body: wp.array2d(dtype=wp.float32),
    contacts: wp.array2d(dtype=wp.float32),
    prev_shaping: wp.array(dtype=wp.float32),
    obs: wp.array2d(dtype=wp.float32),
):
    """Rewrite the observation and shaping baseline after set_state()."""
    i = wp.tid()
    com = wp.vec2(body[i, 0], body[i, 1])
    vel = wp.vec2(body[i, 2], body[i, 3])
    prev_shaping[i] = write_observation(
        params, i, com, vel, body[i, 4], body[i, 5], contacts[i, 0], contacts[i, 1], obs
    )


@wp.kernel(enable_backward=False)
def step_kernel(
    params: LanderParams,
    leg_points: wp.array(dtype=wp.vec2),
    hull_points: wp.array(dtype=wp.vec2),
    actions: wp.array(dtype=wp.int32),
    rng_states: wp.array(dtype=wp.uint32),
    terrain: wp.array2d(dtype=wp.float32),
    body: wp.array2d(dtype=wp.float32),
    contacts: wp.array2d(dtype=wp.float32),
    engine: wp.array2d(dtype=wp.float32),
    prev_shaping: wp.array(dtype=wp.float32),
    sleep_counter: wp.array(dtype=wp.int32),
    obs: wp.array2d(dtype=wp.float32),
    final_obs: wp.array2d(dtype=wp.float32),
    rewards: wp.array(dtype=wp.float32),
    terminated: wp.array(dtype=wp.float32),
    truncated: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    com = wp.vec2(body[i, 0], body[i, 1])
    vel = wp.vec2(body[i, 2], body[i, 3])
    angle = body[i, 4]
    omega = body[i, 5]

    # -- engines: impulses, exactly as Gymnasium applies them ---------------
    a = actions[i]
    tip = wp.vec2(wp.sin(angle), wp.cos(angle))
    side = wp.vec2(-tip[1], tip[0])
    rng = rng_states[i]
    d0 = wp.randf(rng, -1.0, 1.0) / 30.0
    d1 = wp.randf(rng, -1.0, 1.0) / 30.0
    rng_states[i] = rng

    origin = com - rotate(angle, params.com)
    m_power = float(0.0)
    s_power = float(0.0)
    side_dir = float(0.0)

    if a == 2:
        m_power = 1.0
        ox = tip[0] * (4.0 / 30.0 + 2.0 * d0) + side[0] * d1
        oy = -tip[1] * (4.0 / 30.0 + 2.0 * d0) - side[1] * d1
        point = origin + wp.vec2(ox, oy)
        impulse = wp.vec2(-ox * params.main_power * m_power, -oy * params.main_power * m_power)
        r = point - com
        vel += impulse / params.mass
        omega += (r[0] * impulse[1] - r[1] * impulse[0]) / params.inertia
    elif a == 1 or a == 3:
        s_power = 1.0
        side_dir = float(a - 2)
        ox = tip[0] * d0 + side[0] * (3.0 * d1 + side_dir * 12.0 / 30.0)
        oy = -tip[1] * d0 - side[1] * (3.0 * d1 + side_dir * 12.0 / 30.0)
        point = origin + wp.vec2(ox - tip[0] * 17.0 / 30.0, oy + tip[1] * 14.0 / 30.0)
        impulse = wp.vec2(-ox * params.side_power * s_power, -oy * params.side_power * s_power)
        r = point - com
        vel += impulse / params.mass
        omega += (r[0] * impulse[1] - r[1] * impulse[0]) / params.inertia

    # -- rigid-body integration with penalty contacts on the two legs -------
    h = params.dt / float(params.substeps)
    c0 = float(0.0)
    c1 = float(0.0)
    crash = float(0.0)
    for _s in range(params.substeps):
        force = wp.vec2(0.0, params.mass * params.gravity)
        torque = float(0.0)
        for k in range(2):
            p = com + rotate(angle, leg_points[k] - params.com)
            depth = ground_height(terrain, i, p[0], params.chunk_dx) - p[1]
            if depth > 0.0:
                if k == 0:
                    c0 = 1.0
                else:
                    c1 = 1.0
                n = ground_normal(terrain, i, p[0], params.chunk_dx)
                t = wp.vec2(-n[1], n[0])
                r = p - com
                point_vel = vel + omega * wp.vec2(-r[1], r[0])
                if wp.dot(point_vel, n) < -params.crash_speed:
                    crash = 1.0
                fn = wp.max(params.contact_k * depth - params.contact_c * wp.dot(point_vel, n), 0.0)
                ft = wp.clamp(
                    -params.contact_kt * wp.dot(point_vel, t), -params.contact_mu * fn, params.contact_mu * fn
                )
                f = fn * n + ft * t
                force += f
                torque += r[0] * f[1] - r[1] * f[0]
        vel += force / params.mass * h
        omega += torque / params.inertia * h
        com += vel * h
        angle += omega * h

    # -- the hull touching the ground is a crash too -------------------------
    for k in range(hull_points.shape[0]):
        p = com + rotate(angle, hull_points[k] - params.com)
        if p[1] < ground_height(terrain, i, p[0], params.chunk_dx):
            crash = 1.0

    # -- observation, reward, termination -----------------------------------
    shaping = write_observation(params, i, com, vel, angle, omega, c0, c1, final_obs)
    reward = shaping - prev_shaping[i] - 0.30 * m_power - 0.03 * s_power
    prev_shaping[i] = shaping

    # "came to rest": Box2D would put the body to sleep here (and any engine
    # impulse would wake it again)
    at_rest = 0
    if wp.length(vel) < params.sleep_v and wp.abs(omega) < params.sleep_w and m_power + s_power == 0.0:
        at_rest = sleep_counter[i] + 1
    sleep_counter[i] = at_rest

    term = float(0.0)
    if crash > 0.0 or wp.abs(final_obs[i, 0]) >= 1.0:
        term = 1.0
        reward = -100.0
    elif at_rest >= params.sleep_steps:
        term = 1.0
        reward = 100.0

    rewards[i] = reward
    terminated[i] = term
    truncated[i] = 0.0

    body[i, 0] = com[0]
    body[i, 1] = com[1]
    body[i, 2] = vel[0]
    body[i, 3] = vel[1]
    body[i, 4] = angle
    body[i, 5] = omega
    contacts[i, 0] = c0
    contacts[i, 1] = c1
    engine[i, 0] = m_power
    engine[i, 1] = side_dir * s_power

    for k in range(8):
        obs[i, k] = final_obs[i, k]


# ---------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------


class LunarLanderVectorEnv(WarpVecEnv):
    """``num_envs`` independent lunar landers stepped in lockstep.

    Actions: 0 = do nothing, 1 = fire left engine, 2 = fire main engine,
    3 = fire right engine.  Observation, reward and termination follow
    LunarLander-v3; see the module docstring for the physics deviations.
    """

    env_id = "lunarlander"
    obs_dim = spec.OBS_DIM
    num_actions = spec.NUM_ACTIONS

    def __init__(self, num_envs: int = 1, *, max_episode_steps: int = spec.MAX_EPISODE_STEPS, **kwargs):
        self.params = default_params()
        super().__init__(num_envs, max_episode_steps=max_episode_steps, **kwargs)

        n, d = self.num_envs, self.device
        self.body = wp.zeros((n, 6), dtype=wp.float32, device=d)
        self.contacts = wp.zeros((n, 2), dtype=wp.float32, device=d)
        self.engine = wp.zeros((n, 2), dtype=wp.float32, device=d)
        self.terrain = wp.zeros((n, spec.CHUNKS), dtype=wp.float32, device=d)
        self._raw_terrain = wp.zeros((n, spec.CHUNKS + 1), dtype=wp.float32, device=d)
        self.prev_shaping = wp.zeros(n, dtype=wp.float32, device=d)
        self.sleep_counter = wp.zeros(n, dtype=wp.int32, device=d)

        self.leg_points = wp.array(spec.LEG_POINTS, dtype=wp.vec2, device=d)
        self.hull_points = wp.array(spec.HULL_POINTS, dtype=wp.vec2, device=d)

    def observation_high(self) -> np.ndarray:
        # matches LunarLander-v3's declared observation space
        return np.array([2.5, 2.5, 10.0, 10.0, 2.0 * math.pi, 10.0, 1.0, 1.0], dtype=np.float32)

    def set_state(self, body, terrain=None) -> None:
        """Overwrite the rigid-body state (and optionally the terrain)."""
        self.body.assign(np.asarray(body, dtype=np.float32).reshape(self.num_envs, 6))
        if terrain is not None:
            self.terrain.assign(np.asarray(terrain, dtype=np.float32).reshape(self.num_envs, spec.CHUNKS))
        self.contacts.zero_()
        self.engine.zero_()
        self.sleep_counter.zero_()
        self.steps.zero_()
        self.ep_return.zero_()
        self.ep_length.zero_()
        wp.launch(
            refresh_kernel,
            dim=self.num_envs,
            inputs=[self.params, self.body, self.contacts, self.prev_shaping, self.obs],
            device=self.device,
        )

    def render_state(self) -> dict:
        return {
            "body": self.body.numpy(),
            "terrain": self.terrain.numpy(),
            "contacts": self.contacts.numpy(),
            "engine": self.engine.numpy(),
            "steps": self.steps.numpy(),
            "ep_return": self.ep_return.numpy(),
        }

    def _reset(self) -> None:
        wp.launch(
            reset_kernel,
            dim=self.num_envs,
            inputs=[
                self.params,
                self.needs_reset,
                self.rng_states,
                self._raw_terrain,
                self.terrain,
                self.body,
                self.contacts,
                self.engine,
                self.prev_shaping,
                self.sleep_counter,
                self.obs,
            ],
            device=self.device,
        )

    def _step(self, actions: wp.array) -> None:
        wp.launch(
            step_kernel,
            dim=self.num_envs,
            inputs=[
                self.params,
                self.leg_points,
                self.hull_points,
                actions,
                self.rng_states,
                self.terrain,
                self.body,
                self.contacts,
                self.engine,
                self.prev_shaping,
                self.sleep_counter,
                self.obs,
                self.final_obs,
                self.rewards,
                self.terminated,
                self.truncated,
            ],
            device=self.device,
        )
