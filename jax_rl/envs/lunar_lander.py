"""LunarLander in JAX.

The same model as the Warp backend, from the same constants and the same
rigid-body properties in :mod:`rl_common.specs.lunar_lander`: Gymnasium's task
definition (terrain, observation, shaping reward, engine impulses, termination)
on a single rigid body with penalty leg contacts, integrated with 8 substeps.
``tests/test_backend_parity.py`` checks the two implementations against each
other step for step.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from rl_common.specs import lunar_lander as spec

from ..vec_env import JaxVecEnv

_COM = jnp.asarray(spec.COM, dtype=jnp.float32)
_LEG_POINTS = jnp.asarray(spec.LEG_POINTS, dtype=jnp.float32)
_HULL_POINTS = jnp.asarray(spec.HULL_POINTS, dtype=jnp.float32)
_DT = jnp.float32(1.0 / spec.FPS)
_H = jnp.float32(1.0 / spec.FPS / spec.SUBSTEPS)


@struct.dataclass
class LanderState:
    com: jax.Array  # (2,) centre of mass position
    vel: jax.Array  # (2,)
    angle: jax.Array  # scalar
    omega: jax.Array  # scalar
    contacts: jax.Array  # (2,) leg contact flags
    engine: jax.Array  # (2,) [main on, side direction]
    terrain: jax.Array  # (CHUNKS,)
    prev_shaping: jax.Array  # scalar
    sleep: jax.Array  # scalar int32


def _rotate(angle, v):
    c, s = jnp.cos(angle), jnp.sin(angle)
    return jnp.stack([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def _ground(terrain, x):
    """Height and upward normal of the terrain polyline at ``x``."""
    idx = jnp.clip(jnp.floor(x / spec.CHUNK_DX).astype(jnp.int32), 0, spec.CHUNKS - 2)
    t = jnp.clip(x / spec.CHUNK_DX - idx.astype(jnp.float32), 0.0, 1.0)
    y0, y1 = terrain[idx], terrain[idx + 1]
    height = y0 * (1.0 - t) + y1 * t
    normal = jnp.stack([-(y1 - y0), jnp.float32(spec.CHUNK_DX)])
    return height, normal / jnp.linalg.norm(normal)


def _observation(com, vel, angle, omega, contacts):
    """Gymnasium's 8-dim observation (from the body origin, not the centre of mass)."""
    origin = com - _rotate(angle, _COM)
    half_w, half_h = spec.WORLD_W * 0.5, spec.WORLD_H * 0.5
    return jnp.stack(
        [
            (origin[0] - half_w) / half_w,
            (origin[1] - (spec.HELIPAD_Y + spec.LEG_DOWN / spec.SCALE)) / half_h,
            vel[0] * half_w / spec.FPS,
            vel[1] * half_h / spec.FPS,
            angle,
            20.0 * omega / spec.FPS,
            contacts[0],
            contacts[1],
        ]
    ).astype(jnp.float32)


def _shaping(obs):
    return (
        -100.0 * jnp.sqrt(obs[0] * obs[0] + obs[1] * obs[1])
        - 100.0 * jnp.sqrt(obs[2] * obs[2] + obs[3] * obs[3])
        - 100.0 * jnp.abs(obs[4])
        + 10.0 * obs[6]
        + 10.0 * obs[7]
    )


class LunarLanderEnv:
    """Functional lunar lander: ``reset(key)`` and ``step(key, state, action)``."""

    obs_dim = spec.OBS_DIM
    num_actions = spec.NUM_ACTIONS

    def observation_high(self) -> np.ndarray:
        return np.array([2.5, 2.5, 10.0, 10.0, 2.0 * np.pi, 10.0, 1.0, 1.0], dtype=np.float32)

    def observation_low(self) -> np.ndarray:
        return -self.observation_high()

    # -- reset --------------------------------------------------------------

    def reset(self, key: jax.Array):
        terrain_key, kick_key = jax.random.split(key)

        # CHUNKS+1 random heights, a flat helipad in the middle, then the same
        # 0.33-weighted smoothing Gymnasium uses (0.33, not 1/3)
        raw = jax.random.uniform(terrain_key, (spec.CHUNKS + 1,), jnp.float32, 0.0, spec.WORLD_H * 0.5)
        raw = raw.at[3:8].set(spec.HELIPAD_Y)
        left = jnp.roll(raw, 1)[: spec.CHUNKS]  # raw[j-1], wrapping like Python's -1
        terrain = 0.33 * (left + raw[: spec.CHUNKS] + raw[1 : spec.CHUNKS + 1])

        # the lander starts at the top centre with a random shove, which
        # Gymnasium applies as a one-frame force to the centre of mass
        kick = spec.INITIAL_RANDOM / spec.FPS / spec.MASS
        vel = jax.random.uniform(kick_key, (2,), jnp.float32, -1.0, 1.0) * kick
        com = jnp.array([spec.WORLD_W * 0.5, spec.WORLD_H], jnp.float32) + _COM
        angle = jnp.float32(0.0)
        omega = jnp.float32(0.0)
        contacts = jnp.zeros(2, jnp.float32)

        obs = _observation(com, vel, angle, omega, contacts)
        state = LanderState(
            com=com,
            vel=vel,
            angle=angle,
            omega=omega,
            contacts=contacts,
            engine=jnp.zeros(2, jnp.float32),
            terrain=terrain,
            prev_shaping=_shaping(obs),
            sleep=jnp.int32(0),
        )
        return state, obs

    # -- step ---------------------------------------------------------------

    def step(self, key: jax.Array, state: LanderState, action: jax.Array):
        com, vel, angle, omega = state.com, state.vel, state.angle, state.omega
        terrain = state.terrain

        # -- engines: impulses, exactly as Gymnasium applies them ------------
        tip = jnp.stack([jnp.sin(angle), jnp.cos(angle)])
        side = jnp.stack([-tip[1], tip[0]])
        d0, d1 = jax.random.uniform(key, (2,), jnp.float32, -1.0, 1.0) / 30.0

        origin = com - _rotate(angle, _COM)
        m_power = jnp.where(action == 2, 1.0, 0.0).astype(jnp.float32)
        s_power = jnp.where((action == 1) | (action == 3), 1.0, 0.0).astype(jnp.float32)
        side_dir = jnp.where(s_power > 0, action.astype(jnp.float32) - 2.0, 0.0)

        # main engine
        m_ox = tip[0] * (4.0 / 30.0 + 2.0 * d0) + side[0] * d1
        m_oy = -tip[1] * (4.0 / 30.0 + 2.0 * d0) - side[1] * d1
        m_point = origin + jnp.stack([m_ox, m_oy])
        m_impulse = jnp.stack([-m_ox, -m_oy]) * spec.MAIN_ENGINE_POWER * m_power

        # side engine
        s_ox = tip[0] * d0 + side[0] * (3.0 * d1 + side_dir * 12.0 / 30.0)
        s_oy = -tip[1] * d0 - side[1] * (3.0 * d1 + side_dir * 12.0 / 30.0)
        s_point = origin + jnp.stack([s_ox - tip[0] * 17.0 / 30.0, s_oy + tip[1] * 14.0 / 30.0])
        s_impulse = jnp.stack([-s_ox, -s_oy]) * spec.SIDE_ENGINE_POWER * s_power

        for point, impulse in ((m_point, m_impulse), (s_point, s_impulse)):
            r = point - com
            vel = vel + impulse / spec.MASS
            omega = omega + (r[0] * impulse[1] - r[1] * impulse[0]) / spec.INERTIA

        # -- rigid-body integration with penalty contacts on the two legs ----
        contacts = jnp.zeros(2, jnp.float32)
        crash = jnp.float32(0.0)

        for _ in range(spec.SUBSTEPS):
            force = jnp.array([0.0, spec.MASS * spec.GRAVITY], jnp.float32)
            torque = jnp.float32(0.0)
            for k in range(2):
                p = com + _rotate(angle, _LEG_POINTS[k] - _COM)
                height, normal = _ground(terrain, p[0])
                depth = height - p[1]
                touching = depth > 0.0

                tangent = jnp.stack([-normal[1], normal[0]])
                r = p - com
                point_vel = vel + omega * jnp.stack([-r[1], r[0]])
                v_n = jnp.dot(point_vel, normal)
                v_t = jnp.dot(point_vel, tangent)

                fn = jnp.maximum(spec.CONTACT_STIFFNESS * depth - spec.CONTACT_DAMPING * v_n, 0.0)
                ft = jnp.clip(
                    -spec.CONTACT_TANGENT_DAMPING * v_t, -spec.CONTACT_FRICTION * fn, spec.CONTACT_FRICTION * fn
                )
                f = jnp.where(touching, fn * normal + ft * tangent, 0.0)

                contacts = contacts.at[k].max(jnp.where(touching, 1.0, 0.0))
                crash = jnp.maximum(crash, jnp.where(touching & (v_n < -spec.CRASH_SPEED), 1.0, 0.0))
                force = force + f
                torque = torque + r[0] * f[1] - r[1] * f[0]

            vel = vel + force / spec.MASS * _H
            omega = omega + torque / spec.INERTIA * _H
            com = com + vel * _H
            angle = angle + omega * _H

        # -- the hull touching the ground is a crash too ---------------------
        for k in range(spec.HULL_POINTS.shape[0]):
            p = com + _rotate(angle, _HULL_POINTS[k] - _COM)
            height, _ = _ground(terrain, p[0])
            crash = jnp.maximum(crash, jnp.where(p[1] < height, 1.0, 0.0))

        # -- observation, reward, termination --------------------------------
        obs = _observation(com, vel, angle, omega, contacts)
        shaping = _shaping(obs)
        reward = shaping - state.prev_shaping - 0.30 * m_power - 0.03 * s_power

        # "came to rest": Box2D would put the body to sleep here (and any
        # engine impulse would wake it again)
        resting = (
            (jnp.linalg.norm(vel) < spec.SLEEP_V) & (jnp.abs(omega) < spec.SLEEP_W) & (m_power + s_power == 0.0)
        )
        sleep = jnp.where(resting, state.sleep + 1, 0)

        out_of_bounds = jnp.abs(obs[0]) >= 1.0
        failed = (crash > 0.0) | out_of_bounds
        landed = (~failed) & (sleep >= spec.SLEEP_STEPS)

        terminated = jnp.where(failed | landed, 1.0, 0.0).astype(jnp.float32)
        reward = jnp.where(failed, -100.0, jnp.where(landed, 100.0, reward)).astype(jnp.float32)

        state = LanderState(
            com=com,
            vel=vel,
            angle=angle,
            omega=omega,
            contacts=contacts,
            engine=jnp.stack([m_power, side_dir * s_power]),
            terrain=terrain,
            prev_shaping=shaping,
            sleep=sleep,
        )
        return state, obs, reward, terminated


class LunarLanderVectorEnv(JaxVecEnv):
    """``num_envs`` independent lunar landers stepped in lockstep."""

    env_id = "lunarlander"
    obs_dim = spec.OBS_DIM
    num_actions = spec.NUM_ACTIONS
    env_cls = LunarLanderEnv

    def __init__(self, num_envs: int = 1, *, max_episode_steps: int = spec.MAX_EPISODE_STEPS, **kwargs):
        super().__init__(num_envs, max_episode_steps=max_episode_steps, **kwargs)

    def render_state(self) -> dict:
        physics = self.state.physics
        body = jnp.concatenate(
            [physics.com, physics.vel, physics.angle[:, None], physics.omega[:, None]], axis=1
        )
        return {
            "body": np.asarray(body),
            "terrain": np.asarray(physics.terrain),
            "contacts": np.asarray(physics.contacts),
            "engine": np.asarray(physics.engine),
            "steps": np.asarray(self.state.steps),
            "ep_return": np.asarray(self.state.ep_return),
        }

    def set_state(self, body, terrain) -> None:
        """Overwrite the rigid-body state and terrain (used by the parity tests)."""
        body = jnp.asarray(np.asarray(body, dtype=np.float32).reshape(self.num_envs, 6))
        terrain = jnp.asarray(np.asarray(terrain, dtype=np.float32).reshape(self.num_envs, spec.CHUNKS))
        contacts = jnp.zeros((self.num_envs, 2), jnp.float32)
        obs = jax.vmap(_observation)(body[:, 0:2], body[:, 2:4], body[:, 4], body[:, 5], contacts)
        physics = LanderState(
            com=body[:, 0:2],
            vel=body[:, 2:4],
            angle=body[:, 4],
            omega=body[:, 5],
            contacts=contacts,
            engine=jnp.zeros((self.num_envs, 2), jnp.float32),
            terrain=terrain,
            prev_shaping=jax.vmap(_shaping)(obs),
            sleep=jnp.zeros(self.num_envs, jnp.int32),
        )
        self.state = self.state.replace(
            physics=physics,
            obs=obs,
            steps=jnp.zeros_like(self.state.steps),
            ep_return=jnp.zeros_like(self.state.ep_return),
        )
