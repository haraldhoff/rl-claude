"""LunarLander specification: Gymnasium's constants plus our rigid-body model.

Everything the agent sees comes from ``gymnasium/envs/box2d/lunar_lander.py``
(LunarLander-v3, discrete).  The dynamics constants below the divider are ours:
Box2D's constraint solver is replaced by a single rigid body -- hull plus welded
legs, with mass, centre of mass and inertia computed here from the very same
polygons and densities Box2D is handed -- touching the ground through two leg
contact points with a penalty spring-damper.

Deviations from Box2D, shared by both backends:

* rigid legs instead of sprung revolute joints, compliant contacts instead of
  hard constraints, friction 0.5 rather than Box2D's slippery 0.1;
* touching down faster than :data:`CRASH_SPEED` counts as a crash (our springy
  legs would otherwise absorb an unsurvivable slam);
* "came to rest" (:data:`SLEEP_V` / :data:`SLEEP_W` for :data:`SLEEP_STEPS`
  with no engine firing) replaces Box2D's sleep test for the +100 bonus;
* ``reset()`` skips Gymnasium's extra zero-action step; no wind, no continuous
  action space.
"""

from __future__ import annotations

import numpy as np

# --- Gymnasium's constants, verbatim ---------------------------------------

FPS = 50.0
SCALE = 30.0
MAIN_ENGINE_POWER = 13.0
SIDE_ENGINE_POWER = 0.6
INITIAL_RANDOM = 1000.0

LANDER_POLY = [(-14, +17), (-17, 0), (-17, -10), (+17, -10), (+17, 0), (+14, +17)]
LEG_AWAY = 20.0
LEG_DOWN = 18.0
LEG_W = 2.0
LEG_H = 8.0
SIDE_ENGINE_HEIGHT = 14.0
SIDE_ENGINE_AWAY = 12.0

VIEWPORT_W = 600.0
VIEWPORT_H = 400.0
WORLD_W = VIEWPORT_W / SCALE  # 20.0
WORLD_H = VIEWPORT_H / SCALE  # 13.33
CHUNKS = 11
HELIPAD_Y = WORLD_H / 4.0

GRAVITY = -10.0
HULL_DENSITY = 5.0
LEG_DENSITY = 1.0

OBS_DIM = 8
NUM_ACTIONS = 4
MAX_EPISODE_STEPS = 1000

# --- our rigid-body / contact model ----------------------------------------

CONTACT_STIFFNESS = 3000.0
CONTACT_DAMPING = 120.0
CONTACT_FRICTION = 0.5
CONTACT_TANGENT_DAMPING = 200.0
SUBSTEPS = 8
CRASH_SPEED = 4.0

SLEEP_V = 0.06  # m/s
SLEEP_W = 0.06  # rad/s
SLEEP_STEPS = 25  # 0.5 s at 50 Hz, matching Box2D's b2_timeToSleep


def polygon_properties(points, density: float):
    """Mass, centroid and inertia about the centroid of a uniform polygon."""
    pts = np.asarray(points, dtype=np.float64)
    x, y = pts[:, 0], pts[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    area = 0.5 * cross.sum()
    cx = (cross * (x + x1)).sum() / (6.0 * area)
    cy = (cross * (y + y1)).sum() / (6.0 * area)
    # second moment about the origin, then shifted to the centroid
    inertia = (cross * (x * x + x * x1 + x1 * x1 + y * y + y * y1 + y1 * y1)).sum() / 12.0
    mass = density * abs(area)
    inertia = density * abs(inertia) - mass * (cx * cx + cy * cy)
    return mass, np.array([cx, cy]), inertia


def _box(cx: float, cy: float, half_w: float, half_h: float):
    return [
        (cx - half_w, cy - half_h),
        (cx + half_w, cy - half_h),
        (cx + half_w, cy + half_h),
        (cx - half_w, cy + half_h),
    ]


def body_properties():
    """Rigid-body properties of the hull plus the two welded legs."""
    hull = [(px / SCALE, py / SCALE) for px, py in LANDER_POLY]
    parts = [(hull, HULL_DENSITY)]
    for sign in (-1.0, 1.0):
        parts.append((_box(sign * LEG_AWAY / SCALE, -LEG_DOWN / SCALE, LEG_W / SCALE, LEG_H / SCALE), LEG_DENSITY))
    masses, centroids, inertias = [], [], []
    for points, density in parts:
        m, c, i = polygon_properties(points, density)
        masses.append(m)
        centroids.append(c)
        inertias.append(i)
    mass = float(sum(masses))
    com = sum(m * c for m, c in zip(masses, centroids)) / mass
    inertia = float(sum(i + m * float(np.dot(c - com, c - com)) for m, c, i in zip(masses, centroids, inertias)))
    return mass, com.astype(np.float32), inertia


MASS, COM, INERTIA = body_properties()

# contact points, in body-origin coordinates
LEG_POINTS = np.array(
    [[-LEG_AWAY / SCALE, -(LEG_DOWN + LEG_H) / SCALE], [LEG_AWAY / SCALE, -(LEG_DOWN + LEG_H) / SCALE]],
    dtype=np.float32,
)
HULL_POINTS = np.array([[px / SCALE, py / SCALE] for px, py in LANDER_POLY], dtype=np.float32)

CHUNK_DX = WORLD_W / (CHUNKS - 1)


def observation(origin, velocity, angle, omega, contacts):
    """Gymnasium's 8-dim observation, from the body origin (not the centre of mass)."""
    half_w, half_h = WORLD_W / 2.0, WORLD_H / 2.0
    return np.stack(
        [
            (origin[..., 0] - half_w) / half_w,
            (origin[..., 1] - (HELIPAD_Y + LEG_DOWN / SCALE)) / half_h,
            velocity[..., 0] * half_w / FPS,
            velocity[..., 1] * half_h / FPS,
            angle,
            20.0 * omega / FPS,
            contacts[..., 0],
            contacts[..., 1],
        ],
        axis=-1,
    )


def shaping(obs):
    """Gymnasium's reward-shaping potential, from the observation."""
    obs = np.asarray(obs)
    return (
        -100.0 * np.sqrt(obs[..., 0] ** 2 + obs[..., 1] ** 2)
        - 100.0 * np.sqrt(obs[..., 2] ** 2 + obs[..., 3] ** 2)
        - 100.0 * np.abs(obs[..., 4])
        + 10.0 * obs[..., 6]
        + 10.0 * obs[..., 7]
    )
