"""MountainCar-v0 specification."""

from __future__ import annotations

import numpy as np

MIN_POSITION = -1.2
MAX_POSITION = 0.6
MAX_SPEED = 0.07
GOAL_POSITION = 0.5
GOAL_VELOCITY = 0.0
FORCE = 0.001
GRAVITY = 0.0025
RESET_LOW = -0.6
RESET_HIGH = -0.4

OBS_DIM = 2
NUM_ACTIONS = 3
MAX_EPISODE_STEPS = 200  # physics steps, as in MountainCar-v0

# the agent decides every ACTION_REPEAT physics steps while training; with
# uniform random actions the flag is unreachable (0 of 4096 episodes), with
# held actions the car resonates up the hill
ACTION_REPEAT = 8


def height(x):
    """Terrain profile, as Gymnasium draws it."""
    return np.sin(3.0 * np.asarray(x)) * 0.45 + 0.55
