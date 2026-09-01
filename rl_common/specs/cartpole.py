"""CartPole-v1 specification.

Constants only -- both backends implement the same update rule from these, and
`tests/test_cartpole_env.py` checks each of them against Gymnasium.
"""

from __future__ import annotations

import math

GRAVITY = 9.8
MASSCART = 1.0
MASSPOLE = 0.1
TOTAL_MASS = MASSCART + MASSPOLE
LENGTH = 0.5  # actually half the pole's length
POLEMASS_LENGTH = MASSPOLE * LENGTH
FORCE_MAG = 10.0
TAU = 0.02  # seconds between state updates
THETA_THRESHOLD = 12.0 * 2.0 * math.pi / 360.0
X_THRESHOLD = 2.4
RESET_BOUND = 0.05

OBS_DIM = 4
NUM_ACTIONS = 2
MAX_EPISODE_STEPS = 500
