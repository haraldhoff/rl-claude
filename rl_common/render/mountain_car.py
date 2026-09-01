"""Mountain Car renderer: hill, flag and the tilted car."""

from __future__ import annotations

import math

import numpy as np

from ..specs import mountain_car as spec
from .base import TiledRenderer

_HILL = (60, 60, 70)
_GROUND = (215, 215, 225)
_CAR = (20, 20, 30)
_WHEEL = (128, 128, 128)
_FLAG_POLE = (20, 20, 30)
_FLAG = (204, 204, 0)

_CAR_W, _CAR_H = 40.0, 20.0
_CLEARANCE = 10.0


class MountainCarRenderer(TiledRenderer):
    def setup(self) -> None:
        self.scale = self.tile_w / (spec.MAX_POSITION - spec.MIN_POSITION)
        self.xs = np.linspace(spec.MIN_POSITION, spec.MAX_POSITION, 100)
        self.ys = spec.height(self.xs)

    def stats_label(self, index: int) -> str:
        position, velocity = self.state["obs"][index]
        return (
            f"env {index}  t={int(self.steps[index]):3d}  R={self.returns[index]:.0f}  "
            f"x={position:+.2f}  v={velocity:+.3f}"
        )

    def _to_screen(self, origin, x, y) -> tuple[float, float]:
        # y is measured up from the bottom of the tile, in "height" units;
        # plain floats because pygame rejects numpy scalars in point sequences
        return (
            float(origin[0] + (x - spec.MIN_POSITION) * self.scale),
            float(origin[1] + self.tile_h - y * self.scale),
        )

    def draw_tile(self, origin: tuple[float, float], index: int) -> None:
        import pygame

        surf = self.surface
        hill = [self._to_screen(origin, x, y) for x, y in zip(self.xs, self.ys)]
        pygame.draw.polygon(
            surf,
            _GROUND,
            [
                self._to_screen(origin, spec.MIN_POSITION, 0.0),
                *hill,
                self._to_screen(origin, spec.MAX_POSITION, 0.0),
            ],
        )
        pygame.draw.lines(surf, _HILL, False, hill, max(1, int(2 * self.k)))

        # flag on the goal
        flag_base = self._to_screen(origin, spec.GOAL_POSITION, float(spec.height(spec.GOAL_POSITION)))
        flag_top = (flag_base[0], flag_base[1] - 50 * self.k)
        pygame.draw.line(surf, _FLAG_POLE, flag_base, flag_top, max(1, int(2 * self.k)))
        pygame.draw.polygon(
            surf,
            _FLAG,
            [flag_top, (flag_top[0] + 25 * self.k, flag_top[1] + 10 * self.k), (flag_top[0], flag_top[1] + 20 * self.k)],
        )

        # car: a box rotated to the slope, sitting `clearance` above the curve
        position = float(self.state["obs"][index, 0])
        angle = math.cos(3.0 * position)  # Gymnasium rotates by cos(3x) directly
        c, s = math.cos(angle), math.sin(angle)
        base = self._to_screen(origin, position, float(spec.height(position)))
        base = (base[0], base[1] - _CLEARANCE * self.k)

        def place(px, py):
            px, py = px * self.k, py * self.k
            return (base[0] + c * px - s * py, base[1] - (s * px + c * py))

        body = [
            place(x, y)
            for x, y in ((-_CAR_W / 2, 0.0), (-_CAR_W / 2, _CAR_H), (_CAR_W / 2, _CAR_H), (_CAR_W / 2, 0.0))
        ]
        pygame.draw.polygon(surf, _CAR, body)
        for wheel_x in (-_CAR_W / 4, _CAR_W / 4):
            pygame.draw.circle(surf, _WHEEL, place(wheel_x, 0.0), max(2.0, _CAR_H / 2.5 * self.k))
