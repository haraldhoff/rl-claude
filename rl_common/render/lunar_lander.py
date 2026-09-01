"""Lunar Lander renderer: moon surface, helipad, lander and exhaust."""

from __future__ import annotations

import math

import numpy as np

from ..specs import lunar_lander as spec
from .base import TiledRenderer

_SKY = (5, 5, 15)
_MOON = (140, 140, 150)
_MOON_EDGE = (255, 255, 255)
_LANDER = (128, 102, 230)
_LANDER_EDGE = (90, 70, 170)
_LEG = (180, 165, 235)
_FLAG = (204, 204, 0)
_FLAME = (255, 160, 40)
_CONTACT = (120, 230, 140)
_PAD = (90, 90, 110)


class LunarLanderRenderer(TiledRenderer):
    background = _SKY

    def setup(self) -> None:
        self.scale = self.tile_w / spec.WORLD_W  # px per metre
        self.chunk_x = np.linspace(0.0, spec.WORLD_W, spec.CHUNKS)

    def stats_label(self, index: int) -> str:
        x, y, vx, vy, angle, _ = self.state["body"][index]
        legs = int(self.state["contacts"][index].sum())
        return (
            f"env {index}  t={int(self.steps[index]):4d}  R={self.returns[index]:+7.1f}  "
            f"v=({vx:+.1f},{vy:+.1f})  th={math.degrees(angle):+5.1f}  legs={legs}"
        )

    def _to_screen(self, origin, x, y) -> tuple[float, float]:
        # plain floats: pygame rejects numpy scalars in point sequences
        return (float(origin[0] + x * self.scale), float(origin[1] + self.tile_h - y * self.scale))

    def draw_tile(self, origin: tuple[float, float], index: int) -> None:
        import pygame

        surf = self.surface
        heights = self.state["terrain"][index]

        # moon surface
        ground = [self._to_screen(origin, x, h) for x, h in zip(self.chunk_x, heights)]
        polygon = [self._to_screen(origin, 0.0, 0.0), *ground, self._to_screen(origin, spec.WORLD_W, 0.0)]
        pygame.draw.polygon(surf, _MOON, polygon)
        pygame.draw.lines(surf, _MOON_EDGE, False, ground, max(1, int(2 * self.k)))

        # helipad: flat chunks 4..6, flagged at both ends
        pad_left = self.chunk_x[spec.CHUNKS // 2 - 1]
        pad_right = self.chunk_x[spec.CHUNKS // 2 + 1]
        pad_y = float(heights[spec.CHUNKS // 2])
        pygame.draw.line(
            surf,
            _PAD,
            self._to_screen(origin, pad_left, pad_y),
            self._to_screen(origin, pad_right, pad_y),
            max(2, int(4 * self.k)),
        )
        for fx in (pad_left, pad_right):
            base = self._to_screen(origin, fx, pad_y)
            top = self._to_screen(origin, fx, pad_y + 1.0)
            pygame.draw.line(surf, _MOON_EDGE, base, top, max(1, int(2 * self.k)))
            pygame.draw.polygon(
                surf, _FLAG, [top, (top[0] + 12 * self.k, top[1] + 6 * self.k), (top[0], top[1] + 12 * self.k)]
            )

        # lander
        x, y, _, _, angle, _ = self.state["body"][index]
        com = np.array([x, y], dtype=np.float64)
        c, s = math.cos(angle), math.sin(angle)

        def to_world(point):
            local = np.asarray(point, dtype=np.float64) - spec.COM
            return com + np.array([c * local[0] - s * local[1], s * local[0] + c * local[1]])

        hull = [self._to_screen(origin, *to_world(p)) for p in spec.HULL_POINTS]
        pygame.draw.polygon(surf, _LANDER, hull)
        pygame.draw.polygon(surf, _LANDER_EDGE, hull, max(1, int(2 * self.k)))

        contacts = self.state["contacts"][index]
        for leg in range(2):
            sign = -1.0 if leg == 0 else 1.0
            hip = to_world((sign * spec.LEG_AWAY / spec.SCALE, -spec.LEG_DOWN / spec.SCALE + spec.LEG_H / spec.SCALE))
            foot = to_world(spec.LEG_POINTS[leg])
            colour = _CONTACT if contacts[leg] > 0 else _LEG
            pygame.draw.line(
                surf, colour, self._to_screen(origin, *hip), self._to_screen(origin, *foot), max(2, int(4 * self.k))
            )

        # exhaust
        main, side = self.state["engine"][index]
        if main > 0:
            nozzle = to_world((0.0, -spec.LEG_DOWN / spec.SCALE))
            plume = to_world((0.0, -spec.LEG_DOWN / spec.SCALE - 0.55))
            pygame.draw.line(
                surf, _FLAME, self._to_screen(origin, *nozzle), self._to_screen(origin, *plume), max(2, int(6 * self.k))
            )
        if side != 0:
            direction = -1.0 if side > 0 else 1.0
            height = spec.SIDE_ENGINE_HEIGHT / spec.SCALE - spec.LEG_DOWN / spec.SCALE
            nozzle = to_world((direction * 17.0 / spec.SCALE, height))
            plume = to_world((direction * (17.0 / spec.SCALE + 0.4), height))
            pygame.draw.line(
                surf, _FLAME, self._to_screen(origin, *nozzle), self._to_screen(origin, *plume), max(2, int(4 * self.k))
            )
