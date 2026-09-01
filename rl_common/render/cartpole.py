"""CartPole renderer: the classic-control picture, from ``render_state()``."""

from __future__ import annotations

import math

from ..specs import cartpole as spec
from .base import TiledRenderer

_CART_W, _CART_H = 50.0, 30.0
_POLE_W = 10.0
_CART_Y = 300.0  # cart centre line, measured from the top of a 400px tile

_CART = (0, 0, 0)
_POLE = (202, 152, 101)
_AXLE = (129, 132, 203)
_TRACK = (0, 0, 0)
_LIMIT = (220, 90, 90)


class CartPoleRenderer(TiledRenderer):
    def setup(self) -> None:
        # a little margin beyond +/- x_threshold so the limit markers are visible
        self.scale = self.tile_w / (spec.X_THRESHOLD * 2.0 * 1.15)
        self.pole_len = self.scale * (2.0 * spec.LENGTH)
        self.cart_y = _CART_Y * (self.tile_h / 400.0)

    def stats_label(self, index: int) -> str:
        x, _, theta, _ = self.state["obs"][index]
        return (
            f"env {index}  t={int(self.steps[index]):3d}  R={self.returns[index]:.0f}  "
            f"x={x:+.2f}  th={math.degrees(theta):+5.1f}"
        )

    def draw_tile(self, origin: tuple[float, float], index: int) -> None:
        import pygame

        ox, oy = origin
        surf = self.surface
        x, _, theta, _ = (float(v) for v in self.state["obs"][index])

        cart_x = ox + self.tile_w / 2.0 + x * self.scale
        cart_y = oy + self.cart_y
        cart_w, cart_h = _CART_W * self.k, _CART_H * self.k
        pole_w = max(2.0, _POLE_W * self.k)

        # track and the +/- x_threshold limits that end an episode
        pygame.draw.line(surf, _TRACK, (ox, cart_y), (ox + self.tile_w, cart_y), max(1, int(2 * self.k)))
        for sign in (-1.0, 1.0):
            lx = ox + self.tile_w / 2.0 + sign * spec.X_THRESHOLD * self.scale
            pygame.draw.line(
                surf, _LIMIT, (lx, cart_y - 20 * self.k), (lx, cart_y + 20 * self.k), max(1, int(2 * self.k))
            )

        pygame.draw.rect(surf, _CART, pygame.Rect(cart_x - cart_w / 2, cart_y - cart_h / 2, cart_w, cart_h))

        # pole as a rotated quad: +theta tilts it to the right
        axle = (cart_x, cart_y - cart_h / 4.0)
        along = (math.sin(theta), -math.cos(theta))
        across = (math.cos(theta) * pole_w / 2.0, math.sin(theta) * pole_w / 2.0)
        tip = (axle[0] + along[0] * self.pole_len, axle[1] + along[1] * self.pole_len)
        pygame.draw.polygon(
            surf,
            _POLE,
            [
                (axle[0] - across[0], axle[1] - across[1]),
                (axle[0] + across[0], axle[1] + across[1]),
                (tip[0] + across[0], tip[1] + across[1]),
                (tip[0] - across[0], tip[1] - across[1]),
            ],
        )
        pygame.draw.circle(surf, _AXLE, axle, pole_w / 2.0)
