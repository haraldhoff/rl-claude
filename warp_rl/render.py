"""Shared pygame rendering machinery.

:class:`TiledRenderer` owns the window (or the off-screen surface), the grid
layout when several environments are shown at once, the per-tile HUD and the
event handling.  An environment's renderer subclasses it and implements just
two methods: :meth:`fetch` (pull the device state it needs to the host, once
per frame) and :meth:`draw_tile` (draw one environment).
"""

from __future__ import annotations

import math

import numpy as np

pygame = None  # imported lazily: the core has no display dependency


def _require_pygame():
    """Import pygame on first use and cache it in the module namespace."""
    global pygame
    if pygame is None:
        try:
            import pygame as _pygame
        except ImportError as exc:  # pragma: no cover
            raise ImportError("rendering requires pygame (pip install pygame)") from exc
        pygame = _pygame
    return pygame


BACKGROUND = (255, 255, 255)
TEXT = (60, 60, 60)
BORDER = (200, 200, 200)

# every geometric constant in a renderer is written for a 600px-wide tile and
# scaled by ``self.k``, so a grid of small tiles keeps the same proportions
REF_WIDTH = 600.0
REF_HEIGHT = 400.0


class TiledRenderer:
    """Renders one or more environments of a :class:`WarpVecEnv` side by side."""

    default_tile_size = (600, 400)
    background = BACKGROUND

    def __init__(
        self,
        env,
        *,
        mode: str = "human",
        num_render: int | None = None,
        cols: int | None = None,
        tile_size: tuple[int, int] | None = None,
        fps: int = 50,
        show_stats: bool = True,
        caption: str = "Warp RL",
    ):
        _require_pygame()
        if mode not in ("human", "rgb_array"):
            raise ValueError(f"unknown render mode: {mode}")

        self.env = env
        self.mode = mode
        self.fps = fps
        self.show_stats = show_stats
        self.n = min(num_render or env.num_envs, env.num_envs)
        self.cols = cols or max(1, math.ceil(math.sqrt(self.n)))
        self.rows = math.ceil(self.n / self.cols)
        self.tile_w, self.tile_h = tile_size or self.default_tile_size
        self.width = self.cols * self.tile_w
        self.height = self.rows * self.tile_h
        self.k = self.tile_w / REF_WIDTH  # geometry scale factor

        pygame.init()
        if mode == "human":
            pygame.display.set_caption(caption)
            self.surface = pygame.display.set_mode((self.width, self.height))
            self.clock = pygame.time.Clock()
        else:
            self.surface = pygame.Surface((self.width, self.height))
            self.clock = None
        self.font = pygame.font.SysFont("consolas,couriernew,monospace", max(11, int(16 * self.k)))
        self.closed = False
        self.setup()

    # -- subclass hooks -----------------------------------------------------

    def setup(self) -> None:
        """Called once, after the surface exists (compute scales here)."""

    def fetch(self) -> None:
        """Copy whatever device arrays the frame needs to the host."""
        self.states = self.env.obs.numpy()
        self.steps = self.env.steps.numpy()
        self.returns = self.env.ep_return.numpy()

    def draw_tile(self, origin: tuple[float, float], index: int) -> None:
        raise NotImplementedError

    def stats_label(self, index: int) -> str:
        return f"env {index}  t={int(self.steps[index]):4d}  R={self.returns[index]:.0f}"

    # -- helpers for subclasses ---------------------------------------------

    def label(self, origin: tuple[float, float], text: str, line: int = 0) -> None:
        surf = self.font.render(text, True, TEXT)
        self.surface.blit(surf, (origin[0] + 8 * self.k, origin[1] + (6 + line * 18) * self.k))

    # -- api ----------------------------------------------------------------

    def render(self) -> np.ndarray | None:
        """Draw the current state of every rendered environment.

        Returns an ``(H, W, 3)`` uint8 frame in ``rgb_array`` mode, ``None`` in
        ``human`` mode (or once the window has been closed).
        """
        if self.closed:
            return None
        if self.mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    self.close()
                    return None

        self.fetch()
        self.surface.fill(self.background)
        for i in range(self.n):
            origin = ((i % self.cols) * self.tile_w, (i // self.cols) * self.tile_h)
            self.draw_tile(origin, i)
            if self.show_stats:
                self.label(origin, self.stats_label(i))
            if self.n > 1:
                pygame.draw.rect(self.surface, BORDER, pygame.Rect(*origin, self.tile_w, self.tile_h), 1)

        if self.mode == "human":
            pygame.display.flip()
            self.clock.tick(self.fps)
            return None
        return np.transpose(pygame.surfarray.array3d(self.surface), (1, 0, 2))

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            pygame.quit()
